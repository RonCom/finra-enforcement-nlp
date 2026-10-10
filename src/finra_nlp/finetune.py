"""Fine-tuned transformer (spec, "Classifier", H1): a small encoder with a multi-label head, trained on the
train split's masked summaries and scored on the validation split with the baseline's metrics.

Default model: answerdotai/ModernBERT-base (8,192-token context; summaries are cut at --max-len tokens and the
share cut is reported). Training uses mixed precision on a GPU, AdamW with linear warmup and decay, a loss that
weights each series' positive cases by sqrt(negatives / positives) so rare series aren't pushed under the
threshold, and keeps
the epoch with the best macro-F1 over series with 10+ validation cases. Writes reports/finetune_validation.md,
model.finetune_validation (each validation case's probabilities) and the best model to data/models/finetune/.
The test split is not read.

Install (once; the CUDA build of torch, into the project's environment):
    uv pip install torch --index-url https://download.pytorch.org/whl/cu126
    uv pip install "transformers>=4.48"

Usage:
    uv run python -m finra_nlp.finetune --epochs 1 --limit-train 200   # check memory and speed first
    uv run python -m finra_nlp.finetune                                # seed 0; --seed 1, --seed 2 for the spread
    uv run python -m finra_nlp.finetune --ensemble                     # average of the three saved seeds
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from finra_nlp.baseline import MIN_SUPPORT, _labels, macro_supported, score

DEFAULT_MODEL = "answerdotai/ModernBERT-base"


def load(name: str, n_labels: int):
    """Tokenizer and a multi-label classification model from the Hugging Face hub."""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    config = AutoConfig.from_pretrained(name, num_labels=n_labels, problem_type="multi_label_classification")
    if hasattr(config, "reference_compile"):
        config.reference_compile = False  # ModernBERT's torch.compile path needs Triton, which Windows lacks
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, config=config)
    return tok, model


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, tok, texts: list[str], max_len: int, batch: int, device) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i: i + batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**enc).logits
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config.num_labels))


def pos_weights(y: np.ndarray, cap: float = 10.0) -> np.ndarray:
    """Per-series weight on positive cases: sqrt(negatives / positives), between 1 and cap. Without it the
    loss favors predicting 'no' for rare series, which then fall under the 0.5 threshold."""
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1.0, cap).astype(np.float32)


def run(db: str, out: str, model_name: str = DEFAULT_MODEL, epochs: int = 6, batch: int = 16, micro: int = 8,
        max_len: int = 512, lr: float = 5e-5, seed: int = 0, limit_train: int | None = None,
        save_dir: str | None = "data/models/finetune", loader=load, weighted: bool = True) -> tuple[pd.DataFrame, float]:
    import torch

    _seed(seed)
    con = duckdb.connect(db)
    data = con.execute("SELECT case_no, split, masked_text, labels FROM model.dataset "
                       "WHERE split IN ('train', 'validation')").df()
    train = data[data.split == "train"].sample(frac=1, random_state=seed).reset_index(drop=True)
    if limit_train:
        train = train.head(limit_train)
    val = data[data.split == "validation"].sort_values("case_no").reset_index(drop=True)
    classes = sorted({lab for s in data[data.split == "train"]["labels"] for lab in _labels(s)})
    y_train = np.array([[float(c in _labels(s)) for c in classes] for s in train["labels"]], dtype=np.float32)
    y_val = np.array([[int(c in _labels(s)) for c in classes] for s in val["labels"]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = loader(model_name, len(classes))
    model.to(device)
    texts = train.masked_text.fillna("").tolist()
    vtexts = val.masked_text.fillna("").tolist()
    lengths = [len(x) for x in tok(texts[:2000], truncation=False)["input_ids"]]
    cut = float(np.mean([n > max_len for n in lengths]))
    print(f"{len(train)} training and {len(val)} validation cases, {len(classes)} series, device {device}; "
          f"{cut:.1%} of training summaries are longer than {max_len} tokens and are cut", flush=True)

    accum = max(1, batch // micro)
    steps = math.ceil(len(train) / micro / accum) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    weights = pos_weights(y_train) if weighted else np.ones(len(classes), dtype=np.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device=device))

    best, best_epoch, best_prob, log = -1.0, 0, None, []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train))
        total, n = 0.0, 0
        opt.zero_grad()
        for b, i in enumerate(range(0, len(order), micro), 1):
            idx = order[i: i + micro]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            target = torch.tensor(y_train[idx], device=device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), target) / accum
            scaler.scale(loss).backward()
            if b % accum == 0 or i + micro >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            total += loss.item() * accum
            n += 1
            if b % 100 == 0:
                print(f"  epoch {epoch}, batch {b}/{math.ceil(len(order) / micro)}, loss {total / n:.4f}, "
                      f"{(time.monotonic() - start) / 60:.1f} min", flush=True)
        prob = predict(model, tok, vtexts, max_len, micro * 2, device)
        per, macro = score(classes, y_val, prob)
        m10, n10 = macro_supported(per)
        log.append({"epoch": epoch, "train_loss": total / max(n, 1), "macro_f1": macro, "macro_f1_10plus": m10})
        print(f"epoch {epoch}: train loss {total / max(n, 1):.4f}, validation macro-F1 {macro:.3f}, "
              f"10+ series {m10:.3f} | {(time.monotonic() - start) / 60:.1f} min", flush=True)
        if m10 > best:
            best, best_epoch, best_prob = m10, epoch, prob
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_dir)
                tok.save_pretrained(save_dir)
                Path(save_dir, "series.json").write_text(json.dumps(classes), encoding="utf-8")

    per, macro = score(classes, y_val, best_prob)
    m10, n10 = macro_supported(per)
    probs = pd.DataFrame(best_prob, columns=classes)
    probs.insert(0, "case_no", val.case_no.values)
    probs.insert(1, "labels", val["labels"].values)
    con.execute("CREATE SCHEMA IF NOT EXISTS model")
    con.register("probs", probs)
    con.execute("CREATE OR REPLACE TABLE model.finetune_validation AS SELECT * FROM probs")
    con.close()
    text = "\n".join([
        "# Fine-tuned transformer, validation split", "",
        f"Model {model_name}, {epochs} epochs (best: epoch {best_epoch}), batch {batch}, learning rate {lr}, "
        f"positive weights {'sqrt(neg/pos), capped at 10' if weighted else 'none'}, "
        f"max {max_len} tokens ({cut:.1%} of training summaries cut), seed {seed}, device {device}. "
        f"Train cases: {len(train)}. Validation cases: {len(val)}. Threshold 0.5.", "",
        f"Macro-F1: {macro:.3f} (all {len(per)} series)", "",
        f"Macro-F1, series with {MIN_SUPPORT}+ validation cases: {m10:.3f} ({n10} series)", "",
        per.round(3).to_markdown(), "",
        "## By epoch", "", pd.DataFrame(log).round(4).to_markdown(index=False), "",
    ])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    return per, macro


ENSEMBLE_DIRS = ["data/models/finetune", "data/models/finetune_seed1", "data/models/finetune_seed2"]


def load_saved(path: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return (AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path),
            json.loads(Path(path, "series.json").read_text(encoding="utf-8")))


def ensemble(db: str, out: str, dirs: list[str] = ENSEMBLE_DIRS, max_len: int = 512, batch: int = 16,
             loader=load_saved) -> tuple[pd.DataFrame, float]:
    """Average the saved seeds' probabilities on the validation split and score the average."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    con = duckdb.connect(db)
    val = con.execute("SELECT case_no, masked_text, labels FROM model.dataset WHERE split = 'validation' "
                      "ORDER BY case_no").df()
    probs, classes, rows = [], None, []
    for d in dirs:
        tok, model, cls = loader(d)
        if classes is not None and cls != classes:
            raise SystemExit(f"{d} was trained on different series than {dirs[0]}")
        classes = cls
        model.to(device)
        p = predict(model, tok, val.masked_text.fillna("").tolist(), max_len, batch, device)
        y = np.array([[int(c in _labels(s)) for c in classes] for s in val["labels"]])
        per_d, macro_d = score(classes, y, p)
        rows.append({"model": d, "macro_f1": macro_d, "macro_f1_10plus": macro_supported(per_d)[0]})
        probs.append(p)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prob = np.mean(probs, axis=0)
    y = np.array([[int(c in _labels(s)) for c in classes] for s in val["labels"]])
    per, macro = score(classes, y, prob)
    m10, n10 = macro_supported(per)
    rows.append({"model": "average", "macro_f1": macro, "macro_f1_10plus": m10})
    table = pd.DataFrame(prob, columns=classes)
    table.insert(0, "case_no", val.case_no.values)
    table.insert(1, "labels", val["labels"].values)
    con.register("ens", table)
    con.execute("CREATE OR REPLACE TABLE model.finetune_ensemble_validation AS SELECT * FROM ens")
    con.close()
    text = "\n".join([
        f"# Fine-tuned transformer, average of {len(dirs)} seeds, validation split", "",
        f"Macro-F1: {macro:.3f} (all {len(per)} series)", "",
        f"Macro-F1, series with {MIN_SUPPORT}+ validation cases: {m10:.3f} ({n10} series)", "",
        per.round(3).to_markdown(), "", "## Each seed and the average", "",
        pd.DataFrame(rows).round(4).to_markdown(index=False), "",
    ])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    return per, macro


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--out", default="reports/finetune_validation.md")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16, help="effective batch size")
    ap.add_argument("--micro", type=int, default=8, help="cases per step on the GPU; lower it if memory runs out")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, help="train on this many cases only (to check memory and speed)")
    ap.add_argument("--no-weights", action="store_true", help="unweighted loss (the first validation run)")
    ap.add_argument("--ensemble", action="store_true",
                    help="score the average of the saved seed 0, 1 and 2 models on validation (no training)")
    a = ap.parse_args()
    if a.ensemble:
        ensemble(a.db, "reports/finetune_ensemble_validation.md")
        return
    out = a.out if a.seed == 0 or a.out != "reports/finetune_validation.md" else f"reports/finetune_validation_seed{a.seed}.md"
    run(a.db, out, a.model, a.epochs, a.batch, a.micro, a.max_len, a.lr, a.seed, a.limit_train,
        save_dir=f"data/models/finetune_seed{a.seed}" if a.seed else "data/models/finetune", weighted=not a.no_weights)


if __name__ == "__main__":
    main()
