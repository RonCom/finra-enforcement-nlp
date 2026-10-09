import httpx

from finra_nlp import http


def test_429_waits_longer_and_slows_down(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(http.time, "sleep", sleeps.append)
    answers = iter([429, 429, 200])
    c = http.PoliteClient(cache_dir=tmp_path, max_per_second=1.0, retry_wait=30.0, max_interval=4.0)
    c.client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(next(answers), content=b"ok")))
    assert c.get("https://www.finra.org/x") == (200, b"ok")
    assert [s for s in sleeps if s >= 30] == [30.0, 60.0]
    assert c.min_interval == 4.0


def test_rate_recovers_after_successes(tmp_path, monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    msgs = []
    answers = iter([429] + [200] * 4)
    c = http.PoliteClient(cache_dir=tmp_path, max_per_second=1.0, max_interval=4.0, recover_after=2, log=msgs.append)
    c.client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(next(answers), content=b"ok")))
    c.get("https://x/a")
    assert c.min_interval == 2.0 and c.throttled == 1 and "429" in msgs[0]
    c.get("https://x/b")  # second success in a row: back to 1 per second
    assert c.min_interval == 1.0
    assert c.cached("https://x/a") and not c.cached("https://x/zzz")
