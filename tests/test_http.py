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
