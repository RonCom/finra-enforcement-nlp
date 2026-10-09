"""Rate-limited HTTP client with retries and an on-disk cache.

SEC asks automated clients to declare a User-Agent with a name and email and to stay
at or under 10 requests per second. The same header and a 2-per-second default are used
for finra.org, which has no published rate in the sources checked.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import httpx

DEFAULT_USER_AGENT = os.environ.get("SEC_USER_AGENT", "Chris L cflave@gmail.com")
RETRY_STATUS = {429, 500, 502, 503, 504}


class PoliteClient:
    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        user_agent: str = DEFAULT_USER_AGENT,
        max_per_second: float = 2.0,
        max_retries: int = 5,
        timeout: float = 30.0,
        retry_wait: float = 0.0,
        max_interval: float = 1.0,
    ) -> None:
        """retry_wait: seconds to wait before retry n is at least n * retry_wait (a server that answers
        429 for a minute needs more than the default 1, 2, 4, 8, 16 s). Each 429 also doubles the gap
        between requests, up to max_interval seconds, for the rest of the run."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / max_per_second
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.max_interval = max(max_interval, self.min_interval)
        self._last = 0.0
        self._lock = threading.Lock()
        self.client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
        )

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / digest[:2] / digest

    def _wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()

    def get(self, url: str, use_cache: bool = True) -> tuple[int, bytes]:
        """Return (status, body). 404s are cached as empty bodies so reruns skip them."""
        path = self._cache_path(url)
        miss = path.with_suffix(".404")
        if use_cache and path.exists():
            return 200, path.read_bytes()
        if use_cache and miss.exists():
            return 404, b""

        for attempt in range(self.max_retries + 1):
            self._wait()
            try:
                resp = self.client.get(url)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                time.sleep(2**attempt)
                continue
            if resp.status_code == 429:
                self.min_interval = min(self.min_interval * 2, self.max_interval)
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(max(wait, (attempt + 1) * self.retry_wait))
                continue
            break

        if resp.status_code == 403 and "x-deny-reason" in resp.headers:
            raise RuntimeError(f"Blocked by local proxy ({resp.headers['x-deny-reason']}): {url}")
        if resp.status_code == 200:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resp.content)
        elif resp.status_code == 404:
            miss.parent.mkdir(parents=True, exist_ok=True)
            miss.touch()
        return resp.status_code, resp.content
