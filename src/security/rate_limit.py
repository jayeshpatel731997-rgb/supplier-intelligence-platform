"""Simple local rate limiting middleware with Redis-ready shape."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


@dataclass(slots=True)
class InMemoryRateLimiter:
    limit: int
    window_seconds: int
    buckets: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        bucket = self.buckets[key]
        while bucket and now - bucket[0] >= self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.limit:
            retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
            return False, retry_after
        bucket.append(now)
        return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, enabled: bool, limit: int, window_seconds: int):
        super().__init__(app)
        self.enabled = enabled
        self.limiter = InMemoryRateLimiter(max(1, limit), max(1, window_seconds))

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled or request.url.path in {"/health", "/live", "/ready"}:
            return await call_next(request)
        tenant_id = request.headers.get("X-Tenant-ID", "public")
        api_key = request.headers.get("X-API-Key", "")
        api_prefix = api_key[:12] if api_key else request.client.host if request.client else "unknown"
        key = f"{tenant_id}:{api_prefix}"
        allowed, retry_after = self.limiter.allow(key)
        if not allowed:
            request_id = getattr(request.state, "request_id", request.headers.get("X-Request-ID", ""))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded.", "request_id": request_id},
                headers={"Retry-After": str(retry_after), "X-Request-ID": request_id},
            )
        return await call_next(request)
