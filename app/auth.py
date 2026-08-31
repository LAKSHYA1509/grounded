"""
Access control.

WHY THIS FILE EXISTS
--------------------
The deployed instance is public and holds a model provider API key. /ask makes at
least two paid model calls per request. An unauthenticated endpoint that
spends money on behalf of whoever finds the URL is not a demo — it's an open
wallet, and the bill arrives before you notice.

So: if API_KEY is set, every request except /health must present it.

WHY A SHARED SECRET AND NOT REAL AUTH
-------------------------------------
Because it is proportionate. This is a single-user demo service with no
per-user data and nothing to authorise *between* users — there is exactly
one thing to decide, "may you call this at all," and a shared secret decides
it correctly.

Real user accounts here would be more code, more attack surface, and no more
security. Knowing which control is proportionate to the risk is the actual
skill; reaching for the heaviest available option isn't rigour, it's noise.

WHY compare_digest AND NOT ==
-----------------------------
`==` on strings short-circuits at the first differing byte, so it returns
faster for a wrong guess that shares a longer prefix. That timing difference
is measurable over many requests and leaks the secret one byte at a time.

`compare_digest` takes the same time regardless. The attack is impractical
here — but using the constant-time comparison costs one import and means you
never have to reason about whether it matters.
"""

from secrets import compare_digest

from fastapi import Header, HTTPException, status

from app.config import settings

# Paths that must stay reachable without a key, so a hosting platform can
# health-check the container and so /docs is browsable.
OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    """
    FastAPI dependency. Returns None when allowed, raises 401 otherwise.

    No-op when API_KEY is unset — that keeps local development and the test
    suite frictionless. The deployment is where the key gets set.
    """
    if not settings.api_key:
        return

    if not x_api_key or not compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            # Say what is wrong and how to fix it. Never hint at whether the
            # key was close — that turns an error message into an oracle.
            detail="Missing or invalid API key. Send it as the X-API-Key header.",
        )
