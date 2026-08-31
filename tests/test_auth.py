"""
Tests for the API key gate.

WHY THIS MATTERS MORE THAN IT LOOKS
-----------------------------------
This is the only thing standing between a public URL and a drained quota.
A bug here has a direct financial consequence, and it fails silently — an
auth check that accidentally lets everything through looks exactly like an
auth check that works, right up until the invoice.

So it gets tested properly, including the case people skip: that the gate is
genuinely off when unconfigured. A half-configured gate that rejects
everything is just as broken as one that allows everything.

A NOTE ON HOW THESE TESTS PATCH CONFIG
--------------------------------------
`Settings` is a frozen dataclass, so you cannot assign to it. Instead we
build a modified copy with `dataclasses.replace` and swap the module-level
name. That keeps the immutability guarantee in the real code — the thing
that makes settings safe to trust — while still letting tests vary it.
"""

import asyncio
from dataclasses import replace

import pytest
from fastapi import HTTPException

from app import auth
from app.config import settings


def call(api_key: str, header: str):
    """Point auth.settings at a copy with the given key, then invoke it."""
    auth.settings = replace(settings, api_key=api_key)
    return asyncio.run(auth.require_api_key(x_api_key=header))


@pytest.fixture(autouse=True)
def restore_settings():
    """Put the real settings back, whatever the test did."""
    yield
    auth.settings = settings


def test_no_key_configured_means_open():
    """
    Local development and CI must work with no key set. If this breaks,
    every contributor is blocked on a secret they don't need.
    """
    assert call("", "") is None
    assert call("", "anything") is None


def test_correct_key_is_accepted():
    assert call("s3cret", "s3cret") is None


def test_wrong_key_is_rejected():
    with pytest.raises(HTTPException) as e:
        call("s3cret", "wrong")
    assert e.value.status_code == 401


def test_missing_header_is_rejected():
    """The common case: someone finds the URL and just calls it."""
    with pytest.raises(HTTPException) as e:
        call("s3cret", "")
    assert e.value.status_code == 401


def test_prefix_of_the_real_key_is_rejected():
    """
    Guards the constant-time comparison. A prefix must fail exactly like any
    other wrong value — no early accept, no different error.
    """
    with pytest.raises(HTTPException) as e:
        call("s3cret", "s3c")
    assert e.value.status_code == 401


def test_error_does_not_leak_the_key():
    """
    The message must say what to do, never how close the guess was.
    An error that reveals proximity turns itself into an oracle.
    """
    with pytest.raises(HTTPException) as e:
        call("s3cret", "wrong")
    assert "s3cret" not in e.value.detail
    assert "X-API-Key" in e.value.detail


def test_health_is_in_the_open_paths():
    """A platform must be able to probe liveness without a secret."""
    assert "/health" in auth.OPEN_PATHS
