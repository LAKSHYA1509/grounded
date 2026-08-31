"""
Tests for the router — the function that decides whether to loop.

WHY THIS IS THE MOST IMPORTANT TEST FILE
----------------------------------------
The cycle is the whole point of using a graph. A bug in route_after_grade
either breaks the retry (so it never loops, and the grading step is dead
weight) or breaks the termination (so it loops forever and burns money on
model calls until someone notices).

The router is a pure function — state in, string out. No model, no network.
So the risky part of the system is also the cheap part to test. That is not
a coincidence; it's what the design was for.
"""

from app.graph import MAX_ATTEMPTS, route_after_grade, widen


def test_sufficient_context_goes_straight_to_generate():
    state = {"grade": "sufficient", "attempts": 1}
    assert route_after_grade(state) == "generate"


def test_insufficient_context_retries():
    state = {"grade": "insufficient", "attempts": 1}
    assert route_after_grade(state) == "retrieve"


def test_retry_stops_at_max_attempts():
    """
    The termination guarantee. Without this the graph can loop forever on a
    question the documents simply cannot answer.
    """
    state = {"grade": "insufficient", "attempts": MAX_ATTEMPTS}
    assert route_after_grade(state) == "generate"


def test_retry_stops_past_max_attempts():
    state = {"grade": "insufficient", "attempts": MAX_ATTEMPTS + 5}
    assert route_after_grade(state) == "generate"


def test_widen_increases_k():
    """
    Retrying with the same k retrieves the same chunks and grades the same
    way — a loop that spends money and changes nothing. Each retry must cast
    a wider net or the cycle is pointless.
    """
    assert widen({"k": 4})["k"] == 8


def test_widen_handles_missing_k():
    """Falls back to the configured default rather than crashing on None."""
    assert widen({})["k"] > 0
