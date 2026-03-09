"""
test_api.py — Test script for AI Chat API
Run with: python test_api.py
Make sure the server is running first: uvicorn main:app --reload --port 8000

NOTE: Gemini free tier = 5 requests/minute.
      Delays are added between tests to stay within quota.
"""

import requests
import json
import time

BASE_URL    = "http://127.0.0.1:8000"
SESSION_ID  = "test-session-001"
SUMMARY_SESSION = "summary-session-001"

# Seconds to wait between each API call (free tier = 5 req/min = 1 req/12s)
DELAY = 13

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_result(label: str, response: requests.Response):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Status : {response.status_code}")
    try:
        data = response.json()
        print(f"  Response:\n{json.dumps(data, indent=4)}")
    except Exception:
        print(f"  Raw: {response.text}")


def post(endpoint: str, payload: dict) -> requests.Response:
    return requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=60)


def delete(endpoint: str) -> requests.Response:
    return requests.delete(f"{BASE_URL}{endpoint}", timeout=10)


def wait(seconds: int = DELAY):
    print(f"  ⏳ Waiting {seconds}s (free tier rate limit)...")
    time.sleep(seconds)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health():
    print("\n>>> TEST: Health Check (GET /)  — no API call, no delay needed")
    r = requests.get(f"{BASE_URL}/", timeout=10)
    print_result("GET /", r)
    assert r.status_code == 200, "Health check failed!"
    print("  ✓ PASSED")


def test_basic_chat():
    print("\n>>> TEST: Basic Chat Message")
    r = post("/chat", {
        "message": "What is the capital of France?",
        "session_id": SESSION_ID,
    })
    print_result("POST /chat — basic question", r)
    assert r.status_code == 200
    assert r.json().get("message") != "This is taking longer than expected. Please try again.", \
        "Got fallback — likely rate limited"
    print("  ✓ PASSED")
    wait()


def test_context_followup():
    print("\n>>> TEST: Context Follow-up (should remember Paris)")
    r = post("/chat", {
        "message": "What is the population of that city?",
        "session_id": SESSION_ID,
    })
    print_result("POST /chat — follow-up (context test)", r)
    assert r.status_code == 200
    assert r.json().get("message") != "This is taking longer than expected. Please try again.", \
        "Got fallback — likely rate limited"
    print("  ✓ PASSED")
    wait()


def test_multi_turn_history():
    print("\n>>> TEST: Multi-turn — 3 messages (condensed to respect rate limit)")
    questions = [
        "Tell me one fun fact about space.",
        "Tell me another fun fact about space.",
        "What was the first fun fact you told me?",
    ]
    for i, q in enumerate(questions, 1):
        r = post("/chat", {"message": q, "session_id": SESSION_ID})
        print_result(f"POST /chat — turn {i}", r)
        assert r.status_code == 200
        assert r.json().get("message") != "This is taking longer than expected. Please try again.", \
            f"Turn {i} got fallback — likely rate limited"
        if i < len(questions):
            wait()
    print("  ✓ PASSED")
    wait()


def test_empty_message():
    print("\n>>> TEST: Empty Message (should return 400 — no API call made)")
    r = post("/chat", {"message": "", "session_id": SESSION_ID})
    print_result("POST /chat — empty message", r)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print("  ✓ PASSED")
    # No delay needed — Gemini was never called


def test_summarize_initial():
    print("\n>>> TEST: Summarize — initial long text")
    long_text = (
        "Summarize this: Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to natural intelligence displayed by animals including humans. AI research has "
        "been defined as the field of study of intelligent agents, which refers to any system that "
        "perceives its environment and takes actions that maximize its chance of achieving its goals."
    )
    r = post("/summarize", {"text": long_text, "session_id": SUMMARY_SESSION})
    print_result("POST /summarize — initial", r)
    assert r.status_code == 200
    assert r.json().get("message") != "This is taking longer than expected. Please try again.", \
        "Got fallback — likely rate limited"
    print("  ✓ PASSED")
    wait()


def test_summarize_refinement_shorter():
    print("\n>>> TEST: Summarize — refinement: make it shorter")
    r = post("/summarize", {
        "text": "make it shorter",
        "session_id": SUMMARY_SESSION,
    })
    print_result("POST /summarize — make it shorter", r)
    assert r.status_code == 200
    assert r.json().get("message") != "This is taking longer than expected. Please try again.", \
        "Got fallback — likely rate limited"
    print("  ✓ PASSED")
    wait()


def test_summarize_refinement_bullets():
    print("\n>>> TEST: Summarize — refinement: convert to bullet points")
    r = post("/summarize", {
        "text": "convert it to bullet points",
        "session_id": SUMMARY_SESSION,
    })
    print_result("POST /summarize — bullet points", r)
    assert r.status_code == 200
    assert r.json().get("message") != "This is taking longer than expected. Please try again.", \
        "Got fallback — likely rate limited"
    print("  ✓ PASSED")
    wait()


def test_clear_session():
    print("\n>>> TEST: Clear Session (DELETE — no API call)")
    r = delete(f"/chat/{SESSION_ID}")
    print_result(f"DELETE /chat/{SESSION_ID}", r)
    assert r.status_code == 200
    assert r.json().get("status") == "cleared"
    print("  ✓ PASSED")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AI Chat API — Test Suite")
    print("  Free tier: 5 req/min → 13s delay between calls")
    print("  Estimated total time: ~2 minutes")
    print("="*55)

    tests = [
        test_health,
        test_basic_chat,
        test_context_followup,
        test_multi_turn_history,
        test_empty_message,
        test_summarize_initial,
        test_summarize_refinement_shorter,
        test_summarize_refinement_bullets,
        test_clear_session,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED — {e}")
            failed += 1
        except requests.exceptions.ConnectionError:
            print("\n  ✗ CONNECTION ERROR — Is the server running?")
            print("  Start it with: uvicorn main:app --reload --port 8000")
            break

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*55}\n")