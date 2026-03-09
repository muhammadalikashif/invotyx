import os
import asyncio
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ── Gemini client setup ───────────────────────────────────────────────────────

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL  = "gemini-2.5-flash"
MAX_HISTORY   = 5   # keep last 5 message pairs per session
REQUEST_TIMEOUT = 60  # seconds per attempt
MAX_RETRIES   = 3   # retry up to 3 times on 429
RETRY_DELAY   = 15  # seconds to wait between retries

# In-memory store: session_id -> list of {role, text}
conversation_store: dict[str, list[dict]] = defaultdict(list)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Chat API",
    description="Minimal AI chat backend using Gemini 2.5 Flash",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    message: str
    session_id: str


class SummarizeRequest(BaseModel):
    text: str
    session_id: Optional[str] = "summary-default"


# ── Helpers ───────────────────────────────────────────────────────────────────

def trim_history(session_id: str):
    """Keep only the last MAX_HISTORY message pairs (user + model = 2 entries each)."""
    history = conversation_store[session_id]
    max_entries = MAX_HISTORY * 2
    if len(history) > max_entries:
        conversation_store[session_id] = history[-max_entries:]


def build_contents(history: list[dict], new_message: str) -> list[types.Content]:
    """Convert stored history + new message into Gemini Content objects."""
    contents = []
    for msg in history:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["text"])],
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=new_message)],
        )
    )
    return contents


async def call_gemini(contents: list[types.Content]) -> str:
    """
    Call Gemini 2.5 Flash with automatic retry on 429 rate-limit errors.
    Runs the sync SDK call in a thread pool to avoid blocking the event loop.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            def _generate():
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                )
                return response.text

            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _generate),
                timeout=REQUEST_TIMEOUT,
            )

        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str

            if is_rate_limit and attempt < MAX_RETRIES:
                print(f"[WARN] Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                      f"Retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)
                continue

            # Final attempt or non-retryable error
            print(f"[ERROR] {type(e).__name__}: {e}")
            raise


FALLBACK = "This is taking longer than expected. Please try again."

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Chat API is running."}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /chat — accepts a user message, maintains context (last 5 exchanges),
    calls Gemini 2.5 Flash with retry on rate limits, returns AI response.
    """
    session_id  = request.session_id or "default"
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    contents = build_contents(conversation_store[session_id], user_message)

    try:
        ai_reply = await call_gemini(contents)
    except Exception:
        return ChatResponse(message=FALLBACK, session_id=session_id)

    conversation_store[session_id].append({"role": "user",  "text": user_message})
    conversation_store[session_id].append({"role": "model", "text": ai_reply})
    trim_history(session_id)

    return ChatResponse(message=ai_reply, session_id=session_id)


@app.post("/summarize", response_model=ChatResponse)
async def summarize(request: SummarizeRequest):
    """
    POST /summarize — submit long text or a refinement instruction.
    Session context is preserved so follow-ups like 'make it shorter' work.
    """
    session_id  = request.session_id or "summary-default"
    user_message = request.text.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    contents = build_contents(conversation_store[session_id], user_message)

    try:
        ai_reply = await call_gemini(contents)
    except Exception:
        return ChatResponse(message=FALLBACK, session_id=session_id)

    conversation_store[session_id].append({"role": "user",  "text": user_message})
    conversation_store[session_id].append({"role": "model", "text": ai_reply})
    trim_history(session_id)

    return ChatResponse(message=ai_reply, session_id=session_id)


@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a given session."""
    conversation_store.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}