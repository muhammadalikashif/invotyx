# AI Chat API

A minimal AI chat backend built with **FastAPI** and **Gemini 2.5 Flash**.

---

## How to Run

### 1. Clone / copy the project

```bash
cd chatbot
```

### 2. Create a `.env` file

```bash
cp .env.example .env
```

Edit `.env` and paste your Gemini Free API key:

```
GEMINI_API_KEY=your_actual_key_here
```

> Get a free key at: https://aistudio.google.com/app/apikey

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/chat` | Send a chat message |
| POST | `/summarize` | Summarize or refine text |
| DELETE | `/chat/{session_id}` | Clear session history |

---

## Sample Requests & Responses

### POST /chat

**Request**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?", "session_id": "user-123"}'
```

**Response**
```json
{
  "message": "The capital of France is Paris.",
  "session_id": "user-123"
}
```

---

### Follow-up (context preserved)

**Request**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is its population?", "session_id": "user-123"}'
```

**Response**
```json
{
  "message": "Paris has a population of approximately 2.1 million in the city proper, and around 12 million in the greater metropolitan area.",
  "session_id": "user-123"
}
```

---

### POST /summarize

**Request**
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Summarize this: Artificial intelligence is a branch of computer science...[long text]",
    "session_id": "summary-abc"
  }'
```

**Response**
```json
{
  "message": "AI is a field of computer science focused on building systems that can perform tasks requiring human-like intelligence.",
  "session_id": "summary-abc"
}
```

**Refinement follow-up**
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "make it shorter", "session_id": "summary-abc"}'
```

**Timeout / Error Response**
```json
{
  "message": "This is taking longer than expected. Please try again.",
  "session_id": "user-123"
}
```

---

## Project Structure

```
ai-chat-api/
├── app/
│   └── main.py          # All API logic
├── .env.example         # Environment variable template
├── requirements.txt     # Python dependencies
└── README.md
```

---

---

# Part 2 — AI Backend Design (Short Answers)

---

## 1. Smart Mode Routing

The routing layer sits in front of the LLM and classifies each incoming query before dispatching it. A lightweight classifier (either a small fine-tuned model or a set of regex/keyword heuristics) inspects the user's intent and tags the request as one of: `realtime`, `general_knowledge`, or `summarization`. Based on this tag, a router function calls the appropriate backend — a web-search-augmented pipeline (e.g., Gemini + Google Search grounding) for real-time queries, a standard LLM call for general knowledge, or a dedicated lighter summarization prompt for summarization tasks.

The architecture looks like:

```
User Request → Intent Classifier → Router
                                    ├─ realtime      → Web Search API + LLM
                                    ├─ general       → LLM (Gemini 2.5 Flash)
                                    └─ summarization → Lightweight Summarization Prompt
```

Each branch can also have its own caching layer (Redis TTL cache) to avoid redundant API calls for repeated queries. Confidence scores from the classifier can trigger a fallback to `general_knowledge` if routing certainty is low.

---

## 2. Context Management & Token Limits

To prevent conversations from exceeding the model's context window, the system maintains a rolling window of the last N message pairs (user + assistant) in memory. Older messages beyond this window are replaced with a **compressed summary**: when the buffer fills, the system sends the oldest 50% of messages to a lightweight prompt asking Gemini to produce a concise paragraph summary, then stores that summary as a single synthetic "history" message.

This gives the model awareness of earlier conversation topics without consuming excessive tokens. The system also counts approximate tokens before each API call (using a character-based estimate or the `tiktoken` library), and if the total exceeds 80% of the model's limit, it triggers an early compression pass on the oldest messages before proceeding.

---

## 3. Guardrails

Three guardrail categories and their handling:

**1. Sexual / CSAM Content** — A pre-processing filter using a fine-tuned classifier (or a moderation API such as OpenAI Moderation or Google's SafetySettings on Gemini) flags sexual content before the message reaches the LLM. If flagged, the system returns a polite refusal message and logs the event for review. Gemini's built-in `harm_block_threshold` settings are also configured to block explicit sexual content at the API level.

**2. Illegal Instructions** — Keyword and semantic similarity checks screen for requests involving weapons synthesis, drug manufacturing, hacking instructions, or fraud. A second-pass LLM "judge" prompt can evaluate borderline cases. Flagged requests are blocked with a static refusal and the session is rate-limited or flagged for human review if repeated.

**3. Hate Speech** — A hate speech classifier (e.g., a fine-tuned BERT model or a third-party moderation API) inspects the input for targeted slurs, dehumanizing language, or content that incites violence against protected groups. Detected content is rejected before reaching the main model, and repeat offenders can have their session terminated. All three guardrail layers run in parallel as async tasks so latency impact is minimal.

---

## 4. Image Input — "What is this product?" Pipeline

**Step 1 — Receive & Preprocess:** The backend receives the image as a base64-encoded payload or multipart upload, validates file type (JPEG/PNG/WEBP), and resizes it to a standard dimension to reduce token usage before sending to the vision model.

**Step 2 — Object Identification:** The image is sent to Gemini 2.5 Flash (which supports vision input) along with the user's question. The model returns a structured JSON response containing the identified product name, brand (if visible), key visual features, and a brief description.

**Step 3 — Generate Product Description:** A second prompt pass (or a single compound prompt) instructs the model to produce a consumer-friendly product description — covering use case, notable features, and estimated category — based on the identification result.

**Step 4 — Purchase Links:** The extracted product name and brand are passed to a product search API (e.g., Google Shopping API, SerpApi, or Amazon Product Advertising API) to retrieve real purchase links. These are appended to the response JSON alongside the AI-generated description.

```
Image Upload → Preprocess → Gemini Vision (ID + Describe)
                                         ↓
                              Product Search API (links)
                                         ↓
                              Combined JSON Response → User
```

---

---

# Part 3 — Text Summary Module Design

---

## How the Backend Processes the Request

The `/summarize` endpoint accepts any text input — either a long document on first call or a short refinement instruction on follow-up calls. On the first request, a system-level instruction is prepended to the Gemini prompt: *"You are a summarization assistant. When given text, summarize it clearly and concisely. Accept follow-up commands to refine the summary."* The user's text is then appended and sent to Gemini 2.5 Flash.

## How Context is Maintained Across Refinement Steps

Each summarization session is identified by a `session_id`. The conversation history (original text summary, follow-up instructions, and model responses) is stored in the same in-memory `conversation_store` used by the chat endpoint. When the user sends a follow-up like *"make it shorter"* or *"convert to bullet points"*, the full prior history is included in the `contents` array sent to Gemini — so the model knows what it previously summarized and applies the refinement correctly.

## How Token Limits Are Handled

Because summaries can involve very long source texts, the system checks the approximate character count of the input before sending it. If the combined input (history + new text) exceeds a safe threshold (~100,000 characters as a proxy for token budget), the original submitted text is chunked into segments and summarized in a map-reduce pattern: each chunk is summarized individually, then the chunk summaries are combined into a final summary. For subsequent refinement steps, only the summary (not the original long text) is kept in history, which keeps token usage minimal. If even the rolling window grows too large, the oldest history entries are compressed using the same summarization strategy described in Part 2.
