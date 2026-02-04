from fastapi import FastAPI, Request
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
import os
import requests
import asyncio
import traceback

from streamlit_app import User, TelegramSession, Chunk, engine  # reuse models

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

app = FastAPI()
SessionLocal = sessionmaker(bind=engine)

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


# -------------------- TELEGRAM --------------------
def send_message(chat_id, text):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10
    )


# -------------------- EMBEDDING --------------------
def embed_text_msg(text_value: str):
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text_value
    ).data[0].embedding


# -------------------- VECTOR SEARCH --------------------
def search_similar_chunks(user_id: int, query: str, top_k=5):
    query_embedding = embed_text_msg(query)

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT chunk_text
                FROM document_chunks
                WHERE user_id = :uid
                ORDER BY embedding <-> CAST(:embedding AS vector)   -- ✅ FIXED
                LIMIT :top_k
            """),
            {
                "uid": user_id,
                "embedding": query_embedding,
                "top_k": top_k
            }
        )

        return [row[0] for row in result.fetchall()]


# -------------------- LLM --------------------
def call_llm(prompt: str):
    response = openai_client.chat.completions.create(
        model="mistralai/mixtral-8x7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        timeout=30
    )
    return response.choices[0].message.content


def generate_answer(user_id: int, user_message: str):
    chunks = search_similar_chunks(user_id, user_message)

    if not chunks:
        return "I don’t want to give incorrect information. Let me connect you with a team member."

    context = ""
    MAX_CHARS = 3000
    for c in chunks:
        if len(context) + len(c) > MAX_CHARS:
            break
        context += c + "\n"

    prompt = f"""
You are a business support agent.
Use ONLY the context below.
If the answer is missing, escalate.

Context:
{context}

User:
{user_message}
"""

    return call_llm(prompt)


# -------------------- WEBHOOK --------------------
@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()

    if "message" not in data:
        return {"ok": True}

    chat_id = str(data["message"]["chat"]["id"])
    text_msg = data["message"].get("text", "")

    db = SessionLocal()

    try:
        # -------- START COMMAND --------
        if text_msg.startswith("/start"):
            parts = text_msg.split(" ")

            if len(parts) < 2:
                send_message(chat_id, "Usage: /start <business_token>")
                return {"ok": True}

            token = parts[1]

            user = db.execute(
                text("SELECT * FROM users WHERE token = :t"),
                {"t": token}
            ).fetchone()

            if not user:
                send_message(chat_id, "Invalid business token.")
                return {"ok": True}

            session_obj = TelegramSession(chat_id=chat_id, user_id=user.id)
            db.merge(session_obj)
            db.commit()

            send_message(chat_id, f"Connected to {user.username}. How can I help?")
            return {"ok": True}

        # -------- SESSION CHECK --------
        session = db.query(TelegramSession).filter_by(chat_id=chat_id).first()
        if not session:
            send_message(chat_id, "Please start with /start <business_token>")
            return {"ok": True}

        # -------- RAG (RUN IN THREAD) --------
        reply = await asyncio.to_thread(   # ✅ FIXED
            generate_answer,
            session.user_id,
            text_msg
        )

        send_message(chat_id, reply)
        return {"ok": True}

    except Exception as e:
        print("🔥 TELEGRAM ERROR:", e)
        traceback.print_exc()
        send_message(chat_id, "Something went wrong. Please try again later.")
        return {"ok": True}

    finally:
        db.close()


# -------------------- HEALTH --------------------
@app.get("/")
def health():
    return {"status": "ok"}
