import os
import requests
from fastapi import FastAPI, Request
from sqlalchemy import create_engine, Column, Integer, Text, event
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from openai import OpenAI
from sqlalchemy.sql import text as sql_text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy import text

# -------------------- ENV --------------------
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# -------------------- DB --------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# -------------------- MODELS --------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(Text, unique=True)
    password_hash = Column(Text)
    token = Column(Text, unique=True)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    filename = Column(Text)
    file_hash = Column(Text)

class Chunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    document_id = Column(Integer)
    chunk_text = Column(Text)
    chunk_hash = Column(Text)
    embedding = Column(Vector(1536))

class TelegramSession(Base):
    __tablename__ = "telegram_sessions"
    chat_id = Column(Text, primary_key=True)
    user_id = Column(Integer)

# -------------------- 🔒 NUL-BYTE DEFENSE (THE FIX) --------------------
@event.listens_for(Chunk, "before_insert")
@event.listens_for(Chunk, "before_update")
def sanitize_chunk_text(mapper, connection, target):
    if target.chunk_text:
        target.chunk_text = target.chunk_text.replace("\x00", "")

# -------------------- DATABASE SETUP --------------------
try:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
except ProgrammingError:
    pass

Base.metadata.create_all(engine)

# -------------------- AI CLIENT --------------------
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

def embed_text(text_value: str):
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text_value
    ).data[0].embedding

def search_similar_chunks(user_id: int, query: str, top_k=5):
    query_embedding = embed_text(query)
    with engine.connect() as conn:
        result = conn.execute(sql_text("""
            SELECT chunk_text
            FROM document_chunks
            WHERE user_id = :uid
            ORDER BY embedding <-> :embedding
            LIMIT :top_k
        """), {
            "uid": user_id,
            "embedding": query_embedding,
            "top_k": top_k
        })
        return [row[0] for row in result.fetchall()]

def call_llm(prompt: str):
    response = openai_client.chat.completions.create(
        model="mistralai/mixtral-8x7b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_answer(user_id: int, user_message: str):
    chunks = search_similar_chunks(user_id, user_message)

    if not chunks:
        return "I don’t want to give incorrect information. Let me connect you with a team member."

    context = "\n".join(chunks)

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

# -------------------- TELEGRAM --------------------
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send_message(chat_id, text):
    requests.post(f"{TELEGRAM_API}/sendMessage", json={
        "chat_id": chat_id,
        "text": text
    })

# -------------------- FASTAPI --------------------
app = FastAPI()

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data:
        return {"ok": True}

    chat_id = str(data["message"]["chat"]["id"])
    text = data["message"].get("text", "")

    db = SessionLocal()

    if text.startswith("/start"):
        parts = text.split(" ")
        if len(parts) < 2:
            send_message(chat_id, "Usage: /start <business_token>")
            return {"ok": True}

        token = parts[1]
        user = db.execute(
            sql_text("SELECT * FROM users WHERE token=:t"),
            {"t": token}
        ).fetchone()

        if not user:
            send_message(chat_id, "Invalid business token.")
            return {"ok": True}

        db.merge(TelegramSession(chat_id=chat_id, user_id=user.id))
        db.commit()
        send_message(chat_id, f"Connected to {user.username}. How can I help?")
        db.close()
        return {"ok": True}

    session = db.query(TelegramSession).filter_by(chat_id=chat_id).first()
    if not session:
        send_message(chat_id, "Please start with /start <business_token>")
        db.close()
        return {"ok": True}

    reply = generate_answer(session.user_id, text)
    send_message(chat_id, reply)
    db.close()
    return {"ok": True}

@app.get("/")
def health():
    return {"status": "ok"}
