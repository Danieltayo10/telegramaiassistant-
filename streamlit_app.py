# ===================== STREAMLIT APP =====================
import streamlit as st
import hashlib
from sqlalchemy import create_engine, Column, Integer, Text, text
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from openai import OpenAI
from sqlalchemy.exc import ProgrammingError
import os

# -------------------- ENV --------------------
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

# -------------------- DATABASE SETUP --------------------
try:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
except ProgrammingError:
    pass

Base.metadata.create_all(engine)

# -------------------- AI --------------------
openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENAI_API_KEY)

def embed_text(text_value: str):
    return openai_client.embeddings.create(
        model="mistralai/mixtral-8x7b-instruct",
        input=text_value
    ).data[0].embedding

# -------------------- 🔒 NUL BYTE SANITIZER --------------------
def sanitize_text(text: str) -> str:
    return text.replace("\x00", "")

# -------------------- STREAMLIT --------------------
st.title("Client Knowledge Base Manager")

# -------------------- AUTH --------------------
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

def login(username, password):
    db = SessionLocal()
    user = db.execute(
        text("SELECT * FROM users WHERE username=:u AND password_hash=:p"),
        {"u": username, "p": hash_password(password)}
    ).fetchone()
    db.close()
    return user

def signup(username, password, token=None):
    db = SessionLocal()
    try:
        db.execute(
            text("INSERT INTO users (username, password_hash, token) VALUES (:u, :p, :t)"),
            {"u": username, "p": hash_password(password), "t": token}
        )
        db.commit()
        user = db.execute(text("SELECT * FROM users WHERE username=:u"), {"u": username}).fetchone()
    except:
        db.rollback()
        user = None
    db.close()
    return user

# -------------------- LOGIN / SIGNUP --------------------
if "user_id" not in st.session_state:
    st.title("🔑 Login / Signup")

    tab = st.radio("Action", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    token = st.text_input("Business token (optional, for Telegram)") if tab=="Signup" else None
    submit = st.button("Submit")

    if submit:
        if tab=="Login":
            user = login(username, password)
            if user:
                st.session_state["user_id"] = user.id
                st.success(f"Logged in as {username}")
                st.rerun()
            else:
                st.error("Invalid credentials")
        else:
            user = signup(username, password, token)
            if user:
                st.session_state["user_id"] = user.id
                st.success(f"Account created for {username}")
                st.rerun()
            else:
                st.error("Username already exists")

else:
    user_id = st.session_state["user_id"]

    uploaded_file = st.file_uploader("Upload text file")

    if uploaded_file:
        raw_bytes = uploaded_file.read()
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = raw_bytes.decode("latin-1")

        content = sanitize_text(content)  # ✅ FIX

        file_hash = hashlib.sha256(content.encode()).hexdigest()
        db = SessionLocal()

        exists = db.execute(
            text("SELECT 1 FROM documents WHERE user_id=:uid AND file_hash=:fh"),
            {"uid": user_id, "fh": file_hash}
        ).fetchone()

        if exists:
            st.warning("Duplicate file already exists.")
        else:
            result = db.execute(
                text("""
                    INSERT INTO documents (user_id, filename, file_hash)
                    VALUES (:uid,:fn,:fh)
                    RETURNING id
                """),
                {"uid": user_id, "fn": uploaded_file.name, "fh": file_hash}
            )
            document_id = result.fetchone()[0]

            chunks = [content[i:i+500] for i in range(0, len(content), 500)]
            for idx, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                embedding = embed_text(chunk)

                db.execute(
                    text("""
                        INSERT INTO document_chunks
                        (user_id, document_id, chunk_text, chunk_hash, embedding)
                        VALUES (:uid,:did,:ct,:ch,:emb)
                    """),
                    {
                        "uid": user_id,
                        "did": document_id,
                        "ct": sanitize_text(chunk),  # ✅ FIX
                        "ch": chunk_hash,
                        "emb": Vector(embedding)
                    }
                )

            db.commit()
            st.success("File uploaded and indexed successfully.")

    if st.button("Logout"):
        st.session_state.pop("user_id")
        st.rerun()


# ===================== FASTAPI BACKEND =====================
from fastapi import FastAPI, Request

# -------------------- FASTAPI --------------------
app = FastAPI()

# -------------------- TELEGRAM --------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send_message(chat_id, text):
    requests.post(f"{TELEGRAM_API}/sendMessage", json={
        "chat_id": chat_id,
        "text": text
    })

# -------------------- FASTAPI ENDPOINTS --------------------
@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data:
        return {"ok": True}

    chat_id = str(data["message"]["chat"]["id"])
    text_msg = data["message"].get("text", "")

    db = SessionLocal()

    if text_msg.startswith("/start"):
        parts = text_msg.split(" ")
        if len(parts) < 2:
            send_message(chat_id, "Usage: /start <business_token>")
            return {"ok": True}

        token = parts[1]
        user = db.execute(
            text("SELECT * FROM users WHERE token=:t"),
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

    # -------------------- RAG --------------------
    def embed_text_msg(text_value: str):
        return openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text_value
        ).data[0].embedding

    def search_similar_chunks(user_id: int, query: str, top_k=5):
        query_embedding = embed_text_msg(query)
        with engine.connect() as conn:
            result = conn.execute(text("""
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
            return [sanitize_text(row[0]) for row in result.fetchall()]  # ✅ FIX

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

    reply = generate_answer(session.user_id, text_msg)
    send_message(chat_id, reply)
    db.close()
    return {"ok": True}

@app.get("/")
def health():
    return {"status": "ok"}
