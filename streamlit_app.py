import streamlit as st
import hashlib
from sqlalchemy import create_engine, Column, Integer, Text, text
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from openai import OpenAI
import os
from sqlalchemy.exc import ProgrammingError

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
        # Create pgvector extension if not exists
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
except ProgrammingError:
    pass

# Create tables if missing
Base.metadata.create_all(engine)

# -------------------- AI --------------------
openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENAI_API_KEY)

def embed_text(text_value: str):
    return openai_client.embeddings.create(
        model="mistralai/mixtral-8x7b-instruct",
        input=text_value
    ).data[0].embedding

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
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")
        else:
            user = signup(username, password, token)
            if user:
                st.session_state["user_id"] = user.id
                st.success(f"Account created for {username}")
                st.experimental_rerun()
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
                        "ct": chunk,
                        "ch": chunk_hash,
                        "emb": Vector(embedding)
                    }
                )

            db.commit()
            st.success("File uploaded and indexed successfully.")

    if st.button("Logout"):
        st.session_state.pop("user_id")
        st.experimental_rerun()
