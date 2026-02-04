# ===================== STREAMLIT APP =====================
import streamlit as st
import hashlib
from sqlalchemy import create_engine, Column, Integer, Text, text
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from openai import OpenAI
from sqlalchemy.exc import ProgrammingError
import os
import docx2txt
import PyPDF2

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
        model="text-embedding-3-small",
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

    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # Extract text from any file type
        content = ""
        try:
            if file_ext in ["txt", "csv", "log"]:
                try:
                    content = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = file_bytes.decode("latin-1")
            elif file_ext in ["docx"]:
                with open(f"/tmp/{uploaded_file.name}", "wb") as f:
                    f.write(file_bytes)
                content = docx2txt.process(f"/tmp/{uploaded_file.name}")
            elif file_ext in ["pdf"]:
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    content += page.extract_text() or ""
            else:
                st.warning("Unsupported file type, attempting raw text decoding")
                try:
                    content = file_bytes.decode("utf-8")
                except:
                    content = file_bytes.decode("latin-1")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

        content = sanitize_text(content)

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

                new_chunk = Chunk(
                    user_id=user_id,
                    document_id=document_id,
                    chunk_text=sanitize_text(chunk),
                    chunk_hash=chunk_hash,
                    embedding=embedding
                )
                db.add(new_chunk)

            db.commit()
            st.success("File uploaded and indexed successfully.")

    if st.button("Logout"):
        st.session_state.pop("user_id")
        st.rerun()
