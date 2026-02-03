import streamlit as st
import hashlib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from openai import OpenAI
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Client Knowledge Base Manager",
    layout="centered"
)

# -------------------- ENV --------------------
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# -------------------- DB --------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# -------------------- AI --------------------
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

def embed_text(text_value: str):
    return openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text_value
    ).data[0].embedding

# -------------------- UI --------------------
st.title("📚 Client Knowledge Base Manager")
st.caption("Upload and index client documents with vector embeddings")

with st.sidebar:
    st.header("Client Settings")
    client_id = st.number_input("Client ID", min_value=1, step=1)
    st.markdown("---")
    st.info("Supported files: `.txt` (any encoding)")

uploaded_file = st.file_uploader("Upload text file", type=["txt"])

# -------------------- MAIN LOGIC --------------------
if uploaded_file and client_id:
    raw_bytes = uploaded_file.read()

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_bytes.decode("latin-1")

    file_hash = hashlib.sha256(content.encode()).hexdigest()

    db = SessionLocal()

    exists = db.execute(
        text("SELECT 1 FROM documents WHERE client_id=:cid AND file_hash=:fh"),
        {"cid": client_id, "fh": file_hash}
    ).fetchone()

    if exists:
        st.warning("⚠️ Duplicate file already exists for this client.")
    else:
        result = db.execute(
            text("""
                INSERT INTO documents (client_id, filename, file_hash)
                VALUES (:cid,:fn,:fh)
                RETURNING id
            """),
            {"cid": client_id, "fn": uploaded_file.name, "fh": file_hash}
        )
        document_id = result.fetchone()[0]

        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        progress = st.progress(0)

        for idx, chunk in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
            embedding = embed_text(chunk)

            db.execute(
                text("""
                    INSERT INTO document_chunks
                    (client_id, document_id, chunk_text, chunk_hash, embedding)
                    VALUES (:cid,:did,:ct,:ch,:emb)
                """),
                {
                    "cid": client_id,
                    "did": document_id,
                    "ct": chunk,
                    "ch": chunk_hash,
                    "emb": Vector(embedding)
                }
            )

            progress.progress((idx + 1) / len(chunks))

        db.commit()
        st.success("✅ File uploaded and indexed successfully!")
        st.balloons()
