import streamlit as st
import hashlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from openai import OpenAI
import os

# -------------------- ENV --------------------
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- DB --------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# -------------------- AI --------------------
openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENAI_API_KEY)

def embed_text(text_value: str):
    return openai_client.embeddings.create(
        model="mistralai/mixtral-8x7b-instruct",
        input=text_value
    ).data[0].embedding

# -------------------- STREAMLIT --------------------
st.title("Client Knowledge Base Manager")

client_id = st.number_input("Client ID", min_value=1, step=1)
uploaded_file = st.file_uploader("Upload text file")

if uploaded_file and client_id:
    content = uploaded_file.read().decode("utf-8")
    file_hash = hashlib.sha256(content.encode()).hexdigest()

    db = SessionLocal()

    # Check duplicate
    exists = db.execute(
        "SELECT 1 FROM documents WHERE client_id=:cid AND file_hash=:fh",
        {"cid": client_id, "fh": file_hash}
    ).fetchone()

    if exists:
        st.warning("Duplicate file already exists.")
    else:
        # Insert document
        result = db.execute(
            "INSERT INTO documents (client_id, filename, file_hash) VALUES (:cid,:fn,:fh) RETURNING id",
            {"cid": client_id, "fn": uploaded_file.name, "fh": file_hash}
        )
        document_id = result.fetchone()[0]

        # Chunk + embed
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
            embedding = embed_text(chunk)

            db.execute("""
                INSERT INTO document_chunks
                (client_id, document_id, chunk_text, chunk_hash, embedding)
                VALUES (:cid,:did,:ct,:ch,:emb)
            """, {
                "cid": client_id,
                "did": document_id,
                "ct": chunk,
                "ch": chunk_hash,
                "emb": embedding
            })

        db.commit()
        st.success("File uploaded and indexed successfully.")
