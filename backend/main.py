from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from pathlib import Path
import os

app = FastAPI(title="Face Verification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[1]
TEMP_DIR = BASE_DIR / 'storage' / 'uploads'
TEMP_DIR.mkdir(parents=True, exist_ok=True)

from ai.face_embedding_manager import extract_embedding, save_embedding, verify_face


@app.post('/register')
async def register(user_id: str = Form(...), file: UploadFile = File(...)):
    # Save uploaded file
    file_ext = Path(file.filename).suffix
    tmp_name = f"{uuid.uuid4()}{file_ext}"
    tmp_path = TEMP_DIR / tmp_name
    with tmp_path.open('wb') as f:
        shutil.copyfileobj(file.file, f)
    try:
        emb = extract_embedding(str(tmp_path))
        save_embedding(user_id, emb)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"ok": True, "user_id": user_id}


@app.post('/verify')
async def verify(user_id: str = Form(...), file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix
    tmp_name = f"{uuid.uuid4()}{file_ext}"
    tmp_path = TEMP_DIR / tmp_name
    with tmp_path.open('wb') as f:
        shutil.copyfileobj(file.file, f)
    try:
        match, score = verify_face(str(tmp_path), user_id)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"match": match, "score": float(score) if score is not None else None}


@app.get('/users')
async def list_users():
    from ai.face_embedding_manager import load_embeddings
    data = load_embeddings()
    return {"count": len(data), "users": list(data.keys())}


# To run: uvicorn backend.main:app --reload --port 8000
