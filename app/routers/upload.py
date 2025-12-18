import os
import tempfile
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from app.schemas import UploadResponse
from app.services.chunker import chunk_text
from app.services.embeddings import embed_texts
from app.services.vector_store import save_vectors
from app.services.db import save_metadata
from app.utils.docx_extractor import extract_docx_text
from app.utils.pdf_extractor import extract_pdf_text
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


def process_document(file_path: str, original_name: str, strategy: str = "fixed"):
    try:
        logger.info(f"[BACKGROUND] Starting processing: {original_name}")

        # Extract
        if original_name.lower().endswith(".pdf"):
            text = extract_pdf_text(file_path)
        elif original_name.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif original_name.lower().endswith(".docx"):
            text = extract_docx_text(file_path)
        else:
            logger.error("Unsupported file")
            return

        logger.info(f"[BACKGROUND] Extracted {len(text):,} chars")

        # FORCE SIMPLE FAST CHUNKING 
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        logger.info(f"[BACKGROUND] Chunked into {len(chunks)} pieces (fast mode)")

        # Embedding
        logger.info("[BACKGROUND] Starting embedding...")
        embeddings = embed_texts(chunks)  
        logger.info(f"[BACKGROUND] Embedding done: {len(embeddings)} vectors")

        # Save to Qdrant
        metadata = {"source": original_name, "strategy": "fixed"}
        vector_ids = save_vectors(chunks, embeddings, metadata)
        logger.info(f"[BACKGROUND] Saved to Qdrant: {len(vector_ids)} vectors")

        
        logger.warning("[BACKGROUND] Skipping DB metadata save for speed")

        logger.info(f"[BACKGROUND] SUCCESS: {original_name} → {len(chunks)} chunks indexed!")

    except Exception as e:
        logger.error(f"[BACKGROUND] ERROR: {e}", exc_info=True)
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info("[BACKGROUND] Temp file deleted")



@router.post("/file", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    strategy: str = "fixed",
    background_tasks: BackgroundTasks = None,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    original_name = file.filename
    if not original_name.lower().endswith((".pdf", ".txt", ".docx")):
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, .docx allowed")

    suffix = os.path.splitext(original_name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
    except Exception:
        os.unlink(tmp.name)
        raise HTTPException(status_code=500, detail="Failed to save file")

    background_tasks.add_task(process_document, tmp.name, original_name, strategy)

    logger.info(f"File queued: {original_name} (strategy: {strategy})")
    return UploadResponse(
        message="File uploaded. Processing in background...",
        filename=original_name,
        file_type=suffix[1:],
        strategy=strategy
    )