import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.schemas import UploadResponse
from app.services.chunker import chunk_text
from app.services.embeddings import embed_texts
from app.services.vector_store import save_vectors
from app.services.db import save_metadata
from app.utils.docx_extractor import extract_docx_text
from app.utils.pdf_extractor import extract_pdf_text
from app.utils.logger import get_logger
import asyncio

logger = get_logger(__name__)
router = APIRouter()

async def process_file_async(tmp_path: str, filename: str, strategy: str):
    """Async background processing function"""
    try:
        # EXTRACT TEXT
        if filename.endswith(".pdf"):
            text = extract_pdf_text(tmp_path)
        elif filename.endswith(".txt"):
            with open(tmp_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".docx"):
            text = extract_docx_text(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        if not text.strip():
            logger.warning(f"No text extracted from {filename}")
            return

        # CHUNK
        chunks = chunk_text(text, strategy=strategy)
        logger.info(f"Created {len(chunks)} chunks from {filename}")

        # EMBED
        embeddings = embed_texts(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # SAVE IN QDRANT (fixed parameter order)
        metadata_base = {
            "source": filename,
            "strategy": strategy,
            "total_chunks": len(chunks)
        }
        ids = save_vectors(chunks, embeddings, metadata_base)
        logger.info(f"Saved vectors to Qdrant with IDs: {ids}")

        # SAVE METADATA
        await save_metadata({
            "file_name": filename,
            "total_chunks": len(chunks),
            "strategy": strategy,
            "vector_ids": ids,
            "text_preview": text[:200]
        })
        logger.info(f"Saved metadata for {filename}")

    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
    finally:
        # CLEANUP
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Cleaned up temp file: {tmp_path}")

def process_file_background(tmp_path: str, filename: str, strategy: str):
    """Wrapper to run async function in background"""
    asyncio.run(process_file_async(tmp_path, filename, strategy))

@router.post("/file", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: str = "fixed"
):
    """
    Upload and process documents (.pdf, .txt, .docx)
    
    Args:
        file: Document file to upload
        strategy: Chunking strategy - 'fixed', 'simple', or 'paragraph'
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    filename = file.filename.lower()

    # VALIDATE FILE TYPE
    valid_extensions = (".pdf", ".txt", ".docx")
    if not filename.endswith(valid_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Only {', '.join(valid_extensions)} files are supported"
        )

    # VALIDATE STRATEGY
    valid_strategies = ["fixed", "simple", "paragraph"]
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Use one of: {', '.join(valid_strategies)}"
        )

    # SAVE TO TEMPORARY FILE
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="File is empty")
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        logger.error(f"Error saving temp file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # ADD BACKGROUND TASK
    #background_tasks.add_task(process_file_background, tmp_path, filename, strategy)
    await process_file_async(tmp_path, filename, strategy)   # inline
    
    logger.info(f"File '{filename}' queued for processing with strategy '{strategy}'")

    return UploadResponse(
        message=f"File '{filename}' uploaded successfully. Processing in background.",
        file_type=filename.split(".")[-1],
        filename=filename
    )

