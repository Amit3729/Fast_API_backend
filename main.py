import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI
from app.routers import upload, rag, booking
from app.utils.logger import setup_logging

setup_logging()
app = FastAPI(title="FastAPI RAG with Interview Booking")

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(rag.router, prefix="/rag", tags=["rag"])
app.include_router(booking.router, prefix="/booking", tags=["booking"])

@app.get("/")
async def root():
    return {
        "message": "FastAPI RAG System with Interview Booking",
        "endpoints": {
            "upload": "/upload/file",
            "rag": "/rag/ask",
            "booking": "/booking/schedule"
        }
    }

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)