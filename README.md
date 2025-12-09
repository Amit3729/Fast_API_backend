# Private RAG + Interview Booking Assistant

A  RAG system with interview scheduling — built from scratch with FastAPI.

## Features

- Upload PDF/DOCX resumes → background processing
- Local embeddings using **all-MiniLM-L6-v2** (384-dim, private, free)
- Vector search with **Qdrant**
- Conversational RAG with **Groq Llama 3.3 70B** (free tier, 500+ tokens/sec)
- Multi-turn conversation memory via **Redis**
- Intelligent interview booking system (multi-turn extraction)
- Direct booking API + full CRUD
- Structured JSON responses with sources
- Clean, modular, typed, senior-level code

## Tech Stack

| Layer               | Technology                     |
|---------------------|--------------------------------|
| Backend             | FastAPI + Uvicorn              |
| Embeddings          | `sentence-transformers` (local) |
| LLM                 | Groq (Llama 3.3 70B)            |
| Vector DB           | Qdrant                         |
| Memory              | Redis                          |
| Persistence         | MongoDB (async via Motor)      |
| Document Processing | PyMuPDF + python-docx          |

## API Endpoints

```text
POST   /upload/upload/file     → Upload resume (PDF/DOCX)
POST   /rag/ask                → Ask questions OR schedule interview
POST   /booking/schedule       → Direct booking (optional)
GET    /booking/list           → List all bookings