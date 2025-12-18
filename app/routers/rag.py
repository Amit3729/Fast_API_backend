from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from pydantic import BaseModel
import asyncio

from app.schemas import AskRequest, AskResponse
from app.services.embeddings import embed_texts
from app.services.vector_store import search_similar
from app.services.redis_memory import add_message, get_messages, clear_session
from app.services.llm_service import LLMService
from app.services.booking_service import BookingService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG & Booking"])

# Singletons — instantiated once
llm_service = LLMService()
booking_service = BookingService()


class SourceInfo(BaseModel):
    source: str
    preview: str = ""


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, background_tasks: BackgroundTasks = None):
    """
    Main RAG + Booking endpoint with conversation memory
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = req.session_id or "anonymous"

    try:
        # === BOOKING DETECTION ===
        booking_intent = await llm_service.detect_booking_intent(query)

        if booking_intent.get("is_booking", False):
            history = get_messages(session_id)
            booking_data = await booking_service.extract_booking_info(query, history)

            if booking_data.get("complete", False):
                # Finalize booking
                booking_record = await booking_service.save_booking(
                    booking_data["data"], session_id
                )

                answer = (
                    f"Great! I've scheduled your interview for **{booking_data['data']['name']}** "
                    f"on **{booking_data['data']['date']}** at **{booking_data['data']['time']}**. "
                    f"A confirmation has been sent to {booking_data['data']['email']}."
                )

                add_message(session_id, "user", query)
                add_message(session_id, "assistant", answer)

                return AskResponse(
                    answer=answer,
                    sources=[],
                    session_id=session_id,
                    booking_detected=True,
                    booking_data=booking_data["data"],
                )

            # Still collecting info
            missing = booking_data.get("missing_fields", [])
            answer = f"Sure! To complete your interview booking, please provide: **{', '.join(missing)}**."

            add_message(session_id, "user", query)
            add_message(session_id, "assistant", answer)

            return AskResponse(
                answer=answer,
                sources=[],
                session_id=session_id,
                booking_detected=True,
                booking_data=None,
            )

        # === REGULAR RAG FLOW ===
        query_vector = embed_texts([query])[0]
        hits = search_similar(query_vector, top_k=5)  

        contexts = []
        sources = []
        for hit in hits:
            text = hit.get("text", "")
            meta = hit.get("meta", {})
            source = meta.get("source", "Unknown document")
            if text.strip():
                contexts.append(text.strip())
                sources.append({
                    "source": source,
                    "preview": text.strip()[:150] + ("..." if len(text.strip()) > 150 else "")
                    })

            
        # Conversation history (last 6 messages = 3 turns)
        history_msgs = get_messages(session_id)[-6:]
        history_text = "\n".join([f"{m['role'].title()}: {m['text']}" for m in history_msgs]) if history_msgs else "None"

        context_block = "\n\n".join([f"Document: {s['source']}\nContent: {ctx}" for ctx, s in zip(contexts, sources)]) if contexts else "No relevant documents found."

        prompt = f"""You are an expert assistant. Answer the user's question using only the provided context and conversation history.

RELEVANT DOCUMENTS:
{context_block}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer clearly and concisely
- Cite sources like: "According to [Document: resume.pdf]..."
- If no relevant info, say: "I don't have enough information from the uploaded documents to answer that."
- Be friendly and professional"""

        #Add timeout protection
        try:
            answer = await asyncio.wait_for(llm_service.generate_answer(prompt), timeout=30.0)
        except asyncio.TimeoutError:
            answer = "Sorry, the response took too long. Please try again."

        # Save conversation
        add_message(session_id, "user", query)
        add_message(session_id, "assistant", answer)

        # Dedupe sources
        unique_sources = []
        seen = set()
        for s in sources:
            if s['source'] not in seen:
                unique_sources.append(s)
                seen.add(s['source'])

        return AskResponse(
            answer=answer,
            sources=unique_sources,
            session_id=session_id,
            booking_detected=False,
            booking_data=None,
        )

    except Exception as e:
        logger.error(f"RAG endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


@router.delete("/session/{session_id}")
async def clear_session_route(session_id: str):
    """Clear conversation memory"""
    try:
        clear_session(session_id)
        return {"detail": f"Session '{session_id}' cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session")


@router.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """Debug endpoint — see conversation history"""
    return {"session_id": session_id, "history": get_messages(session_id)}