from fastapi import APIRouter, HTTPException
from typing import Optional, List
from app.schemas import AskRequest, AskResponse
from app.services.embeddings import embed_texts
from app.services.vector_store import search_similar
from app.services.redis_memory import add_message, get_messages
from app.services.llm_service import LLMService
from app.services.booking_service import BookingService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

llm_service = LLMService()
booking_service = BookingService()

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Handle RAG queries with booking detection.
    Supports multi-turn conversations via session_id.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    session_id = req.session_id or "anonymous"
    
    try:
        # Step 1: Check if query is booking-related
        booking_intent = await llm_service.detect_booking_intent(req.query)
        
        if booking_intent.get("is_booking", False):
            # Handle booking flow
            history = get_messages(session_id)
            booking_data = await booking_service.extract_booking_info(
                req.query, 
                history
            )
            
            if booking_data.get("complete", False):
                # Save booking
                booking_record = await booking_service.save_booking(
                    booking_data["data"],
                    session_id
                )
                answer = f"Great! I've scheduled your interview for {booking_data['data']['name']} on {booking_data['data']['date']} at {booking_data['data']['time']}. Confirmation will be sent to {booking_data['data']['email']}."
                
                # Save to memory
                add_message(session_id, "user", req.query)
                add_message(session_id, "assistant", answer)
                
                return AskResponse(
                    answer=answer,
                    sources=[],
                    session_id=session_id,
                    booking_detected=True,
                    booking_data=booking_data["data"]
                )
            else:
                # Need more info
                missing_fields = booking_data.get("missing_fields", [])
                answer = f"To schedule your interview, I need: {', '.join(missing_fields)}. Please provide these details."
                
                add_message(session_id, "user", req.query)
                add_message(session_id, "assistant", answer)
                
                return AskResponse(
                    answer=answer,
                    sources=[],
                    session_id=session_id,
                    booking_detected=True,
                    booking_data=None
                )
        
        # Step 2: Regular RAG flow
        # Embed query
        q_vec = embed_texts([req.query])[0]
        
        # Vector search
        hits = search_similar(q_vec, top_k=4)
        contexts = [h["text"] for h in hits if h.get("text")]
        
        # Get conversation history
        history = get_messages(session_id)
        history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history[-6:]])  # Last 3 turns
        
        # Build prompt
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are a helpful AI assistant. Use the provided context and conversation history to answer the user's question.

CONTEXT:
{context_str if context_str else "No relevant context found."}

CONVERSATION HISTORY:
{history_text if history_text else "No previous conversation."}

QUESTION:
{req.query}

Instructions:
- Answer concisely and accurately
- Reference which context you used (e.g., "Based on Context 1...")
- If the context doesn't contain relevant information, say so
- Maintain conversation continuity using the history"""

        # Call LLM
        answer = await llm_service.generate_answer(prompt)
        
        # Save to memory
        add_message(session_id, "user", req.query)
        add_message(session_id, "assistant", answer)
        
        # Extract sources
        source_list = [h["meta"].get("source", "Unknown") if h.get("meta") else "Unknown" for h in hits]
        unique_sources = list(dict.fromkeys([s for s in source_list if s != "Unknown"]))
        
        return AskResponse(
            answer=answer,
            sources=unique_sources,
            session_id=session_id,
            booking_detected=False,
            booking_data=None
        )
        
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    from app.services.redis_memory import clear_session
    try:
        clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear session")