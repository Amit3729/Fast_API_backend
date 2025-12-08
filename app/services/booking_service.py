import httpx
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.utils.config import settings
from app.utils.logger import get_logger
from app.services.db import bookings_collection
from app.schemas import BookingRecord
from bson import ObjectId

logger = get_logger(__name__)

class BookingService:
    def __init__(self):
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
        self.model = "llama-3.1-8b-instant"  # Fast & free

    async def extract_booking_info(self, query: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        history_text = "\n".join([f"{m['role']}: {m['text']}" for m in conversation_history[-6:]])
        
        prompt = f"""Extract booking details from this conversation.

CONVERSATION:
{history_text}

LATEST MESSAGE: {query}

Return ONLY valid JSON:
{{
  "complete": true/false,
  "data": {{"name": "str or null", "email": "str or null", "date": "YYYY-MM-DD or null", "time": "HH:MM or null"}},
  "missing_fields": ["name", "email", "date", "time"] filtered
}}

Rules:
- Only complete=true if ALL 4 fields are present
- Convert "tomorrow" → actual date
- Convert "3pm" → "15:00"
- Validate email format
"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(self.url, json=payload, headers=self.headers)
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"]
                text = text.replace("```json", "").replace("```", "").strip()
                result = json.loads(text)

                data = result.get("data", {})
                missing = [f for f in ["name", "email", "date", "time"] if not data.get(f)]
                result["complete"] = len(missing) == 0
                result["missing_fields"] = missing
                return result

        except Exception as e:
            logger.warning(f"Groq booking extraction failed: {e} → using rule-based fallback")
            return self._rule_based_fallback(query + " " + history_text)

    def _rule_based_fallback(self, text: str) -> Dict[str, Any]:
        text = text.lower()
        data = {"name": "", "email": "", "date": "", "time": ""}
        missing = []

        # Email
        email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
        data["email"] = email.group(0) if email else ""
        if not data["email"]: missing.append("email")

        # Name (simple)
        if "name" in text or "i am" in text or "my name" in text:
            name_match = re.search(r"(?:name is|my name|i am)[\s:]*([a-z\s]+)", text, re.I)
            data["name"] = name_match.group(1).strip().title() if name_match else "User"
        else:
            missing.append("name")

        # Date & Time (basic)
        date_match = re.search(r'\b(\d{4}-\d{2}-\d{2}|\d{1,2}[\/\-]\d{1,2}|tomorrow|dec \d+)', text)
        data["date"] = date_match.group(0) if date_match else ""
        if not data["date"]: missing.append("date")

        time_match = re.search(r'\b(\d{1,2}:\d{2}|\d{1,2}\s?(am|pm))\b', text)
        data["time"] = time_match.group(0) if time_match else ""
        if not data["time"]: missing.append("time")

        return {
            "complete": len(missing) == 0,
            "data": data,
            "missing_fields": missing
        }

    async def save_booking(self, booking_data: Dict[str, Any], session_id: str) -> str:
        try:
            record = {
                **booking_data,
                "session_id": session_id,
                "created_at": datetime.utcnow()
            }
            result = await bookings_collection.insert_one(record)
            logger.info(f"Booking saved: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB save failed: {e}")
            raise

    # get_bookings, get_booking_by_id, delete_booking → keep your existing ones
    
    async def get_bookings(
        self, 
        session_id: Optional[str] = None, 
        limit: int = 50
    ) -> List[BookingRecord]:
        """
        Retrieve bookings from database.
        
        Args:
            session_id: Optional filter by session
            limit: Max results
            
        Returns:
            List of BookingRecord objects
        """
        try:
            query = {"session_id": session_id} if session_id else {}
            cursor = bookings_collection.find(query).limit(limit).sort("created_at", -1)
            
            bookings = []
            async for doc in cursor:
                bookings.append(BookingRecord(
                    name=doc["name"],
                    email=doc["email"],
                    date=doc["date"],
                    time=doc["time"],
                    session_id=doc["session_id"],
                    created_at=doc.get("created_at")
                ))
            
            return bookings
            
        except Exception as e:
            logger.error(f"Error retrieving bookings: {str(e)}")
            raise
    
    async def get_booking_by_id(self, booking_id: str) -> Optional[BookingRecord]:
        """Get a specific booking by ID"""
        try:
            doc = await bookings_collection.find_one({"_id": ObjectId(booking_id)})
            if not doc:
                return None
            
            return BookingRecord(
                name=doc["name"],
                email=doc["email"],
                date=doc["date"],
                time=doc["time"],
                session_id=doc["session_id"],
                created_at=doc.get("created_at")
            )
        except Exception as e:
            logger.error(f"Error retrieving booking by ID: {str(e)}")
            raise
    
    async def delete_booking(self, booking_id: str) -> bool:
        """Delete a booking by ID"""
        try:
            result = await bookings_collection.delete_one({"_id": ObjectId(booking_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting booking: {str(e)}")
            raise