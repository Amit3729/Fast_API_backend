from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import AsyncOpenAI
from app.utils.config import settings
from app.utils.logger import get_logger
from app.services.db import bookings_collection
from app.schemas import BookingRecord
import json
from bson import ObjectId

logger = get_logger(__name__)

class BookingService:
    """Service for handling interview booking logic"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4o-mini"
    
    async def extract_booking_info(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Extract booking information from query and conversation history using LLM.
        
        Args:
            query: Current user query
            conversation_history: Previous conversation messages
            
        Returns:
            Dict with 'complete' boolean, 'data' dict, and 'missing_fields' list
        """
        history_text = "\n".join([
            f"{msg['role']}: {msg['text']}" 
            for msg in conversation_history[-10:]  # Last 5 turns
        ])
        
        prompt = f"""Extract booking information from the conversation. Look for:
- name: Full name of the person
- email: Email address
- date: Date in YYYY-MM-DD format
- time: Time in HH:MM format (24-hour)

CONVERSATION HISTORY:
{history_text if history_text else "No previous conversation"}

CURRENT MESSAGE:
{query}

Respond with ONLY a JSON object:
{{
  "complete": true/false,
  "data": {{
    "name": "extracted name or null",
    "email": "extracted email or null",
    "date": "YYYY-MM-DD or null",
    "time": "HH:MM or null"
  }},
  "missing_fields": ["list", "of", "missing", "fields"]
}}

Important:
- Mark complete=true ONLY if ALL four fields are present
- Convert dates to YYYY-MM-DD (e.g., "tomorrow" → calculate date, "25th Dec" → 2025-12-25)
- Convert times to 24-hour HH:MM (e.g., "2pm" → "14:00", "3:30pm" → "15:30")
- Be lenient with name formats
- Validate email format
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(result_text)
            
            # Validate completeness
            data = result.get("data", {})
            required_fields = ["name", "email", "date", "time"]
            missing = [f for f in required_fields if not data.get(f)]
            
            return {
                "complete": len(missing) == 0,
                "data": data,
                "missing_fields": missing
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse booking extraction JSON: {str(e)}")
            return {
                "complete": False,
                "data": {},
                "missing_fields": ["name", "email", "date", "time"]
            }
        except Exception as e:
            logger.error(f"Error extracting booking info: {str(e)}")
            return {
                "complete": False,
                "data": {},
                "missing_fields": ["name", "email", "date", "time"]
            }
    
    async def save_booking(self, booking_data: Dict[str, Any], session_id: str) -> str:
        """
        Save booking to database.
        
        Args:
            booking_data: Dict with name, email, date, time
            session_id: Session identifier
            
        Returns:
            Booking ID (MongoDB ObjectId as string)
        """
        try:
            booking_record = {
                "name": booking_data["name"],
                "email": booking_data["email"],
                "date": booking_data["date"],
                "time": booking_data["time"],
                "session_id": session_id,
                "created_at": datetime.utcnow()
            }
            
            result = await bookings_collection.insert_one(booking_record)
            logger.info(f"Saved booking: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving booking: {str(e)}")
            raise
    
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