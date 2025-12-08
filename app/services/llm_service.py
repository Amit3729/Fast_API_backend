import httpx
import json
from typing import Dict, Any
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class LLMService:
    """LLM Service using Groq (free tier, 500+ tokens/sec)"""
    
    def __init__(self):
        print("GROQ KEY IN USE:", settings.GROQ_API_KEY)
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = settings.GROQ_API_KEY  
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.model = "llama-3.1-8b-instant"  

    async def generate_answer(self, prompt: str, max_tokens: int = 500) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role":"system", "content":"Your are a helpful assistance."},
                {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Groq generate_answer error: {e}")
            return "Sorry, I'm having trouble responding right now."

    async def detect_booking_intent(self, query: str) -> Dict[str, Any]:
        prompt = f"""Analyze if this query is about scheduling/booking an interview.

Query: "{query}"

Respond with ONLY valid JSON:
{{
  "is_booking": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief reason"
}}

Booking examples: "schedule interview", "book meeting", "can we talk tomorrow"
Non-booking: "what is your experience", "tell me about yourself"
"""

        payload = {
            "model": "llama-3.1-8b-instant",  # Fast for intent detection
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.3
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(self.url, json=payload, headers=self.headers)
                response.raise_for_status()
                result_text = response.json()["choices"][0]["message"]["content"]
                
                # Clean JSON
                result_text = result_text.strip().replace("```json", "").replace("```", "").strip()
                result = json.loads(result_text)
                
                return {
                    "is_booking": bool(result.get("is_booking", False)),
                    "confidence": float(result.get("confidence", 0.0)),
                    "reason": str(result.get("reason", ""))
                }
        except Exception as e:
            logger.warning(f"Booking intent failed, using fallback: {e}")
            # Fallback: simple keyword check
            keywords = ["interview", "schedule", "book", "meet", "call", "talk", "hire"]
            is_booking = any(k in query.lower() for k in keywords)
            return {"is_booking": is_booking, "confidence": 0.9 if is_booking else 0.1, "reason": "fallback"}