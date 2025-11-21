from typing import Dict, Any, List
from openai import AsyncOpenAI
from app.utils.config import settings
from app.utils.logger import get_logger
import json

logger = get_logger(__name__)

class LLMService:
    """Service for LLM operations using OpenAI API"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4o-mini"
    
    async def generate_answer(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate answer using OpenAI LLM.
        
        Args:
            prompt: The complete prompt with context and question
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated answer text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    async def detect_booking_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect if query is related to booking an interview.
        
        Args:
            query: User query text
            
        Returns:
            Dict with 'is_booking' boolean and optional 'confidence' score
        """
        prompt = f"""Analyze if this query is about scheduling/booking an interview or appointment.

Query: "{query}"

Respond with ONLY a JSON object:
{{
  "is_booking": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}

Examples of booking queries:
- "I want to schedule an interview"
- "Can I book a meeting for tomorrow at 2pm?"
- "Schedule interview for John at john@email.com"
- "Book me for 25th December 3pm"

Examples of NON-booking queries:
- "What is the interview process?"
- "Tell me about your company"
- "How do I prepare for interviews?"
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            # Clean potential markdown formatting
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(result_text)
            
            return {
                "is_booking": result.get("is_booking", False),
                "confidence": result.get("confidence", 0.0),
                "reason": result.get("reason", "")
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse booking intent JSON: {str(e)}")
            return {"is_booking": False, "confidence": 0.0, "reason": "Parse error"}
        except Exception as e:
            logger.error(f"Error detecting booking intent: {str(e)}")
            return {"is_booking": False, "confidence": 0.0, "reason": str(e)}