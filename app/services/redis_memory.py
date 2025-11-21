from typing import List, Dict, Any
import redis
import json
from app.utils.config import settings

r = redis.from_url(settings.REDIS_URL, decode_responses=True)

def add_message(session_id: str, role: str, text: str) -> None:
    key = f"conv:{session_id}"
    payload = json.dumps({"role": role, "text": text})
    r.rpush(key, payload)
    # Optional: set TTL for sessions, e.g., 30 days
    r.expire(key, 60 * 60 * 24 * 30)

def get_messages(session_id: str) -> List[Dict[str, str]]:
    key = f"conv:{session_id}"
    items = r.lrange(key, 0, -1)
    return [json.loads(i) for i in items]

def clear_session(session_id: str) -> None:
    r.delete(f"conv:{session_id}")
