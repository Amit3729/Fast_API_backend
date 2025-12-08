from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Optional
from datetime import datetime

class SourceInfo(BaseModel):
    source: str
    preview: str = ""

class UploadResponse(BaseModel):
    message: str
    file_type: str
    filename: str

class AskRequest(BaseModel):
    session_id: Optional[str] = Field(default="anonymous", description="Session identifier for conversation")
    query: str = Field(..., min_length=1, description="User query")


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []        
    session_id: str
    booking_detected: bool = False
    booking_data: Optional[dict] = None

class BookingRecord(BaseModel):
    name: str = Field(..., min_length=1)
    email: EmailStr
    date: str = Field(..., description="ISO date format YYYY-MM-DD")
    time: str = Field(..., description="Time in HH:MM format")
    session_id: str = Field(default="anonymous")
    created_at: Optional[datetime] = None

    @field_validator('date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Date must be in ISO format YYYY-MM-DD")

    @field_validator('time')
    @classmethod
    def validate_time(cls, v: str) -> str:
        try:
            parts = v.split(':')
            if len(parts) != 2:
                raise ValueError
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
            return v
        except (ValueError, AttributeError):
            raise ValueError("Time must be in HH:MM format")

class BookingResponse(BaseModel):
    success: bool
    booking_id: Optional[str] = None
    message: str
    booking: Optional[BookingRecord] = None