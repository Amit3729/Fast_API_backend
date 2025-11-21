from fastapi import APIRouter, HTTPException, Query
from app.schemas import BookingRecord, BookingResponse
from app.services.booking_service import BookingService
from app.utils.logger import get_logger
from typing import List, Optional

logger = get_logger(__name__)
router = APIRouter()
booking_service = BookingService()

@router.post("/schedule", response_model=BookingResponse)
async def schedule_interview(booking: BookingRecord):
    """
    Directly schedule an interview with complete booking information.
    
    Args:
        booking: Complete booking details (name, email, date, time)
    """
    try:
        booking_id = await booking_service.save_booking(booking.model_dump(), booking.session_id)
        
        return BookingResponse(
            success=True,
            booking_id=booking_id,
            message=f"Interview scheduled successfully for {booking.name} on {booking.date} at {booking.time}",
            booking=booking
        )
    except Exception as e:
        logger.error(f"Error scheduling interview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule interview: {str(e)}")

@router.get("/list", response_model=List[BookingRecord])
async def list_bookings(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of bookings to return")
):
    """
    List all bookings or filter by session_id.
    
    Args:
        session_id: Optional session filter
        limit: Maximum number of results
    """
    try:
        bookings = await booking_service.get_bookings(session_id=session_id, limit=limit)
        return bookings
    except Exception as e:
        logger.error(f"Error listing bookings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve bookings")

@router.get("/{booking_id}", response_model=BookingRecord)
async def get_booking(booking_id: str):
    """
    Get a specific booking by ID.
    
    Args:
        booking_id: Unique booking identifier
    """
    try:
        booking = await booking_service.get_booking_by_id(booking_id)
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        return booking
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving booking: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve booking")

@router.delete("/{booking_id}")
async def cancel_booking(booking_id: str):
    """
    Cancel/delete a booking.
    
    Args:
        booking_id: Unique booking identifier
    """
    try:
        success = await booking_service.delete_booking(booking_id)
        if not success:
            raise HTTPException(status_code=404, detail="Booking not found")
        return {"message": f"Booking {booking_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling booking: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel booking")