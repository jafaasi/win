"""
Simple API endpoint that reads predictions from the local engine via Supabase.
Render acts only as a read-only gateway.
"""
import json
from datetime import datetime
from backend.database import SessionLocal, AIBrainState

def get_state():
    """
    Read the latest prediction from local engine and return to frontend.
    """
    db = SessionLocal()
    try:
        # Get the latest prediction from local engine
        live_state = db.query(AIBrainState).filter(
            AIBrainState.model_name == "Live_UI_State"
        ).order_by(AIBrainState.id.desc()).first()
        
        if not live_state or not live_state.synaptic_weights:
            # No prediction yet - return placeholder
            return {
                "status": "waiting",
                "message": "Local engine not yet initialized. Start local_ai_engine.py on your Mac.",
                "currentIssue": None,
                "nextIssue": None,
                "prediction": None,
                "confidence": None,
                "source": "none",
                "latestIssue": None
            }
        
        try:
            prediction_data = json.loads(live_state.synaptic_weights)
            # Ensure all required fields exist
            prediction_data.setdefault("source", "local_engine")
            prediction_data.setdefault("status", "ready")
            return prediction_data
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"Invalid prediction JSON: {str(e)}",
                "source": "error"
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "source": "error"
        }
    
    finally:
        db.close()


# For testing via Vercel function
def handler(request):
    return get_state()


# When called directly
if __name__ == "__main__":
    state = get_state()
    print(json.dumps(state, indent=2))
