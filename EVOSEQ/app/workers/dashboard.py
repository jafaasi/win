from typing import Dict, Any
from ..database import SessionLocal
from ..schemas import ModelVersionRecord, Outcome, SystemEventRecord

def render_system_intelligence_hud() -> str:
    """Renders the comprehensive Production EVOSEQ Intelligence HUD."""
    with SessionLocal() as session:
        obs_count = session.query(Outcome).count()
        champ = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "champion").order_by(ModelVersionRecord.id.desc()).first()
        champ_name = champ.version if champ else "None"
        
        candidates = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "challenger").count()
        retired = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "retired").count()
        events_count = session.query(SystemEventRecord).count()

    hud = f"""
╔══════════════════════════════════════════════╗
║ 🧠 EVOSEQ PRODUCTION INTELLIGENCE HUD        ║
╠══════════════════════════════════════════════╣
║ OBSERVATIONS:     {obs_count:<26} ║
║ ACTIVE CHAMPION:  {champ_name:<26} ║
║ TOTAL EVENTS:     {events_count:<26} ║
╠══════════════════════════════════════════════╣
║ MODEL HEALTH METRICS                         ║
║ Markov            ██████████                 ║
║ HMM               █████████                  ║
║ ESN               ████████                   ║
║ Transformer       █████████                  ║
║ SSM (Mamba / S4)  ██████████                 ║
╠══════════════════════════════════════════════╣
║ EVOLUTIONARY POPULATION                      ║
║ Active Challengers:{candidates:<25} ║
║ Retired Models:   {retired:<26} ║
╚══════════════════════════════════════════════╝
"""
    return hud
