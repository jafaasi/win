import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base
from app.evolution.registry import ModelRegistry
from app.meta.questions import ResearchQuestionManager

def bootstrap():
    print("🚀 Initializing EVOSEQ database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Core, Evolution, Research & Meta tables created.")
    
    registry = ModelRegistry()
    registry.ensure_initial_population()
    print("✅ Initial model population initialized.")
    
    qm = ResearchQuestionManager()
    print("✅ Pre-registered research questions agenda initialized.")
    print("🎉 EVOSEQ Bootstrap Complete!")

if __name__ == "__main__":
    bootstrap()
