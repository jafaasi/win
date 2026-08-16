import os
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Settings:
    database_url: str
    batch_size: int = 500
    feature_window: int = 256
    evaluation_window: int = 5000
    drift_window: int = 1000

def load_settings() -> Settings:
    # Prefer explicit DATABASE_URL, with local fallback for testing
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evoseq_local.db")
        database_url = f"sqlite:///{db_path}"
        
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        
    return Settings(
        database_url=database_url,
        batch_size=int(os.getenv("BATCH_SIZE", "500")),
        feature_window=int(os.getenv("FEATURE_WINDOW", "256")),
        evaluation_window=int(os.getenv("EVALUATION_WINDOW", "5000")),
        drift_window=int(os.getenv("DRIFT_WINDOW", "1000")),
    )
