"""
Application Configuration
Simple settings management
"""

from typing import List


class Settings:
    """Application settings."""
    
    # Application
    APP_NAME: str = "Emotion Sense"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "*"
    ]
    
    # ML Model
    MODEL_PATH: str = "models/sentiment_model.pkl"
    MAX_TEXT_LENGTH: int = 5000
    BATCH_SIZE: int = 100


# Settings instance
settings = Settings()
