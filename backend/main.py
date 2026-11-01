"""
Emotion Sense - Main Application
FastAPI backend for sentiment analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone
import uuid
import uvicorn

from ml.model import SentimentModel
from core.config import settings

app = FastAPI(
    title="Emotion Sense API",
    description="Explainable emotion scoring API - every prediction lists the words behind it",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ML Model instance
model = SentimentModel()

# In-memory storage (use database in production)
analysis_history: List[Dict] = []


# Request/Response Models
class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"text": "I absolutely love this product! It's amazing!"}
        }
    )

    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(
        ..., min_length=1, max_length=100, description="List of texts to analyze"
    )


class EmotionScores(BaseModel):
    happy: float
    sad: float
    angry: float
    fear: float
    surprise: float
    neutral: float


class AnalysisResult(BaseModel):
    id: str
    text: str
    emotion: str
    confidence: float
    scores: EmotionScores
    timestamp: str
    signals: List[Dict] = Field(default_factory=list)
    engine: str = "lexicon"


class AnalyzeResponse(BaseModel):
    status: str
    data: AnalysisResult


class BatchAnalyzeResponse(BaseModel):
    status: str
    results: List[AnalysisResult]
    summary: Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# API Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Emotion Sense API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded,
        version="1.0.0"
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze sentiment of a single text.
    
    Returns emotion prediction with confidence scores.
    """
    try:
        # Analyze text
        result = model.analyze(request.text)
        
        # Create response
        analysis = AnalysisResult(
            id=str(uuid.uuid4()),
            text=request.text,
            emotion=result["emotion"],
            confidence=result["confidence"],
            scores=EmotionScores(**result["scores"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            signals=result.get("signals", []),
            engine=result.get("engine", "lexicon"),
        )
        
        # Save to history
        analysis_history.append(analysis.model_dump())
        
        return AnalyzeResponse(status="success", data=analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Analysis"])
async def analyze_batch(request: BatchAnalyzeRequest):
    """
    Analyze sentiment of multiple texts.
    
    Returns analysis results for each text plus summary statistics.
    """
    try:
        results = []
        emotion_counts = {}
        
        for text in request.texts:
            result = model.analyze(text)
            
            analysis = AnalysisResult(
                id=str(uuid.uuid4()),
                text=text,
                emotion=result["emotion"],
                confidence=result["confidence"],
                scores=EmotionScores(**result["scores"]),
                timestamp=datetime.now(timezone.utc).isoformat(),
                signals=result.get("signals", []),
                engine=result.get("engine", "lexicon"),
            )
            
            results.append(analysis)
            analysis_history.append(analysis.model_dump())
            
            # Count emotions for summary
            emotion_counts[result["emotion"]] = emotion_counts.get(result["emotion"], 0) + 1
        
        # Calculate summary
        total = len(request.texts)
        summary = {
            "total_analyzed": total,
            "emotion_distribution": {
                emotion: {
                    "count": count,
                    "percentage": round(count / total * 100, 2)
                }
                for emotion, count in emotion_counts.items()
            },
            "avg_confidence": round(
                sum(r.confidence for r in results) / total, 3
            )
        }
        
        return BatchAnalyzeResponse(status="success", results=results, summary=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history", tags=["History"])
async def get_history(page: int = 1, limit: int = 10):
    """
    Get analysis history with pagination.
    """
    start = (page - 1) * limit
    end = start + limit
    
    paginated = analysis_history[start:end]
    
    return {
        "status": "success",
        "data": paginated,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": len(analysis_history),
            "pages": (len(analysis_history) + limit - 1) // limit
        }
    }


@app.delete("/api/v1/history", tags=["History"])
async def clear_history():
    """Clear all analysis history."""
    global analysis_history
    analysis_history = []
    return {"status": "success", "message": "History cleared"}


@app.get("/api/v1/stats", tags=["Statistics"])
async def get_stats():
    """Get overall statistics."""
    if not analysis_history:
        return {
            "status": "success",
            "data": {
                "total_analyses": 0,
                "emotion_counts": {},
                "avg_confidence": 0
            }
        }
    
    emotion_counts = {}
    total_confidence = 0
    
    for item in analysis_history:
        emotion = item["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        total_confidence += item["confidence"]
    
    return {
        "status": "success",
        "data": {
            "total_analyses": len(analysis_history),
            "emotion_counts": emotion_counts,
            "avg_confidence": round(total_confidence / len(analysis_history), 3)
        }
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
