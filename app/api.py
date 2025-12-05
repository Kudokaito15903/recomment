from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.recommender import RealtimeRecommender
from app.utils import log_request

router = APIRouter()
recommender = RealtimeRecommender()


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="External user identifier")
    num_results: int = Field(10, ge=1, le=50)
    use_cache: bool = Field(True, description="Whether to use cache")
    context: Optional[Dict] = Field(None, description="Additional context for recommendations")


class RecommendedItem(BaseModel):
    item_id: str
    score: float
    metadata: dict


class RecommendationResponse(BaseModel):
    user_id: str
    latency_ms: float
    items: List[RecommendedItem]
    cached: bool = False
    request_id: Optional[str] = None


class UserFeatureRequest(BaseModel):
    features: List[float] = Field(..., description="Dense user vector matching model dim")


class InteractionRequest(BaseModel):
    item_id: str = Field(..., description="Item identifier")
    interaction_type: str = Field(..., description="Type: click, purchase, view, etc.")
    position: Optional[int] = Field(None, description="Position in recommendation list")
    request_id: Optional[str] = Field(None, description="Original request ID")
    context: Optional[Dict] = None


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(req: RecommendationRequest) -> RecommendationResponse:
    result = recommender.recommend(
        req.user_id,
        req.num_results,
        use_cache=req.use_cache,
        context=req.context,
    )
    log_request(req.user_id, result["latency_ms"] / 1000)
    return RecommendationResponse(
        user_id=req.user_id,
        latency_ms=result["latency_ms"],
        items=result["items"],
        cached=result.get("cached", False),
        request_id=result.get("request_id"),
    )


@router.post("/interactions")
async def log_interaction(user_id: str, interaction: InteractionRequest) -> dict:
    """Log user interaction with recommended item (feedback loop)."""
    if recommender.feedback_logger:
        recommender.feedback_logger.log_interaction(
            user_id,
            interaction.item_id,
            interaction.interaction_type,
            interaction.request_id,
            interaction.position,
            interaction.context,
        )
        return {"status": "logged", "user_id": user_id, "item_id": interaction.item_id}
    return {"status": "feedback_disabled"}


@router.post("/users/{user_id}/features")
async def upsert_user_features(user_id: str, req: UserFeatureRequest) -> dict:
    if not req.features:
        raise HTTPException(status_code=400, detail="Features payload required")
    recommender.feature_store.set_user_features(user_id, req.features)
    return {"user_id": user_id, "dimension": len(req.features)}
