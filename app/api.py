from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.recommender import RealtimeRecommender
from app.utils import log_request

router = APIRouter()
recommender = RealtimeRecommender()


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="External user identifier")
    num_results: int = Field(10, ge=1, le=50)


class RecommendedItem(BaseModel):
    item_id: str
    score: float
    metadata: dict


class RecommendationResponse(BaseModel):
    user_id: str
    latency_ms: float
    items: List[RecommendedItem]


class UserFeatureRequest(BaseModel):
    features: List[float] = Field(..., description="Dense user vector matching model dim")


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(req: RecommendationRequest) -> RecommendationResponse:
    result = recommender.recommend(req.user_id, req.num_results)
    log_request(req.user_id, result["latency_ms"] / 1000)
    return RecommendationResponse(
        user_id=req.user_id,
        latency_ms=result["latency_ms"],
        items=result["items"],
    )


@router.post("/users/{user_id}/features")
async def upsert_user_features(user_id: str, req: UserFeatureRequest) -> dict:
    if not req.features:
        raise HTTPException(status_code=400, detail="Features payload required")
    recommender.feature_store.set_user_features(user_id, req.features)
    return {"user_id": user_id, "dimension": len(req.features)}
