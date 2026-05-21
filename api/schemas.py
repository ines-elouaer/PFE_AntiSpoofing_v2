from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    api_version: str
    video_checkpoint: str
    behavior_pose_model: str
    face_landmarker_model: str
    fusion_formula: str
    decisions: List[str]


class ChallengeStartRequest(BaseModel):
    challenge_type: Optional[str] = None


class ChallengeResponse(BaseModel):
    status: str
    session_id: str
    challenge_id: str
    challenge_type: str
    instruction: str
    expires_in_seconds: int
    expires_at: str
    source: str


class GenericResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class AnalyzeVideoResponse(BaseModel):
    type: Optional[str] = None
    branch: Optional[str] = None
    score: Optional[float] = None
    label: Optional[str] = None
    decision: Optional[str] = None
    next_action: Optional[str] = None
    liveness: Optional[Dict[str, Any]] = None
    model_called: Optional[bool] = None
    model_details: Optional[Dict[str, Any]] = None
    video_quality_score: Optional[float] = None
    message: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    challenge_id: Optional[str] = None
    challenge: Optional[Dict[str, Any]] = None
    original_input: Optional[str] = None
    saved_video_path: Optional[str] = None
    saved_result_path: Optional[str] = None