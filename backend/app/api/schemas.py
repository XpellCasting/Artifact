from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal
from app.modulos.estadistico.episodios import DEFAULT_QUIET_SECONDS
from app.services.application_common import DEFAULT_SUSPICION_THRESHOLD, MIN_SUSPICION_THRESHOLD, MAX_SUSPICION_THRESHOLD

class AnalyzeRequest(BaseModel):
    suspicion_quiet_seconds: int = Field(default=DEFAULT_QUIET_SECONDS, ge=0, le=86400)
    suspicion_threshold: float = Field(
        default=DEFAULT_SUSPICION_THRESHOLD,
        ge=MIN_SUSPICION_THRESHOLD,
        le=MAX_SUSPICION_THRESHOLD,
    )

class ContinuousConfigRequest(BaseModel):
    configs: list[dict[str, Any]] = Field(default_factory=list)

class ValidationReplayAdvanceRequest(BaseModel):
    suspicion_quiet_seconds: int = Field(default=DEFAULT_QUIET_SECONDS, ge=0, le=86400)
    batch_size: int = Field(default=1000, ge=1, le=20000)
    suspicion_threshold: float = Field(
        default=DEFAULT_SUSPICION_THRESHOLD,
        ge=MIN_SUSPICION_THRESHOLD,
        le=MAX_SUSPICION_THRESHOLD,
    )

class FlowRunRequest(BaseModel):
    candidate_limit: int = Field(default=1000, ge=1, le=20000)


class BuildInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotation_ids: list[str] = Field(default_factory=list, max_length=200)
    injection_ids: list[str] = Field(default_factory=list, max_length=500)
    window_seconds: int = Field(default=3600, ge=60, le=604800)
    stride_seconds: int = Field(default=1800, ge=60, le=604800)
    include_drafts: bool = False
    method: Literal["gat_lstm"] = "gat_lstm"
    source_ids: list[str] = Field(default_factory=list, max_length=10)
    gat_config: dict[str, Any] | None = None
