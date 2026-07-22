from typing import Literal

from pydantic import BaseModel

Verdict = Literal["malicious", "benign"]


class AttckTag(BaseModel):
    tactic: str
    technique: str


class FeatureAttention(BaseModel):
    """One of the MLP's 12 feature groups and its softmax fusion weight for this sample.

    The 12 weights sum to 1. They are a useful model-internal diagnostic signal, not a
    causal attribution of why the verdict occurred.
    """

    group: str   # raw EMBER group name, e.g. "imports"
    label: str   # human-readable Chinese label for display
    weight: float


class DetectionResult(BaseModel):
    filename: str
    verdict: Verdict
    confidence: float
    family: str | None
    # Softmax probability of the reported family (0-1). None whenever family is None (benign,
    # catch-all, or below the confidence floor). Shown alongside the name so a moderate-confidence
    # guess reads as "suspected X (62%)" rather than a definitive attribution.
    familyConfidence: float | None = None
    gradcamUrl: str | None
    attck: list[AttckTag]
    llmReport: str
    modelAgreement: Literal["agree", "disagree"]
    lgbmScore: float
    mlpScore: float
    llmVerdict: Verdict | None
    llmConfidence: float | None
    # Row id of this detection in the history store (None only for the stub path, which is
    # not persisted). Lets the frontend link straight to GET /api/history/{id}/report.
    historyId: int | None = None
    # Per-feature-group attention weights (the model's "why"). Populated on single-file
    # detection; None on the batch path, which skips the analysis layer.
    featureAttention: list[FeatureAttention] | None = None


class BatchItem(BaseModel):
    """One file's result inside a batch scan. `ok=False` carries `error` and null scores
    (e.g. the file wasn't a parseable PE); `ok=True` carries the ML verdict. The LLM and
    ATT&CK layers are deliberately skipped in batch mode, so those fields don't appear here."""

    filename: str
    ok: bool
    verdict: Verdict | None
    confidence: float | None
    family: str | None
    familyConfidence: float | None
    lgbmScore: float | None
    mlpScore: float | None
    modelAgreement: Literal["agree", "disagree"] | None
    historyId: int | None
    error: str | None


class BatchDetectionResult(BaseModel):
    items: list[BatchItem]
    total: int
    malicious: int
    benign: int
    failed: int


class HistoryRecord(BaseModel):
    id: int
    createdAt: str
    filename: str
    sha256: str
    source: Literal["single", "batch"]
    verdict: Verdict
    confidence: float
    family: str | None
    familyConfidence: float | None
    lgbmScore: float
    mlpScore: float
    modelAgreement: Literal["agree", "disagree"]
    llmVerdict: Verdict | None
    llmConfidence: float | None
    llmReport: str
    attck: list[AttckTag]


class HealthStatus(BaseModel):
    ok: bool
    ready: bool
    mode: Literal["real", "stub", "unavailable"]
    modelsLoaded: bool
    familyModelLoaded: bool
    llmConfigured: bool
    modelLoadError: str | None
    familyModelLoadError: str | None
    modelProvenanceVerified: bool | None
    modelProvenanceWarning: str | None


class HistoryStats(BaseModel):
    total: int
    malicious: int
    benign: int
    single: int
    batch: int
    modelDisagreements: int
    llmCompared: int
    llmDisagreements: int
    lastCreatedAt: str | None


class ModelMetric(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
