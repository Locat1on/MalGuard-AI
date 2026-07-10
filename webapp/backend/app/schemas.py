from typing import Literal

from pydantic import BaseModel

Verdict = Literal["malicious", "benign"]


class AttckTag(BaseModel):
    tactic: str
    technique: str


class FeatureAttention(BaseModel):
    """One of the MLP's 12 feature groups and the softmax attention weight it received while
    fusing this sample — i.e. how much this group drove the model's decision. The weights over
    all 12 groups sum to 1. This is the real explainability signal (gradcamUrl stays None)."""

    group: str   # raw EMBER group name, e.g. "imports"
    label: str   # human-readable Chinese label for display
    weight: float


class DetectionResult(BaseModel):
    filename: str
    verdict: Verdict
    confidence: float
    family: str | None
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
    lgbmScore: float
    mlpScore: float
    modelAgreement: Literal["agree", "disagree"]
    llmVerdict: Verdict | None
    llmConfidence: float | None
    llmReport: str
    attck: list[AttckTag]


class ModelMetric(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
