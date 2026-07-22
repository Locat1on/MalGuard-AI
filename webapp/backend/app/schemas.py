from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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
    # Row id in the history store. None for stub output or when persistence failed after a
    # genuine detection; the ML result remains valid but cannot link to a saved report.
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
    inferenceConcurrency: int
    detectionConcurrency: int
    apiKeyRequired: bool


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
    model: str = Field(min_length=1)
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class EvaluationMetric(ModelMetric):
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]

    @field_validator("confusion_matrix")
    @classmethod
    def validate_confusion_matrix(
        cls,
        matrix: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if any(value < 0 for row in matrix for value in row):
            raise ValueError("confusion matrix counts must be non-negative")
        return matrix


class EvaluationProtocol(BaseModel):
    dataset: str = Field(min_length=1)
    feature_dimensions: int = Field(gt=0)
    threshold: float = Field(ge=0, le=1)
    ensemble: str = Field(min_length=1)
    inference_batch_size: int = Field(gt=0)
    test_rows: int = Field(gt=0)
    test_malicious: int = Field(ge=0)
    test_benign: int = Field(ge=0)


class EvaluationArtifact(BaseModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)


class EvaluationRuntime(BaseModel):
    generated_at_utc: str = Field(min_length=1)
    python: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    packages: dict[str, str | None]
    cuda_available: bool
    cuda_device: str | None


class EvaluationGit(BaseModel):
    commit: str | None
    branch: str | None
    dirty: bool | None


class EvaluationManifest(BaseModel):
    protocol: EvaluationProtocol
    results: list[EvaluationMetric] = Field(min_length=3, max_length=3)
    artifacts: dict[str, EvaluationArtifact]
    runtime: EvaluationRuntime
    git: EvaluationGit

    @model_validator(mode="after")
    def validate_consistency(self) -> "EvaluationManifest":
        protocol = self.protocol
        if protocol.test_rows != protocol.test_malicious + protocol.test_benign:
            raise ValueError("test class counts do not add up to test_rows")
        if len({result.model for result in self.results}) != len(self.results):
            raise ValueError("evaluation model names must be unique")
        for result in self.results:
            benign_rows = sum(result.confusion_matrix[0])
            malicious_rows = sum(result.confusion_matrix[1])
            if (
                benign_rows != protocol.test_benign
                or malicious_rows != protocol.test_malicious
            ):
                raise ValueError(
                    f"{result.model} confusion matrix does not match class counts"
                )
        required_artifacts = {"lightgbm.txt", "mlp.pt", "scaler.pkl"}
        if not required_artifacts.issubset(self.artifacts):
            raise ValueError("evaluation artifacts are incomplete")
        return self
