from typing import Literal

from pydantic import BaseModel

Verdict = Literal["malicious", "benign"]


class AttckTag(BaseModel):
    tactic: str
    technique: str


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


class ModelMetric(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
