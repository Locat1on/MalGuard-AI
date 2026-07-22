"""Detection predictor.

Real path: extract EMBER-schema features from the uploaded PE file (src/features/extract.py),
run the trained LightGBM + MLP models (checkpoints/), and report agreement between the two.

A hash-based stub exists only for explicit interface development via
ALLOW_STUB_PREDICTIONS=1; normal runs fail visibly when real checkpoints are unavailable.
"""

import hashlib
import json
import os
import pickle
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import torch

from app.schemas import AttckTag, DetectionResult, FeatureAttention
from app.settings import settings
from src.config import load_config
from src.features.extract import SEGMENT_DIMS, extract_features
from src.llm.feature_summary import summarize
from src.llm.report import generate_report
from src.models.family_checkpoint import OTHER_LABEL, unpack_family_checkpoint
from src.models.mlp import MalwareMLP
from src.reproducibility import sha256_file

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
EVALUATION_MANIFEST_PATH = CHECKPOINTS_DIR / "evaluation_manifest.json"


# Human-readable Chinese labels for the 12 EMBER feature groups, so the fusion-weight
# signal is display-ready without the frontend hardcoding this mapping.
FEATURE_GROUP_LABELS = {
    "general": "通用文件属性",
    "histogram": "字节直方图",
    "byteentropy": "字节熵分布",
    "strings": "可打印字符串统计",
    "header": "PE 头部字段",
    "section": "节区信息",
    "imports": "导入表 (API)",
    "exports": "导出表",
    "datadirectories": "数据目录",
    "richheader": "Rich 头",
    "authenticode": "数字签名",
    "pefilewarnings": "PE 解析告警",
}



def verify_evaluated_artifacts(
    manifest_path: Path,
    artifacts: dict[str, Path],
) -> tuple[bool | None, str | None]:
    """Check that deployed artifacts are exactly those used by the published evaluation."""
    if not manifest_path.exists():
        return None, "缺少 evaluation_manifest.json，无法核验当前模型与正式指标的一致性。"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest["artifacts"]
        mismatches = []
        for name, path in artifacts.items():
            expected_hash = expected.get(name, {}).get("sha256")
            if not expected_hash:
                mismatches.append(f"{name} 缺少评估哈希")
            elif sha256_file(path) != expected_hash:
                mismatches.append(f"{name} 与评估版本不一致")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        return None, f"评估来源清单不可用（{type(error).__name__}: {error}）。"
    if mismatches:
        return False, "；".join(mismatches) + "。请重新运行正式评估。"
    return True, None

class ModelUnavailableError(Exception):
    """Raised when real detector checkpoints are unavailable or incompatible."""


class FeatureExtractionError(Exception):
    """Raised when a file's PE structure could not be parsed into a feature vector.

    Some legitimately-signed files (and some malformed ones) trigger bugs in third-party
    parsing libraries (e.g. thrember's Authenticode certificate-chain parsing crashes with
    TypeError on certain certificate stores). Rather than let that surface as a raw 500,
    the router turns this into an honest 422 with a clear message.
    """


def _validate_probability_vector(
    values: object,
    expected_count: int,
    model_name: str,
) -> np.ndarray:
    """Reject malformed model output instead of turning it into a false verdict."""
    try:
        probabilities = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelUnavailableError(
            f"{model_name} 推理输出无法转换为概率。"
        ) from error
    if probabilities.shape != (expected_count,):
        raise ModelUnavailableError(
            f"{model_name} 推理输出形状无效：期望 {(expected_count,)}，"
            f"实际 {probabilities.shape}。"
        )
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ModelUnavailableError(
            f"{model_name} 推理输出包含非有限值或超出 0 到 1 的概率。"
        )
    return probabilities


class Predictor:
    def __init__(self) -> None:
        # Stub output is opt-in: normal runs must fail visibly when real checkpoints are absent
        # instead of returning plausible-looking hash-based fake detections.
        self.stub_enabled = os.environ.get("ALLOW_STUB_PREDICTIONS") == "1"
        self.model_load_error: str | None = None
        self.family_model_load_error: str | None = None
        self.model_provenance_verified: bool | None = None
        self.model_provenance_warning: str | None = None
        self.inference_concurrency = settings.inference_concurrency
        self._inference_slots = threading.BoundedSemaphore(self.inference_concurrency)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self.models_loaded = self._try_load_models()
        self.family_model_loaded = self._try_load_family_model()

    def _try_load_family_model(self) -> bool:
        try:
            return self._load_family_model()
        except Exception as e:
            self.family_model_load_error = f"{type(e).__name__}: {e}"
            return False

    def _load_family_model(self) -> bool:
        """Optional: the family classifier is a separate, later-added model (see
        src/models/train_family.py). Detection itself must keep working even if it hasn't been
        trained yet — `predict()` just reports family=None in that case.

        Reuses the same MalwareMLP architecture and checkpoints/scaler.pkl as the binary
        detector (see train_family.py for why the scaler is shared rather than refit), just with
        its output layer widened to the family class count.
        """
        family_model_path = CHECKPOINTS_DIR / "family_mlp.pt"
        family_labels_path = CHECKPOINTS_DIR / "family_labels.json"
        if not family_model_path.exists():
            self.family_model_load_error = "缺少 family_mlp.pt。"
            return False

        checkpoint = torch.load(
            family_model_path, map_location=self.device, weights_only=True
        )
        state_dict, self.family_labels = unpack_family_checkpoint(
            checkpoint, family_labels_path
        )

        family_config = load_config("family")
        self.family_confidence_floor: float = family_config["family_confidence_floor"]
        self.family_model = MalwareMLP(
            hidden_dims=family_config["hidden_dims"],
            dropout=family_config["dropout"],
            embed_dim=family_config["embed_dim"],
            num_classes=len(self.family_labels),
        ).to(self.device)
        self.family_model.load_state_dict(state_dict)
        self.family_model.eval()
        return True

    def _decode_family_probability(
        self, probabilities: np.ndarray
    ) -> tuple[str | None, float | None]:
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        name = self.family_labels[index]
        if name == OTHER_LABEL or confidence < self.family_confidence_floor:
            return None, None
        return name, round(confidence, 4)

    def _predict_family(
        self, features_scaled: np.ndarray
    ) -> tuple[str | None, float | None]:
        """Return a family suspicion only when the model is sufficiently confident."""
        with torch.inference_mode():
            tensor = torch.from_numpy(
                features_scaled.astype(np.float32, copy=False)
            ).to(self.device)
            logits = self.family_model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return self._decode_family_probability(probabilities)

    def _try_load_models(self) -> bool:
        try:
            return self._load_models()
        except Exception as e:
            self.model_load_error = f"{type(e).__name__}: {e}"
            return False

    def _load_models(self) -> bool:
        lightgbm_path = CHECKPOINTS_DIR / "lightgbm.txt"
        mlp_path = CHECKPOINTS_DIR / "mlp.pt"
        scaler_path = CHECKPOINTS_DIR / "scaler.pkl"
        if not (lightgbm_path.exists() and mlp_path.exists() and scaler_path.exists()):
            self.model_load_error = "缺少 lightgbm.txt、mlp.pt 或 scaler.pkl。"
            return False

        self.lgbm_model = lgb.Booster(model_file=str(lightgbm_path))

        mlp_config = load_config("mlp")
        self.mlp_model = MalwareMLP(
            hidden_dims=mlp_config["hidden_dims"],
            dropout=mlp_config["dropout"],
            embed_dim=mlp_config["embed_dim"],
        ).to(self.device)
        self.mlp_model.load_state_dict(
            torch.load(mlp_path, map_location=self.device, weights_only=True)
        )
        self.mlp_model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.model_provenance_verified, self.model_provenance_warning = (
            verify_evaluated_artifacts(
                EVALUATION_MANIFEST_PATH,
                {
                    lightgbm_path.name: lightgbm_path,
                    mlp_path.name: mlp_path,
                    scaler_path.name: scaler_path,
                },
            )
        )
        return True

    def predict(self, filename: str, content: bytes) -> DetectionResult:
        if self.models_loaded:
            return self._real_predict(filename, content)
        if self.stub_enabled:
            return self._stub_predict(filename, content)
        raise ModelUnavailableError(self.model_load_error or "检测模型 checkpoint 不完整。")

    def extract_feature_vector(self, content: bytes) -> np.ndarray:
        """Extract one EMBER vector and normalize third-party parser failures."""
        try:
            return extract_features(content).reshape(-1)
        except Exception as error:
            raise FeatureExtractionError(
                f"无法解析该文件的 PE 结构（{type(error).__name__}: {error}），可能不是有效的 PE 文件，"
                "或存在第三方解析库尚未处理的特殊结构（例如某些证书链格式）。"
            ) from error

    def predict_features_ml_only(
        self,
        filenames: list[str],
        feature_rows: list[np.ndarray] | np.ndarray,
    ) -> list[DetectionResult]:
        """Vectorized LightGBM, MLP and optional family inference for a valid feature batch."""
        if not self.models_loaded:
            raise ModelUnavailableError(
                self.model_load_error or "检测模型 checkpoint 不完整。"
            )
        if not filenames:
            return []
        features = np.asarray(feature_rows, dtype=np.float32)
        if features.ndim != 2 or len(features) != len(filenames):
            raise ValueError("filenames and feature_rows must form an aligned 2D batch")

        with self._inference_slots:
            lgbm_scores = _validate_probability_vector(
                self.lgbm_model.predict(features), len(filenames), "LightGBM"
            )
            features_scaled = self.scaler.transform(features).astype(np.float32, copy=False)
            with torch.inference_mode():
                tensor = torch.from_numpy(features_scaled).to(self.device)
                mlp_logits = self.mlp_model(tensor)
                mlp_scores = _validate_probability_vector(
                    torch.sigmoid(mlp_logits).float().cpu().numpy().reshape(-1),
                    len(filenames),
                    "MLP",
                )

                ensemble_scores = (lgbm_scores + mlp_scores) / 2
                malicious_mask = ensemble_scores >= 0.5
                family_names: list[str | None] = [None] * len(filenames)
                family_confidences: list[float | None] = [None] * len(filenames)
                malicious_indices = np.flatnonzero(malicious_mask)
                if self.family_model_loaded and len(malicious_indices):
                    index_tensor = torch.from_numpy(malicious_indices).to(self.device)
                    family_logits = self.family_model(tensor.index_select(0, index_tensor))
                    family_probabilities = (
                        torch.softmax(family_logits, dim=1).float().cpu().numpy()
                    )
                    for row_index, probabilities in zip(
                        malicious_indices, family_probabilities
                    ):
                        family_names[row_index], family_confidences[row_index] = (
                            self._decode_family_probability(probabilities)
                        )

        results = []
        for index, filename in enumerate(filenames):
            p_lgbm = float(lgbm_scores[index])
            p_mlp = float(mlp_scores[index])
            is_malicious = bool(malicious_mask[index])
            final_probability = float(ensemble_scores[index])
            confidence = final_probability if is_malicious else 1 - final_probability
            agreement = (
                "agree" if (p_lgbm >= 0.5) == (p_mlp >= 0.5) else "disagree"
            )
            results.append(
                DetectionResult(
                    filename=filename,
                    verdict="malicious" if is_malicious else "benign",
                    confidence=round(confidence, 4),
                    family=family_names[index],
                    familyConfidence=family_confidences[index],
                    gradcamUrl=None,
                    attck=[],
                    llmReport="",
                    modelAgreement=agreement,
                    lgbmScore=round(p_lgbm, 4),
                    mlpScore=round(p_mlp, 4),
                    llmVerdict=None,
                    llmConfidence=None,
                )
            )
        return results

    def predict_ml_only(self, filename: str, content: bytes) -> DetectionResult:
        """Single-item compatibility wrapper around vectorized ML-only inference."""
        if not self.models_loaded:
            if self.stub_enabled:
                return self._stub_predict(filename, content)
            raise ModelUnavailableError(
                self.model_load_error or "检测模型 checkpoint 不完整。"
            )
        features = self.extract_feature_vector(content)
        return self.predict_features_ml_only([filename], [features])[0]

    def _score(
        self, content: bytes
    ) -> tuple[float, float, bool, str | None, float | None, list[FeatureAttention]]:
        """Feature extraction + both ML models + optional family + MLP fusion weights. No LLM.

        The interactive single-file path returns fusion weights for display. Batch detection uses
        ``predict_features_ml_only`` to vectorize scoring and intentionally skips this detail.
        Returns (p_lgbm, p_mlp, is_malicious, family, family_confidence, feature_attention).
        """
        features = self.extract_feature_vector(content).reshape(1, -1)

        with self._inference_slots:
            p_lgbm = float(
                _validate_probability_vector(
                    self.lgbm_model.predict(features), 1, "LightGBM"
                )[0]
            )

            features_scaled = self.scaler.transform(features)
            with torch.inference_mode():
                tensor = torch.from_numpy(
                    features_scaled.astype(np.float32, copy=False)
                ).to(self.device)
                logit, attn = self.mlp_model(tensor, return_attn=True)
                p_mlp = float(
                    _validate_probability_vector(
                        torch.sigmoid(logit).float().cpu().numpy().reshape(-1),
                        1,
                        "MLP",
                    )[0]
                )
                attn_weights = attn[0].cpu().numpy()

            is_malicious = (p_lgbm + p_mlp) / 2 >= 0.5
            family, family_confidence = (
                self._predict_family(features_scaled)
                if is_malicious and self.family_model_loaded
                else (None, None)
            )

        attention = [
            FeatureAttention(
                group=name,
                label=FEATURE_GROUP_LABELS[name],
                weight=round(float(w), 4),
            )
            for (name, _), w in zip(SEGMENT_DIMS, attn_weights)
        ]
        return p_lgbm, p_mlp, is_malicious, family, family_confidence, attention

    def _real_predict(self, filename: str, content: bytes) -> DetectionResult:
        p_lgbm, p_mlp, is_malicious, family, family_confidence, attention = self._score(content)

        final_prob = (p_lgbm + p_mlp) / 2
        confidence = final_prob if is_malicious else 1 - final_prob
        agreement = "agree" if (p_lgbm >= 0.5) == (p_mlp >= 0.5) else "disagree"
        verdict = "malicious" if is_malicious else "benign"

        # The LLM is called for every sample (not just flagged ones) so the result card can
        # always show a genuine three-way comparison — its verdict is independent (the prompt
        # never sees the ML models' output) and shown for comparison only, never averaged into
        # final_prob/confidence above. This trades the earlier "not on the hot path" design
        # intent (see CLAUDE.local.md) for comparison completeness on every interactive
        # single-file upload; a true bulk/batch endpoint should still skip this.
        summary = summarize(content)
        analysis = generate_report(content, summary)
        llm_report = analysis.narrative
        llm_verdict = analysis.verdict
        llm_confidence = analysis.confidence
        attck_tags = (
            [AttckTag(tactic=t.tactic, technique=t.technique) for t in summary.attck_tags]
            if is_malicious
            else []
        )

        return DetectionResult(
            filename=filename,
            verdict=verdict,
            confidence=round(confidence, 4),
            family=family,
            familyConfidence=family_confidence,
            gradcamUrl=None,
            attck=attck_tags,
            llmReport=llm_report,
            modelAgreement=agreement,
            lgbmScore=round(p_lgbm, 4),
            mlpScore=round(p_mlp, 4),
            llmVerdict=llm_verdict,
            llmConfidence=round(llm_confidence, 4) if llm_confidence is not None else None,
            featureAttention=attention,
        )

    def _stub_predict(self, filename: str, content: bytes) -> DetectionResult:
        digest = hashlib.sha256(content).hexdigest()
        is_malicious = int(digest[:2], 16) % 2 == 0
        confidence = 0.90 + (int(digest[2:4], 16) / 255) * 0.09
        lgbm_score = confidence if is_malicious else 1 - confidence
        mlp_score = lgbm_score

        return DetectionResult(
            filename=filename,
            verdict="malicious" if is_malicious else "benign",
            confidence=round(confidence, 4),
            family="Emotet" if is_malicious else None,
            gradcamUrl=None,
            attck=(
                [
                    AttckTag(tactic="Persistence", technique="T1547 Boot or Logon Autostart"),
                    AttckTag(tactic="Privilege Escalation", technique="T1543 Create or Modify System Process"),
                ]
                if is_malicious
                else []
            ),
            llmReport=(
                "[占位输出] 尚未接入真实特征提取与训练模型，这是基于文件哈希的确定性伪造结果，"
                "仅用于验证前后端接口契约，不代表任何真实检测能力。"
            ),
            modelAgreement="agree",
            lgbmScore=round(lgbm_score, 4),
            mlpScore=round(mlp_score, 4),
            llmVerdict="malicious" if is_malicious else "benign",
            llmConfidence=round(confidence, 4),
        )


predictor = Predictor()
