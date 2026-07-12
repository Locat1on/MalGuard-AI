"""Detection predictor.

Real path: extract EMBER-schema features from the uploaded PE file (src/features/extract.py),
run the trained LightGBM + MLP models (checkpoints/), and report agreement between the two.

Falls back to a clearly-labeled stub if the checkpoints don't exist yet, so the API contract
can still be exercised end-to-end from the frontend without a trained model.
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import torch

from app.schemas import AttckTag, DetectionResult, FeatureAttention
from src.config import load_config
from src.features.extract import SEGMENT_DIMS, extract_features
from src.llm.feature_summary import summarize
from src.llm.report import generate_report
from src.models.mlp import MalwareMLP

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Human-readable Chinese labels for the 12 EMBER feature groups, so the attention-weight
# explanation is display-ready without the frontend hardcoding this mapping.
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


class FeatureExtractionError(Exception):
    """Raised when a file's PE structure could not be parsed into a feature vector.

    Some legitimately-signed files (and some malformed ones) trigger bugs in third-party
    parsing libraries (e.g. thrember's Authenticode certificate-chain parsing crashes with
    TypeError on certain certificate stores). Rather than let that surface as a raw 500,
    the router turns this into an honest 422 with a clear message.
    """


class Predictor:
    def __init__(self) -> None:
        # Set before either _try_load_* call: the family model needs a device even if the
        # binary models fail to load (independent optional checkpoints).
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = self._try_load_models()
        self.family_model_loaded = self._try_load_family_model()

    def _try_load_family_model(self) -> bool:
        """Optional: the family classifier is a separate, later-added model (see
        src/models/train_family.py). Detection itself must keep working even if it hasn't been
        trained yet — `predict()` just reports family=None in that case.

        Reuses the same MalwareMLP architecture and checkpoints/scaler.pkl as the binary
        detector (see train_family.py for why the scaler is shared rather than refit), just with
        its output layer widened to the family class count.
        """
        family_model_path = CHECKPOINTS_DIR / "family_mlp.pt"
        family_labels_path = CHECKPOINTS_DIR / "family_labels.json"
        if not (family_model_path.exists() and family_labels_path.exists()):
            return False

        with open(family_labels_path, encoding="utf-8") as f:
            self.family_labels: list[str] = json.load(f)

        family_config = load_config("family")
        self.family_confidence_floor: float = family_config["family_confidence_floor"]
        self.family_model = MalwareMLP(
            hidden_dims=family_config["hidden_dims"],
            dropout=family_config["dropout"],
            embed_dim=family_config["embed_dim"],
            num_classes=len(self.family_labels),
        ).to(self.device)
        self.family_model.load_state_dict(torch.load(family_model_path, map_location=self.device))
        self.family_model.eval()
        return True

    def _predict_family(self, features_scaled: np.ndarray) -> tuple[str | None, float | None]:
        """Return (family_name, confidence) or (None, None).

        The classifier always argmaxes to *some* class, but for a file that doesn't resemble any
        family it saw in training (e.g. the synthetic demo sample, which is out-of-distribution)
        that pick is not trustworthy. So we return None when either (a) the pick is the "其他"
        catch-all, or (b) its softmax probability is below family_confidence_floor. When a name is
        returned its confidence goes with it, so the UI can show "suspected X (62%)" rather than a
        bald claim.
        """
        with torch.no_grad():
            logits = self.family_model(torch.tensor(features_scaled, dtype=torch.float32).to(self.device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(proba.argmax())
        confidence = float(proba[idx])
        name = self.family_labels[idx]
        # "其他" is always the last label — treat catch-all or low-confidence as unknown.
        if name == self.family_labels[-1] or confidence < self.family_confidence_floor:
            return None, None
        return name, round(confidence, 4)

    def _try_load_models(self) -> bool:
        lightgbm_path = CHECKPOINTS_DIR / "lightgbm.txt"
        mlp_path = CHECKPOINTS_DIR / "mlp.pt"
        scaler_path = CHECKPOINTS_DIR / "scaler.pkl"
        if not (lightgbm_path.exists() and mlp_path.exists() and scaler_path.exists()):
            return False

        self.lgbm_model = lgb.Booster(model_file=str(lightgbm_path))

        mlp_config = load_config("mlp")
        self.mlp_model = MalwareMLP(
            hidden_dims=mlp_config["hidden_dims"],
            dropout=mlp_config["dropout"],
            embed_dim=mlp_config["embed_dim"],
        ).to(self.device)
        self.mlp_model.load_state_dict(torch.load(mlp_path, map_location=self.device))
        self.mlp_model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        return True

    def predict(self, filename: str, content: bytes) -> DetectionResult:
        if self.models_loaded:
            return self._real_predict(filename, content)
        return self._stub_predict(filename, content)

    def predict_ml_only(self, filename: str, content: bytes) -> DetectionResult:
        """Batch path: the two ML models (+ optional family) only, no LLM/ATT&CK.

        Batch scanning is the "hot path" the design intent keeps off the LLM (latency + API
        cost) — see CLAUDE.local.md. `llmReport`/`llmVerdict`/`llmConfidence` come back empty
        and `attck` empty, so the response schema stays the same as single-file detection while
        skipping the expensive analysis layer.
        """
        if not self.models_loaded:
            return self._stub_predict(filename, content)

        # batch skips the attention detail
        p_lgbm, p_mlp, is_malicious, family, family_confidence, _ = self._score(content)
        final_prob = (p_lgbm + p_mlp) / 2
        confidence = final_prob if is_malicious else 1 - final_prob
        agreement = "agree" if (p_lgbm >= 0.5) == (p_mlp >= 0.5) else "disagree"
        return DetectionResult(
            filename=filename,
            verdict="malicious" if is_malicious else "benign",
            confidence=round(confidence, 4),
            family=family,
            familyConfidence=family_confidence,
            gradcamUrl=None,
            attck=[],
            llmReport="",
            modelAgreement=agreement,
            lgbmScore=round(p_lgbm, 4),
            mlpScore=round(p_mlp, 4),
            llmVerdict=None,
            llmConfidence=None,
        )

    def _score(
        self, content: bytes
    ) -> tuple[float, float, bool, str | None, float | None, list[FeatureAttention]]:
        """Feature extraction + both ML models + optional family + MLP attention. No LLM.

        Shared by single-file (`_real_predict`) and batch (`predict_ml_only`) so the two paths
        can never diverge on how the verdict itself is computed. Returns
        (p_lgbm, p_mlp, is_malicious, family, family_confidence, feature_attention). The attention
        list is cheap (already computed inside the same forward pass); the batch path ignores it.
        """
        try:
            features = extract_features(content).reshape(1, -1)
        except Exception as e:
            raise FeatureExtractionError(
                f"无法解析该文件的 PE 结构（{type(e).__name__}: {e}），可能不是有效的 PE 文件，"
                "或存在第三方解析库尚未处理的特殊结构（例如某些证书链格式）。"
            ) from e

        p_lgbm = float(self.lgbm_model.predict(features)[0])

        features_scaled = self.scaler.transform(features)
        with torch.no_grad():
            logit, attn = self.mlp_model(
                torch.tensor(features_scaled, dtype=torch.float32).to(self.device), return_attn=True
            )
            p_mlp = float(torch.sigmoid(logit).cpu().item())
            attn_weights = attn[0].cpu().numpy()

        attention = [
            FeatureAttention(group=name, label=FEATURE_GROUP_LABELS[name], weight=round(float(w), 4))
            for (name, _), w in zip(SEGMENT_DIMS, attn_weights)
        ]

        is_malicious = (p_lgbm + p_mlp) / 2 >= 0.5
        family, family_confidence = (
            self._predict_family(features_scaled)
            if is_malicious and self.family_model_loaded
            else (None, None)
        )
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
