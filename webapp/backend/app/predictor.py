"""Detection predictor.

Real path: extract EMBER-schema features from the uploaded PE file (src/features/extract.py),
run the trained LightGBM + MLP models (checkpoints/), and report agreement between the two.

Falls back to a clearly-labeled stub if the checkpoints don't exist yet, so the API contract
can still be exercised end-to-end from the frontend without a trained model.
"""

import hashlib
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import torch

from app.schemas import AttckTag, DetectionResult
from src.config import load_config
from src.features.extract import extract_features
from src.llm.feature_summary import summarize
from src.llm.report import generate_report
from src.models.mlp import MalwareMLP

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

class FeatureExtractionError(Exception):
    """Raised when a file's PE structure could not be parsed into a feature vector.

    Some legitimately-signed files (and some malformed ones) trigger bugs in third-party
    parsing libraries (e.g. thrember's Authenticode certificate-chain parsing crashes with
    TypeError on certain certificate stores). Rather than let that surface as a raw 500,
    the router turns this into an honest 422 with a clear message.
    """


class Predictor:
    def __init__(self) -> None:
        self.models_loaded = self._try_load_models()

    def _try_load_models(self) -> bool:
        lightgbm_path = CHECKPOINTS_DIR / "lightgbm.txt"
        mlp_path = CHECKPOINTS_DIR / "mlp.pt"
        scaler_path = CHECKPOINTS_DIR / "scaler.pkl"
        if not (lightgbm_path.exists() and mlp_path.exists() and scaler_path.exists()):
            return False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def _real_predict(self, filename: str, content: bytes) -> DetectionResult:
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
            logit = self.mlp_model(torch.tensor(features_scaled, dtype=torch.float32).to(self.device))
            p_mlp = float(torch.sigmoid(logit).cpu().item())

        final_prob = (p_lgbm + p_mlp) / 2
        is_malicious = final_prob >= 0.5
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
            family=None,
            gradcamUrl=None,
            attck=attck_tags,
            llmReport=llm_report,
            modelAgreement=agreement,
            lgbmScore=round(p_lgbm, 4),
            mlpScore=round(p_mlp, 4),
            llmVerdict=llm_verdict,
            llmConfidence=round(llm_confidence, 4) if llm_confidence is not None else None,
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
