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

BENIGN_REPORT = "未检测到可疑行为特征，未触发 LLM 深度分析（仅对判定为恶意的样本生成行为分析报告）。"


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
        features = extract_features(content).reshape(1, -1)

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

        # LLM analysis only runs on flagged samples — never on the hot path for bulk/benign
        # detection (see CLAUDE.local.md "Design intent").
        if is_malicious:
            summary = summarize(content)
            llm_report = generate_report(content, summary, verdict, confidence)
            attck_tags = [AttckTag(tactic=t.tactic, technique=t.technique) for t in summary.attck_tags]
        else:
            llm_report = BENIGN_REPORT
            attck_tags = []

        return DetectionResult(
            filename=filename,
            verdict=verdict,
            confidence=round(confidence, 4),
            family=None,
            gradcamUrl=None,
            attck=attck_tags,
            llmReport=llm_report,
            modelAgreement=agreement,
        )

    def _stub_predict(self, filename: str, content: bytes) -> DetectionResult:
        digest = hashlib.sha256(content).hexdigest()
        is_malicious = int(digest[:2], 16) % 2 == 0
        confidence = 0.90 + (int(digest[2:4], 16) / 255) * 0.09

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
        )


predictor = Predictor()
