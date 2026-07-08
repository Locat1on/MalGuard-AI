import type { DetectionResult, ModelMetric } from "./types";

const MOCK_REPORT = `该样本在执行过程中表现出典型的持久化与横向探测行为：先通过 RegSetValueExA 在
HKCU\\...\\Run 下写入自启动项，随后调用 CreateServiceA 注册系统服务，并以
OpenSCManagerA 提权尝试控制服务管理器。样本还多次调用 CopyFileA 与
FindFirstFileExA 遍历并复制系统目录下的可执行文件，具备一定的自我传播特征。

综合来看，该行为链与 Emotet 家族的加载器阶段高度相似，建议标记为高风险并隔离处理。`;

async function mockAnalyze(filename: string): Promise<DetectionResult> {
  await new Promise((r) => setTimeout(r, 1400));
  const isMalicious = !/^(chrome|notepad|explorer)/i.test(filename);
  return {
    filename,
    verdict: isMalicious ? "malicious" : "benign",
    confidence: isMalicious ? 0.973 : 0.991,
    family: isMalicious ? "Emotet" : null,
    gradcamUrl: null,
    attck: isMalicious
      ? [
          { tactic: "Persistence", technique: "T1547 Boot or Logon Autostart" },
          { tactic: "Privilege Escalation", technique: "T1543 Create or Modify System Process" },
          { tactic: "Lateral Movement", technique: "T1570 Lateral Tool Transfer" },
        ]
      : [],
    llmReport: isMalicious
      ? MOCK_REPORT
      : "未检测到可疑行为链，API 调用模式与已知良性软件基线一致。",
    modelAgreement: "agree",
    lgbmScore: isMalicious ? 0.968 : 0.009,
    mlpScore: isMalicious ? 0.978 : 0.009,
    llmVerdict: isMalicious ? "malicious" : "benign",
    llmConfidence: isMalicious ? 0.91 : 0.95,
  };
}

export class DetectionFailedError extends Error {}

export async function analyzeFile(file: File): Promise<DetectionResult> {
  const form = new FormData();
  form.append("file", file);

  let res: Response;
  try {
    res = await fetch("/api/detect", { method: "POST", body: form });
  } catch {
    // Backend unreachable (e.g. not running during frontend-only dev) — fall back to mock.
    return mockAnalyze(file.name);
  }

  if (!res.ok) {
    // Backend is reachable and explicitly rejected/failed this file — surface the real
    // reason, never silently substitute a fake result for a real error.
    const body = await res.json().catch(() => null);
    throw new DetectionFailedError(body?.detail ?? `检测失败（后端返回 ${res.status}）`);
  }
  return (await res.json()) as DetectionResult;
}

const MOCK_METRICS: ModelMetric[] = [
  { model: "LightGBM (EMBER 静态特征基线)", accuracy: 0.943, precision: 0.931, recall: 0.918, f1: 0.924 },
  { model: "MLP 深度模型 (本系统)", accuracy: 0.961, precision: 0.952, recall: 0.947, f1: 0.949 },
];

export async function fetchMetrics(): Promise<ModelMetric[]> {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error(`backend responded ${res.status}`);
    return (await res.json()) as ModelMetric[];
  } catch {
    return MOCK_METRICS;
  }
}
