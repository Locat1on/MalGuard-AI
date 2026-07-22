# 后端新增接口契约：批量检测 + 检测历史

面向前端对接。后端已实现并自测通过；前端页面按本契约接入即可。所有路径都在 `/api` 前缀下。

## 1. 单文件检测（已有，仅新增一个字段）

`POST /api/detect` 的响应 `DetectionResult` 新增一个可选字段：

```ts
historyId: number | null;  // 本次检测在历史库中的记录 id；stub 模式（模型未加载）时为 null
```

拿到 `historyId` 后即可直接跳转 `GET /api/history/{historyId}/report` 导出报告。

另新增 `featureAttention` 字段——**模型内部的特征组融合权重**（`gradcamUrl` 仍为 `null`）：

```ts
interface FeatureAttention {
  group: string;   // EMBER 特征组原名，如 "imports"
  label: string;   // 已配好的中文标签，如 "导入表 (API)"，可直接展示
  weight: number;  // 该组的注意力权重，0–1，12 组之和为 1
}
// DetectionResult 新增：
featureAttention: FeatureAttention[] | null;  // 单文件检测时有 12 项；批量为 null
```

MLP 在融合 12 个特征组时对每组算了一个 softmax 注意力权重，`featureAttention` 就是这组权重——可用于观察模型在本次融合时更重视哪些特征组，但它不是因果归因，不能表述成“某特征导致了该判定”。建议前端按 `weight` 排序画条形图（取 Top-N 即可），标题使用“特征组融合权重”。

再新增 `familyConfidence` 字段，与已有的 `family` 配套：

```ts
family: string | null;            // 已有：恶意家族名，良性/未知时为 null
familyConfidence: number | null;  // 新增：该家族的 softmax 置信度 0–1，family 为 null 时也为 null
```

**重要（展示规范）**：家族分类是"最像哪个已知家族"的概率性推测，**不是取证认定**。请**务必带上置信度展示**，例如「疑似 Wacatac（62%）」，不要显示成光秃秃的「Wacatac」。后端已做低置信兜底：置信度低于阈值（`configs/family.yaml` 的 `family_confidence_floor`）或落入"其他"类时，`family` 直接返回 `null`（即"未知家族"），此时前端不必展示家族。

## 2. 批量检测（新增）

`POST /api/detect/batch` —— multipart，字段名 `files`，可传多个文件（上限 100 个，单文件 100MB）。

**只跑两个 ML 模型 + 可选家族分类，不调 LLM、不出 ATT&CK**（批量是 hot path，刻意跳过昂贵的分析层）。每个成功项也会写入历史。

单个文件解析失败（如非 PE 文件）不会让整个请求失败，而是作为 `ok:false` 项返回。

响应 `BatchDetectionResult`：

```ts
interface BatchItem {
  filename: string;
  ok: boolean;                                    // false 表示该文件解析失败
  verdict: "malicious" | "benign" | null;
  confidence: number | null;
  family: string | null;
  familyConfidence: number | null;
  lgbmScore: number | null;
  mlpScore: number | null;
  modelAgreement: "agree" | "disagree" | null;
  historyId: number | null;
  error: string | null;                           // ok:false 时为失败原因
}

interface BatchDetectionResult {
  items: BatchItem[];
  total: number;
  malicious: number;
  benign: number;
  failed: number;
}
```

前端示例（FormData 多文件）：

```ts
const form = new FormData();
for (const f of fileList) form.append("files", f);
const res = await fetch("/api/detect/batch", { method: "POST", body: form });
```

## 3. 检测历史（新增，后端 SQLite 持久化）

历史存于后端 `data/history.db`，重启、换浏览器都在。**注意：现有前端 `HistoryEntry.id` 是 `string`，后端返回的是 `number`——请把类型改成 `number`。** 后端历史记录字段比现有 `HistoryEntry` 多，前端按需取用即可。

`GET /api/history?limit=50&offset=0` → `HistoryRecord[]`（按时间倒序，`limit` 1–500，默认 50）

`GET /api/history/{id}` → 单条 `HistoryRecord`，不存在返回 404

```ts
interface HistoryRecord {
  id: number;
  createdAt: string;          // ISO8601 UTC，如 "2026-07-10T08:30:00+00:00"
  filename: string;
  sha256: string;
  source: "single" | "batch";
  verdict: "malicious" | "benign";
  confidence: number;
  family: string | null;
  familyConfidence: number | null;
  lgbmScore: number;
  mlpScore: number;
  modelAgreement: "agree" | "disagree";
  llmVerdict: "malicious" | "benign" | null;   // 批量记录为 null
  llmConfidence: number | null;
  llmReport: string;                            // 批量记录为空串
  attck: { tactic: string; technique: string }[];
}
```

## 4. 报告导出（新增）

`GET /api/history/{id}/report` → 自包含 HTML（`Content-Type: text/html`，无外部依赖）。

直接在新标签页打开即可查看，浏览器「打印 → 另存为 PDF」即得 PDF 报告。前端只需给一个链接/按钮指向该 URL，无需自己渲染。

## 5. 删除历史（新增）

`DELETE /api/history/{id}` → `{ "deleted": true }`（不存在返回 404）

`DELETE /api/history` → `{ "deleted": <删除条数> }`（清空全部）
## 6. 服务状态（新增）

`GET /api/health` 始终用于存活检查，并返回组件状态：

```ts
interface HealthStatus {
  ok: boolean;
  ready: boolean;              // 核心 LightGBM + MLP 是否可用于真实检测
  mode: "real" | "stub" | "unavailable";
  modelsLoaded: boolean;
  familyModelLoaded: boolean;  // 可选组件，不影响 ready
  llmConfigured: boolean;      // 可选组件，不影响 ready
  modelLoadError: string | null;
  familyModelLoadError: string | null;
}
```

`GET /api/ready` 返回同一结构；核心模型可用时为 200，否则为 503。checkpoint 缺失或架构不兼容时，检测接口也返回 503，不再默认返回伪造结果。只有显式设置 `ALLOW_STUB_PREDICTIONS=1` 才启用联调用 stub。

## 7. 历史统计（新增）

`GET /api/history/stats` →：

```ts
interface HistoryStats {
  total: number;
  malicious: number;
  benign: number;
  single: number;
  batch: number;
  modelDisagreements: number;
  llmCompared: number;
  llmDisagreements: number;
  lastCreatedAt: string | null;
}
```

这些数据来自 SQLite 聚合，可用于历史页顶部的紧凑统计区。`llmDisagreements / llmCompared` 才是有意义的 LLM 分歧率，批量检测未运行 LLM，不应进入分母。
