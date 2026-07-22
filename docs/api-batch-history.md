# 后端新增接口契约：批量检测 + 检测历史

面向前端对接。后端已实现并自测通过；前端页面按本契约接入即可。所有路径都在 `/api` 前缀下。

## 1. 单文件检测（已有，仅新增一个字段）

`POST /api/detect` 的响应 `DetectionResult` 新增一个可选字段：

```ts
historyId: number | null;  // 历史记录 id；stub 模式或本次历史写入失败时为 null
```

拿到 `historyId` 后即可直接跳转 `GET /api/history/{historyId}/report` 导出报告。模型推理成功后，即使 SQLite 暂时不可写，接口仍保留有效检测结果并返回 `historyId: null`；此时前端应展示结果，但不要生成本地假 ID 或报告链接。

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

单文件中的静态摘要与 LLM 均是可选分析层。摘要提取抛错、LLM 调用异常，或 LLM 返回非法判定、非有限/越界置信度、空说明时，接口仍返回 200 和真实的 LightGBM + MLP 结论，同时令 `llmVerdict` / `llmConfidence` 为 `null`，并在 `llmReport` 中写明降级原因。摘要失败时 `attck` 为空；仅 LLM 失败但摘要成功时，确定性 ATT&CK 标签继续保留。

## 2. 批量检测（新增）

`POST /api/detect/batch` —— multipart，字段名 `files`，可传多个文件（上限 100 个、单文件 100MiB、原始文件合计 500MiB）。声明的 multipart 请求体超过 512MiB 时，后端会在解析文件前直接返回 413。

**只跑两个 ML 模型 + 可选家族分类，不调 LLM、不出 ATT&CK**（批量是 hot path，刻意跳过昂贵的分析层）。每个成功项会尝试写入历史；若整批历史写入失败，检测项仍为 `ok:true`，但其 `historyId` 为 `null`。

后端逐个文件完成 PE 特征提取后立即释放原始字节，只保留约 10KB 的 2568 维向量；所有有效向量随后合并为一次 LightGBM、MLP 和可选家族模型前向。接口仍在全部文件结束后一次性响应，不提供逐文件流式进度。

解析前限制依赖正常客户端发送的 `Content-Length`。使用 chunked transfer 时，路由仍会按单文件 100MiB 和原始文件合计 500MiB 拒绝处理，但 multipart 临时文件可能已经写入磁盘；对外部署必须在 Nginx、Caddy 等反向代理同步设置请求体上限。

单个文件解析失败（如非 PE 文件）不会让整个请求失败，而是作为 `ok:false` 项返回。

后端不依据扩展名判断：文件必须同时具有 DOS `MZ` 头、边界内的 `e_lfanew` 和 `PE\0\0` NT 签名，否则单文件接口返回 422，批量接口返回对应的 `ok:false` 项；第三方特征提取器不会再接收任意文本字节。

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

历史默认存于后端 `data/history.db`，可通过 `MALGUARD_HISTORY_DB` 指向其他绝对路径或仓库根目录下的相对路径。重启、换浏览器后记录仍然存在。**注意：现有前端 `HistoryEntry.id` 是 `string`，后端返回的是 `number`——请把类型改成 `number`。** 后端历史记录字段比现有 `HistoryEntry` 多，前端按需取用即可。

`GET /api/history?limit=50&offset=0` → `HistoryRecord[]`（按时间倒序，`limit` 1–500，默认 50）。响应头 `X-Total-Count` 返回不受分页参数影响的历史总数；即使当前页为空也会返回，可用于计算总页数。

`GET /api/history/{id}` → 单条 `HistoryRecord`，不存在返回 404

`GET /api/history/backup` → 事务一致的独立 SQLite 快照（`application/vnd.sqlite3`、附件下载）。服务端在响应完成后删除临时快照；接口不会暂停检测写入。启用 API Key 时必须携带 `X-API-Key`。

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

未启用 API Key 时可直接在新标签页打开，浏览器「打印 → 另存为 PDF」即得 PDF 报告。启用鉴权后普通链接无法附加请求头，前端必须用带密钥的 `fetch` 获取 HTML 并通过临时 Blob URL 打开。

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
  modelProvenanceVerified: boolean | null; // 当前三项核心 artifact 是否匹配正式评估哈希
  modelProvenanceWarning: string | null;   // 清单缺失/无效或模型漂移时的说明
  inferenceConcurrency: number;             // 共享模型允许的并发推理数，默认 1
  detectionConcurrency: number;             // 每进程完整检测请求并发上限，默认 2
  apiKeyRequired: boolean;                   // 是否要求受保护接口携带 X-API-Key
}
```

`GET /api/ready` 返回同一结构；核心模型可用时为 200，否则为 503。checkpoint 缺失、架构不兼容，或 LightGBM/scaler 的特征维度不是 EMBER 所需的 2568 时，启动加载阶段即令 `ready=false` 并通过 `modelLoadError` 说明原因；LightGBM/MLP 运行时返回非有限值、越界概率及错误形状时，检测接口也返回 503，不再默认返回伪造结果或形成不可信结论。只有显式设置 `ALLOW_STUB_PREDICTIONS=1` 才启用联调用 stub。

家族分类是可选组件：加载后若前向抛出异常，或返回非有限值、错误形状及概率和不为 1 的分布，后端会自动将 `familyModelLoaded` 置为 `false`，在 `familyModelLoadError` 中记录原因，并让本次及后续结果的 `family` / `familyConfidence` 返回 `null`；核心二分类结果与 `/api/ready` 不受影响。

`modelProvenanceVerified=false` 不会阻断检测，但表示当前加载的 `lightgbm.txt`、`mlp.pt` 或 `scaler.pkl` 与完整的 `evaluation_manifest.json` 不一致，此时指标页不能把现有正式分数视为当前部署模型的成绩；应重新运行 `src/eval/compare_models.py`。值为 `null` 表示清单缺失、无法读取、字段不完整或协议/计数自相矛盾。健康状态与两个 metrics 接口复用同一套清单校验。

`inferenceConcurrency` 只描述共享模型前向并发上限；单 GPU 默认值 1 用于降低显存争用风险。`detectionConcurrency` 则覆盖 multipart 解析、特征提取、模型推理、单文件 LLM 和历史写入的完整请求生命周期，默认每进程 2 个。

超过完整请求上限时，`POST /api/detect` 和 `POST /api/detect/batch` 在读取 multipart 前返回 429：

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 1
Content-Type: application/json

{"detail":"检测服务繁忙，请稍后重试。"}
```

后端不维护无限等待队列。该限制是进程内的；启动多个 Uvicorn worker 时，实例总容量约为 `detectionConcurrency × worker 数`，不同主机之间也不共享计数。

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

## 8. 指标来源与请求追踪（新增）

`GET /api/metrics` 从 `evaluation_manifest.json.results` 返回三行正式评估结果：LightGBM、MLP、实际部署使用的二者算术平均集成。缺少来源清单时返回 404；JSON 损坏、协议/运行环境/Git 来源缺失、三项核心 artifact 不完整、指标越界，或混淆矩阵与测试集类别计数不一致时返回 503。接口不会用全零占位数据或部分清单伪装成正式指标。`metrics.json` 仍作为评估导出文件保留，但不再是后端的独立数据源。

`GET /api/metrics/provenance` 返回 `checkpoints/evaluation_manifest.json`，包含：
- 官方测试集名称、样本数、类别数、阈值、集成规则和推理批大小；
- 每个模型的指标与混淆矩阵；
- 模型及 scaler 的 SHA-256、文件大小；
- Python/核心依赖/CUDA 环境和评估开始时的 Git 状态。

清单尚未生成时接口返回 404。前端可把该接口用于“指标来源”抽屉，不应把运行环境和哈希塞进主指标表。

所有 API 响应还会带两个响应头：

```text
X-Request-ID: 32 位十六进制请求标识
X-Process-Time-Ms: 后端处理耗时（毫秒）
```

所有正常响应、FastAPI 业务错误以及未捕获的 500 都带上述响应头。未捕获异常统一返回 `{"detail":"服务器内部错误。"}`，真实异常只写入带同一请求编号的后端日志；允许的跨域来源也可读取这些头。前端错误提示可附带 `X-Request-ID` 方便定位，但不应展示后端堆栈或上传内容。

## 9. API Key 访问保护

未设置 `MALGUARD_API_KEY` 时，接口行为与本地开发模式一致。设置至少 16 字符的 ASCII 密钥后：

- `/api/detect`、`/api/detect/batch` 和全部 `/api/history*` 接口要求请求头 `X-API-Key: <密钥>`；
- `/api/health`、`/api/ready`、`/api/metrics` 和 `/api/metrics/provenance` 仍可匿名访问；
- 缺失或错误密钥返回 401、`WWW-Authenticate: ApiKey` 和 JSON `detail`；
- CORS `OPTIONS` 预检不要求密钥，`X-API-Key` 可作为跨域请求头；
- 响应会向跨域前端暴露 `X-Request-ID`、`X-Process-Time-Ms`、`X-Total-Count`、`Retry-After`、`Content-Disposition` 与 `WWW-Authenticate`；
- 启用密钥时 `/docs` 会显示 `ApiKeyAuth` 授权入口，并只把检测与历史操作标记为受保护。

`GET /api/health` 的 `apiKeyRequired` 用于告诉前端是否需要凭据，不包含密钥本身。报告接口 `/api/history/{id}/report` 也受保护，因此启用鉴权后，前端必须用带请求头的 `fetch` 获取 HTML，再创建临时 Blob URL 打开；普通 `<a href>` 无法附加密钥。密钥不得写入查询参数、日志、错误消息或静态前端构建变量。
