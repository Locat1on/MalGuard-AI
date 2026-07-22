# 前端待办清单

> 本文件只记录前端工作，不直接修改 `webapp/frontend/`。后端接口以 `docs/api-batch-history.md` 和 FastAPI `/docs` 为准。

## P0：结果真实性与接口同步

### 1. 禁止网络失败时静默返回 mock 检测

现状：`src/lib/api.ts::analyzeFile()` 在 `fetch` 抛错时直接调用 `mockAnalyze()`，用户看到的结果外观与真实结果相同，可能把“后端未启动”误认为真实检测。

建议：
- 默认网络失败直接抛出 `DetectionFailedError("无法连接检测服务")`。
- 如仍需纯前端联调，仅在 `VITE_ENABLE_MOCKS=true` 时启用 mock。
- mock 模式必须在全局和结果卡片显示固定的“演示数据”标记，不能只藏在日志里。

验收：关闭后端上传文件时，正式模式只显示连接失败，不产生 verdict、家族名或历史记录。

### 2. 同步后端类型

在 `src/lib/types.ts` 补齐：
- `DetectionResult.historyId: number | null`
- `DetectionResult.featureAttention: FeatureAttention[] | null`
- `FeatureAttention { group, label, weight }`
- `HealthStatus`、`HistoryStats`
- 批量检测的 `BatchItem`、`BatchDetectionResult`
- 历史记录 ID 统一使用 `number`，不要在 API 层转成随机字符串。

验收：类型与 `docs/api-batch-history.md` 一致，`npm run build` 无类型错误。

### 3. 修复单次检测后的历史重复

现状：后端检测成功时已经写入 SQLite，`App.tsx` 又通过 `toHistoryEntry()` 生成随机 ID 并追加本地记录；刷新后会从后端重新加载，容易出现身份不一致或重复。

建议：
- 检测成功后使用响应中的 `historyId`，或直接重新请求 `GET /api/history`。
- 删除 `Date.now() + Math.random()` 生成历史 ID 的逻辑。
- HistoryPage 文案改为“记录保存在后端，刷新后仍可查看”，不再写“仅保存在当前浏览器会话”。

验收：连续检测两次、刷新页面后始终只有两条记录，ID 与后端一致。

### 4. 增加后端就绪状态

- 应用启动时请求 `GET /api/health`。
- `ready=false` 或检测返回 503 时，在上传区显示明确状态和 `modelLoadError`，并禁用上传按钮。
- `familyModelLoaded=false`、`llmConfigured=false` 是可选能力降级，不应阻止核心检测；可用小型状态提示说明“家族分类未加载”或“LLM 说明不可用”。
- 不要把 `/api/health` 的 `ok=true` 误解为模型已经就绪，应使用 `ready/modelsLoaded`。

## P1：把现有后端能力真正展示出来

### 5. 三方结果与 LLM 说明

当前结果卡只展示 LightGBM、MLP 和 LLM 文本，没有展示 `llmVerdict/llmConfidence`。

建议采用紧凑的三列对比：LightGBM 恶意概率、MLP 恶意概率、LLM 独立判定及置信度。明确标注：
- 最终 verdict 只由 LightGBM + MLP 平均得到。
- LLM 是基于有限静态线索的独立辅助意见，不参与最终概率。
- `llmVerdict=null` 时显示“未配置或分析失败”，不要显示 0%。

### 6. 特征组融合权重图

读取 `featureAttention`，按权重降序显示 12 个特征组，建议默认展示 Top 5，可展开全部。

措辞必须使用“特征组融合权重”或“模型内部关注权重”，不能写成“致因”“决定因素”或“该特征导致恶意判定”。注意力权重不是因果归因。

### 7. 完善历史页

- 请求 `GET /api/history/stats`，在表格上方展示总数、恶意/良性、模型分歧、LLM 分歧。
- 历史行增加来源（单文件/批量）、模型一致性、家族置信度。
- 增加查看详情、打开 `/api/history/{id}/report`、删除单条和清空历史。
- 删除操作需要确认；失败时保留现有列表并显示错误。
- 数据较多时使用 `limit/offset` 分页，不一次拉取全部。

### 8. 批量检测页面或模式

接入 `POST /api/detect/batch`：
- 支持多选/拖放，前端提前提示最多 100 个、单文件 100MB。
- 展示总数、恶意、良性、失败统计和逐文件错误。
- 明确提示批量模式不调用 LLM、不生成 ATT&CK 和融合权重。
- 支持按 verdict/失败状态筛选，并允许跳转对应 `historyId` 的报告。

### 9. 移动端导航

当前 `<nav>` 在 `md` 以下直接隐藏，移动端只能看到“开始检测”，无法进入指标和历史页。增加标准菜单按钮和抽屉/下拉菜单，保持键盘可操作、焦点可见，并在路由切换后自动关闭。

## P2：体验与质量

### 10. 加载和错误状态细化

区分：上传中、ML 推理中、LLM 分析中目前后端仍是一次响应，前端至少使用不会误导的统一“分析中”；处理 413、422、503 和网络错误的不同文案。不要把后端返回的技术异常堆栈样式文本做成成功提示。

### 11. 可访问性与布局验证

- 上传控件、删除按钮、移动菜单提供键盘操作和可见焦点。
- 状态变化使用 `aria-live`，错误信息与上传控件关联。
- 检查 375px、768px、1440px 三档，无文本溢出、按钮遮挡或横向滚动。
- 最终在普通本地终端运行 `npm run lint` 和 `npm run build`；Windows 受限沙箱出现 `spawn EPERM` 时，不能直接判断为代码错误。

## 建议实施顺序

1. P0-1～P0-4：先保证结果真实、类型和状态正确。
2. P1-5～P1-7：完善单文件结果与历史闭环。
3. P1-8～P1-9：批量工作流和移动端导航。
4. P2：统一收尾并做响应式、可访问性检查。
## 本轮接口增量

### 12. 指标页展示部署集成结果

`GET /api/metrics` 现在有第三行“LightGBM + MLP 集成（部署模型）”。指标页不要假定只有两个模型，按返回数组渲染，并将部署模型作为默认强调行。它代表线上最终 verdict 使用的算术平均概率。

增加一个紧凑的“指标来源”入口，按需请求 `GET /api/metrics/provenance`，展示测试集样本数、判定阈值、集成规则、评估时间、模型 SHA-256 前 12 位和 Git commit。接口 404 时只隐藏入口，不影响指标表。

### 13. 错误提示附带请求编号

API 层统一读取响应头 `X-Request-ID`。遇到 413、422、500、503 或网络错误时，可在详细信息中显示“请求编号”，便于后端排查；正常成功页面无需展示。不要记录或展示上传文件二进制内容。

### 14. 展示部署模型与正式指标的一致性

`GET /api/health` 新增 `modelProvenanceVerified` 和 `modelProvenanceWarning`：
- `true`：当前加载的 LightGBM、MLP、scaler 与正式评估清单哈希完全一致，可正常展示指标。
- `false`：模型文件已变化但指标未刷新；指标页显示醒目的“当前模型与评估版本不一致”，不要继续强调旧分数。
- `null`：评估清单缺失或不可读；显示中性“指标来源尚未核验”，核心检测仍可用。

状态区只展示简短结论，详细原因放在指标来源抽屉。该状态不是模型加载失败，不能据此禁用上传。
