# MalGuard AI — 深度学习恶意软件检测系统

上传 Windows PE 可执行文件，由 LightGBM + MLP 双模型集成检测，并基于静态结构线索提供规则化风险解释与 ATT&CK 技术关联提示。

基于 **EMBER2024 Win64** 子集的 2568 维静态特征（`thrember` / `pefile` 提取），训练与推理共用同一条特征流水线，无需沙箱即可对任意 `.exe` 做端到端检测。

## 快速开始

### 环境要求

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)（Python 包管理器）
- Node.js 18+ & npm
- Git（`thrember` 从 GitHub 安装需要）

### 1. 后端

```bash
# 创建虚拟环境并安装依赖
uv pip install -p .venv -r requirements.txt

# 如需 GPU 加速（可选，默认 CPU 版 torch）
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# 稳定演示（须从 webapp/backend 目录进入）
cd webapp/backend
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 仅开发后端代码时使用热重载；不要与上一条命令同时运行
# ..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

> **Linux / macOS** 把 `..\..\.venv\Scripts\python.exe` 换成 `../../.venv/bin/python`。
后端启动后可访问 `GET /api/health` 查看可选组件状态，使用 `GET /api/ready` 判断核心模型是否可用于真实检测。checkpoint 缺失或不兼容时，检测接口返回 503；仅纯接口联调时可显式设置 `ALLOW_STUB_PREDICTIONS=1`，正式演示不要启用。

后端启动时会校验以下可选环境变量，值不合法会直接拒绝启动：

- `MALGUARD_CORS_ORIGINS`：允许直接调用 API 的前端来源，使用逗号分隔；默认允许 `http://localhost:5173` 和 `http://127.0.0.1:5173`，不接受 `*`。
- `MALGUARD_INFERENCE_CONCURRENCY`：共享模型可同时执行的推理数，范围 1～8，默认 1。单 GPU 演示建议保留 1；只有经过显存和吞吐压测后再提高。

### 2. 前端

新开一个终端：

```bash
cd webapp/frontend
npm install
npm run dev
```

浏览器打开 <http://localhost:5173> 即可使用。

### 3. LLM 静态风险说明（可选）

系统会将已提取的静态结构线索（敏感导入、节区熵、签名、编译时间、版本信息及受限的 URL/IP/注册表/命令字符串）交给 OpenRouter 生成辅助说明，不会上传文件原始字节；该说明不参与最终检测结论。需要设置环境变量：

```bash
# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "sk-or-v1-..."

# Linux / macOS
export OPENROUTER_API_KEY="sk-or-v1-..."
```

未配置 API Key 时，核心检测不受影响，后端会明确返回 LLM 分析不可用，不会伪造 LLM 结论。

### 4. 家族分类模型（可选）

检测结果中的"家族分类"字段由一个独立的 MLP 多分类模型给出（与二分类 MLP 同一套 `src/models/mlp.py` 分支注意力融合架构，仅输出层宽度换成家族类别数），仅对判定为恶意的样本生效。该模型不是必需的：`checkpoints/` 下若没有 `family_mlp.pt` / `family_labels.json`，后端会自动跳过家族预测，`family` 字段返回 `null`，不影响检测本身。

训练（需要先跑过 `train_mlp.py`——家族模型复用它产出的 `checkpoints/scaler.pkl`，不单独拟合；还需要先按上面步骤跑通特征提取，`data/raw/ember2024/family_train.json` 与 `family_test.json` 需已由 `src/data/extract_family_labels.py` 生成）：

```bash
.venv\Scripts\python.exe src/models/train_mlp.py     # 若 checkpoints/scaler.pkl 尚不存在
.venv\Scripts\python.exe src/models/train_family.py
```

家族训练同样直接从 memmap 按批读取，只在 GPU 训练阶段启用 AMP；验证选模和官方测试保持 FP32。脚本在最佳 checkpoint 确定后才打开 test，并写出 `family_training_manifest.json`。440 类混淆矩阵只展示测试集支持度最高的 30 类，其余已建模家族合并为一个可读分组；完整逐类指标仍保留在文本报告中。

EMBER2024 的家族标签长尾分布严重（数据集全量 6787 个家族，多数只有个位数样本），因此该模型只对训练集中样本数 ≥ `configs/family.yaml` 里 `min_count`（默认 30）的家族分开建模，其余全部归入"其他"类；预测为"其他"时，前端不展示家族名（等价于未知家族）。早期版本用 LightGBM 做多分类，原生多分类目标每轮要为每个类别单独建一棵树，440 个类别 × 200 轮 ≈ 8.8 万棵树，实测训练要 2 小时以上；换成 MLP 后输出层只是把最后一层 `Linear` 的输出维度从 1 换成类别数，训练在分钟级完成。

## 训练与正式评估

以下训练命令会覆盖 `checkpoints/` 中后端正在使用的模型，只应在确认数据、配置和输出目录后手动执行：

```bash
.venv\Scripts\python.exe src/models/train_lightgbm.py
.venv\Scripts\python.exe src/models/train_mlp.py
```

默认数据目录是仓库内的 `data/raw/ember2024`；若数据放在其他磁盘，可设置 `EMBER2024_DATA_DIR`。

MLP 训练保持特征文件为 memmap，增量拟合 `StandardScaler`，并在 DataLoader 中逐批标准化；CUDA 环境默认启用 PyTorch AMP、固定随机种子和 pinned memory。LightGBM 训练只读取固定 train/validation 划分，不接触官方 test。两者完成后会分别写出训练清单，记录配置、划分、最佳验证指标、运行环境、Git 状态和模型 SHA-256。

模型训练完成后，使用同一评估入口刷新正式结果：

```bash
.venv\Scripts\python.exe src/eval/compare_models.py
```

评估脚本对官方有标签 test memmap 分批推理，使用与后端一致的 FP32 MLP 路径，同时评估 LightGBM、MLP 和实际部署的概率平均集成。输出为 `checkpoints/metrics.json`、`evaluation_manifest.json` 和 `confusion_matrices.png`；评估不会覆盖模型权重。

## 从虚拟机访问

如果需要在虚拟机中访问宿主机上运行的服务，前端 `vite.config.ts` 已绑定 `0.0.0.0`，并通过同源 `/api` 代理到宿主机后端，因此开发模式不需要额外 CORS 配置。在虚拟机浏览器中访问 `http://<宿主机IP>:5173`；宿主机 IP 可通过 `ipconfig`（Windows）或 `ip addr`（Linux）查看。

若前端不经过 Vite/反向代理，而是从另一个来源直接请求 `http://<宿主机IP>:8000`，启动后端前必须显式放行该来源：

```powershell
$env:MALGUARD_CORS_ORIGINS = "http://<访问前端所用IP或域名>:5173"
```

还需确保宿主机防火墙仅对可信网络开放所需端口。不要为了省略配置把 CORS 设置成通配符。

## 项目结构

```
.
├── checkpoints/          # 训练好的模型权重（lightgbm.txt, mlp.pt, scaler.pkl,
│                         #   family_mlp.pt/family_labels.json 为可选的家族分类模型，复用 scaler.pkl）
├── configs/              # 超参数配置（YAML）
│   ├── lightgbm.yaml
│   ├── mlp.yaml
│   ├── family.yaml       # 家族分类模型（可选）
│   └── llm.yaml          # LLM 模型与参数
├── src/
│   ├── features/         # PE 特征提取（EMBER 2568 维）
│   ├── data/             # 数据加载与预处理
│   ├── models/           # LightGBM + MLP 模型定义与训练脚本
│   ├── eval/             # 模型对比评估
│   └── llm/              # LLM 分析模块（ATT&CK 规则 + 报告生成）
├── webapp/
│   ├── backend/          # FastAPI 后端
│   └── frontend/         # Vite + React + TypeScript 前端
├── demo_samples/         # 演示用样本构建工具（非真实恶意软件）
└── Paper/                # 参考论文
```

## 模型性能（测试集）

| 模型 | 准确率 | F1 |
|------|--------|------|
| LightGBM (基线) | 97.23% | 97.23% |
| MLP (特征组注意力融合) | 98.35% | 98.34% |
| LightGBM + MLP 集成（部署模型） | 98.38% | 98.37% |

结果来自 240,000 条 EMBER2024 Win64 官方有标签测试样本，阈值为 0.5。最终检测结论由两个模型的恶意概率算术平均决定；完整协议、混淆矩阵和 checkpoint 哈希见 `checkpoints/evaluation_manifest.json`。

## 常见问题

**Q: `ModuleNotFoundError: No module named 'app'`**
后端必须从 `webapp/backend` 目录启动，不能从项目根目录直接跑。

**Q: 前端上传后白屏**
按 F12 查看控制台错误。已知修复：`crypto.randomUUID()` 在非安全上下文（HTTP + 网络 IP）不可用，已替换为兼容方案。

**Q: `signify` 版本报错**
`requirements.txt` 已锁定 `signify==0.8.1`，不要手动升级到 0.9.0+（API 不兼容）。

## 协作与接口文档

- [后端接口契约](docs/api-batch-history.md)
- [前端待办清单](docs/frontend-tasks.md)
