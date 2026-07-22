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

# 启动后端（须从 webapp/backend 目录进入）
cd webapp/backend
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Linux / macOS** 把 `..\..\.venv\Scripts\python.exe` 换成 `../../.venv/bin/python`。
后端启动后可访问 `GET /api/health` 查看可选组件状态，使用 `GET /api/ready` 判断核心模型是否可用于真实检测。checkpoint 缺失或不兼容时，检测接口返回 503；仅纯接口联调时可显式设置 `ALLOW_STUB_PREDICTIONS=1`，正式演示不要启用。

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

EMBER2024 的家族标签长尾分布严重（数据集全量 6787 个家族，多数只有个位数样本），因此该模型只对训练集中样本数 ≥ `configs/family.yaml` 里 `min_count`（默认 30）的家族分开建模，其余全部归入"其他"类；预测为"其他"时，前端不展示家族名（等价于未知家族）。早期版本用 LightGBM 做多分类，原生多分类目标每轮要为每个类别单独建一棵树，440 个类别 × 200 轮 ≈ 8.8 万棵树，实测训练要 2 小时以上；换成 MLP 后输出层只是把最后一层 `Linear` 的输出维度从 1 换成类别数，训练在分钟级完成。

## 从虚拟机访问

如果需要在虚拟机中访问宿主机上运行的服务：

```bash
# 后端已经绑定了 0.0.0.0，无需额外配置
# 前端 vite.config.ts 已配置 server.host: '0.0.0.0'
```

在虚拟机浏览器中访问 `http://<宿主机IP>:5173`。宿主机 IP 可通过 `ipconfig`（Windows）或 `ip addr`（Linux）查看。

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
| MLP (特征组注意力融合) | 98.37% | 98.37% |

最终检测结论由两个模型的概率平均决定，同时报告两模型判定是否一致。

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
