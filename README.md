# MalGuard AI — 深度学习恶意软件检测系统

上传 Windows PE 可执行文件，由 LightGBM + MLP 双模型集成检测，并对恶意样本生成 LLM 行为分析报告与 ATT&CK 战术映射。

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

### 2. 前端

新开一个终端：

```bash
cd webapp/frontend
npm install
npm run dev
```

浏览器打开 <http://localhost:5173> 即可使用。

### 3. LLM 分析报告（可选）

恶意样本会自动调用 OpenRouter 生成行为分析报告。需要设置环境变量：

```bash
# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "sk-or-v1-..."

# Linux / macOS
export OPENROUTER_API_KEY="sk-or-v1-..."
```

未配置 API Key 时，检测功能本身不受影响，仅 LLM 报告部分会回退到前端 mock 数据。

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
├── checkpoints/          # 训练好的模型权重（lightgbm.txt, mlp.pt, scaler.pkl）
├── configs/              # 超参数配置（YAML）
│   ├── lightgbm.yaml
│   ├── mlp.yaml
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
