# 前后端启动与验收教程（Windows / PowerShell）

本文适用于本项目的本地开发、课程演示和虚拟机演示。命令默认在 PowerShell 中执行，Python 一律使用项目内的 `.venv`。

## 1. 启动逻辑

```text
浏览器 http://localhost:5173
  -> Vite 将 /api/* 转发到 http://127.0.0.1:8000
  -> FastAPI 加载 checkpoints/ 中的检测与家族分类模型
  -> 分析历史写入 data/history.db
  -> OpenRouter 生成可选的 LLM 报告
```

请注意：

- 使用两个 PowerShell 窗口，先启动后端，再启动前端。
- 前端代理当前固定指向后端 `8000` 端口。
- 不要同时启动多个后端实例，每个实例都会单独加载模型。
- 日常运行不需要重新训练，也不需要每次重新安装依赖。

## 2. 首次运行前检查

需要 Python 3.11+、`uv`、Node.js 18+、npm 和 Git：

```powershell
python --version
uv --version
node --version
npm --version
git --version
```

## 3. 首次安装后端

```powershell
cd D:\study\Integrated_Design
uv venv .venv --python 3.11
uv pip install -p .venv -r requirements.txt
.\.venv\Scripts\python.exe --version
```

确认核心模型文件存在：

```powershell
Get-Item checkpoints\lightgbm.txt
Get-Item checkpoints\mlp.pt
Get-Item checkpoints\scaler.pkl
Get-Item checkpoints\family_mlp.pt -ErrorAction SilentlyContinue
```

前三个文件用于恶意软件检测，必须存在。`family_mlp.pt` 用于家族分类；缺失时基础检测仍可运行，但家族分类不可用。启动系统不需要重新训练模型。

### 可选：GPU 版 PyTorch

只有在 NVIDIA 驱动和 CUDA 环境兼容时才执行：

```powershell
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

GPU 不是必要条件，CPU 模式可以完成课程演示。

## 4. 首次安装前端

打开另一个 PowerShell 窗口：

```powershell
cd D:\study\Integrated_Design\webapp\frontend
npm install
```

只有首次运行或 `package.json`、`package-lock.json` 变化后才需要重新执行 `npm install`。

## 5. 可选环境变量

变量必须在启动后端的同一个窗口中设置，并且要在运行 Uvicorn 之前设置。

启用 LLM 报告：

```powershell
$env:OPENROUTER_API_KEY = "你的 OpenRouter API Key"
```

不设置不会影响 PE 检测和家族分类，只会使 LLM 报告不可用。

启用后端 API Key：

```powershell
$env:MALGUARD_API_KEY = "至少16位的随机字符串"
```

当前前端尚未接入 API Key 请求头，因此使用前端演示时不要启用此变量。该模式目前适合 Swagger 或脚本调用，请求需携带 `X-API-Key`。

## 6. 启动后端（终端 A）

```powershell
cd D:\study\Integrated_Design\webapp\backend
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

出现 `Uvicorn running on http://127.0.0.1:8000` 后保持窗口打开。稳定演示不建议使用 `--reload`；只有修改后端代码时才使用：

```powershell
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## 7. 验证后端

另开 PowerShell 执行：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health | Format-List
(Invoke-WebRequest http://127.0.0.1:8000/api/ready).StatusCode
Invoke-RestMethod http://127.0.0.1:8000/api/metrics
```

正式演示前应确认：

- `/api/health` 中 `ok` 为 `True`。
- `mode` 为 `real`，不能是模拟模式。
- `modelsLoaded` 为 `True`。
- `modelProvenanceVerified` 为 `True`。
- `/api/ready` 返回 HTTP `200`。
- `familyModelLoaded` 为 `True`（需要家族分类时）。

`/api/health` 只说明进程存活，`/api/ready` 才说明模型已准备好。Swagger 地址为 `http://127.0.0.1:8000/docs`。

## 8. 启动前端（终端 B）

```powershell
cd D:\study\Integrated_Design\webapp\frontend
npm run dev
```

访问 `http://localhost:5173`。Vite 会把 `/api/*` 转发到 `http://127.0.0.1:8000`。如果后端改用其他端口，当前代理将无法连通；日常使用应让后端保持 `8000`。

## 9. 完整联调验收

1. 先打开 `http://127.0.0.1:8000/api/ready`，确认后端正常。
2. 打开 `http://localhost:5173`，确认前端可以加载。
3. 上传来源明确的本地 PE 文件做静态分析，不要运行未知文件。
4. 检查检测结果、置信度、模型状态和可选的家族分类结果。
5. 打开历史记录页面，确认分析记录已写入数据库。
6. 设置了 `OPENROUTER_API_KEY` 时再检查 LLM 报告，否则跳过。

当前前端保留了离线模拟结果兜底：后端连接失败时，页面可能仍显示模拟结果。因此不能仅凭页面出现结果判断后端已连通，正式演示前必须验证 `/api/ready`。

`demo_samples/suspicious_demo.exe` 是项目的合成可疑样本，不应描述为真实恶意软件。部分正常安装包可能误报，也不应只用单个样本证明模型效果。

## 10. 日常启动速查

终端 A：

```powershell
cd D:\study\Integrated_Design\webapp\backend
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

终端 B：

```powershell
cd D:\study\Integrated_Design\webapp\frontend
npm run dev
```

然后访问 `http://localhost:5173`。

## 11. 局域网或虚拟机访问

后端改为监听全部网卡：

```powershell
cd D:\study\Integrated_Design\webapp\backend
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

前端已监听 `0.0.0.0`，正常执行 `npm run dev` 即可。使用 `ipconfig` 查询本机 IPv4 地址，其他设备访问 `http://<运行项目的电脑IP>:5173`。

浏览器仍通过 Vite 代理访问后端，通常只需允许可信专用网络访问前端端口 `5173`。不要在公共网络直接暴露开发服务器、模型接口或 API Key。

## 12. 停止服务

回到前端和后端窗口，分别按 `Ctrl+C`。检查端口：

```powershell
Get-NetTCPConnection -LocalPort 8000,5173 -State Listen -ErrorAction SilentlyContinue
```

若仍有旧进程，先确认身份：

```powershell
$connection = Get-NetTCPConnection -LocalPort 8000 -State Listen
Get-Process -Id $connection.OwningProcess
```

确认是旧项目进程后再执行 `Stop-Process -Id <PID>`。不要在未确认身份时批量结束所有 Python 或 Node 进程。

## 13. 常见问题

### `ModuleNotFoundError: No module named 'app'`

启动目录错误。先进入 `webapp\backend`，再运行 Uvicorn。

### 端口 8000 被占用

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Format-Table -AutoSize
```

确认并停止旧后端后重启。虽然可以换端口，但前端代理也必须同步修改，因此优先保留 `8000`。

### `/api/health` 正常但 `/api/ready` 返回 503

说明进程存活但模型加载失败。查看终端 A 日志，并检查 `checkpoints\lightgbm.txt`、`mlp.pt` 和 `scaler.pkl`。

### 后端关闭后，前端仍显示分析结果

这是前端离线模拟兜底，不代表真实推理完成。先检查 `/api/ready`；移除静默兜底已列入前端待办。

### 前端请求返回 401

后端启用了 API Key，而当前前端没有发送请求头。前端演示时清除变量并重启后端：

```powershell
Remove-Item Env:MALGUARD_API_KEY -ErrorAction SilentlyContinue
```

### LLM 报告不可用

确认在启动后端的同一终端设置了 `OPENROUTER_API_KEY`，然后重启后端。LLM 不影响核心检测和家族分类。

### `uv` 缓存权限或跨磁盘链接错误

```powershell
$env:UV_NO_CACHE = "1"
$env:UV_LINK_MODE = "copy"
uv pip install -p .venv -r requirements.txt
```

### `npm` 或 `node` 命令不存在

安装 Node.js 18+，重新打开 PowerShell，再检查版本。

### 浏览器出现跨域错误

使用 `http://localhost:5173` 并通过相对路径 `/api` 调用时，Vite 代理通常不会产生跨域问题。只有让浏览器直接请求其他地址的后端时，才需要调整 CORS。

## 14. 演示前检查清单

- 后端只有一个实例，监听 `127.0.0.1:8000`。
- `/api/ready` 返回 HTTP `200`。
- 健康检查显示真实模式、模型已加载、来源校验通过。
- 前端运行在 `http://localhost:5173`。
- 已准备来源明确的正常样本和项目合成演示样本。
- 已说明家族分类只在恶意判定后触发，结果属于辅助判断。
- LLM 不可用时仍能完成核心检测流程。
- 不把模拟结果、合成样本或单次预测描述成正式测评结论。

## 15. 修改代码后的自检

后端测试：

```powershell
cd D:\study\Integrated_Design
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

前端检查：

```powershell
cd D:\study\Integrated_Design\webapp\frontend
npm run lint
npm run build
```

接口字段、批量分析和历史记录说明见 [后端接口契约](api-batch-history.md)，前端尚待完成的联调项见 [前端待办清单](frontend-tasks.md)。
