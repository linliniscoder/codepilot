# CodePilot

CodePilot 是一个仓库级 coding agent，用于将自然语言任务转成可执行计划，并在 workspace 中完成文件修改、测试验证和失败后的重试。

项目重点在于四个方面：

- 任务闭环
- 文件安全
- 失败恢复
- 可量化评测

## 当前结果

以下数据来自 `evaluation/benchmark_40.json` 的 40 个任务，使用当前默认配置 `full` 作为基线。

| 变体 | task_success | test_pass | first_pass | recovery | 平均迭代 | 平均延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| full | 0.700 | 0.700 | 0.300 | 0.400 | 5.70 | 28.78s |
| no_debug_reflection | 0.675 | 0.675 | 0.300 | 0.375 | 5.38 | 26.36s |
| no_recovery | 0.300 | 0.300 | 0.300 | 0.000 | 3.03 | 14.07s |
| one_shot | 0.000 | 0.050 | 0.050 | 0.000 | 1.00 | 7.67s |

结论上，恢复机制对成功率影响最大，debug reflection 提供小幅增益，单步模式适合作为对照基线。

## 项目关注点

| 文件 | 作用 |
| --- | --- |
| `agent/controller.py` | 主循环，负责 plan-act-observe-reflect，以及失败后的重新规划。 |
| `tools/file_tool.py` | workspace 边界、先读后改、精确替换，避免越界写入。 |
| `evaluation/benchmark.py` | 隔离 benchmark、hidden tests 和结果统计。 |

这三个文件可以直接反映项目是否只停留在 demo，还是已经具备边界控制和恢复能力。

## 问题定义

很多 coding agent 的问题不在于生成 patch，而在于无法稳定处理仓库边界、测试失败和重复尝试。CodePilot 针对这些问题增加了约束，使任务可以在真实仓库里形成闭环。

## 工作流程

```text
用户任务
   |
   v
Planner -> 结构化 JSON plan
   |
   v
Controller
   |
   +--> Tool Registry
   |       +--> 读文件 / 搜索文件
   |       +--> 精确编辑
   |       +--> workspace shell 命令
   |       +--> pytest 验证
   |
   +--> Observation
   |
   +--> Reflection / Diagnostics
          |
          +--> 通过则结束
          +--> 失败则重新规划
```

核心在于工具结果和反思回路，而不是单次生成的 patch。

## 设计约束

- 只允许在配置的 workspace 里操作文件和命令。
- 既有文件必须先读，再做精确文本替换。
- 替换必须精确命中一次，避免误伤。
- 测试文件默认只读，避免通过改测试掩盖问题。
- pytest 失败后，会把失败路径、import 链和相关源码一起喂回下一轮。
- 上下文有预算控制，避免请求越滚越大。
- retrieval 是可选项，不强依赖。

这些约束用于降低误写、重复修复和上下文膨胀风险。

## 验证机制

`evaluation/benchmark.py` 提供隔离 benchmark。每个任务在临时副本上运行，hidden tests 只在 agent 完成后注入，避免执行阶段直接读取 benchmark 答案。结果会写入 `evaluation/results.json`。

当前单元测试覆盖 controller 行为、JSON 修复、路径处理、diagnostics、retrieval fallback 和 evaluation metrics。

## 项目结构

- `agent/`：planner、controller、reflection、diagnostics 和共享 state。
- `llm/`：vLLM client 和 prompt。
- `tools/`：文件、shell、test 三类工具。
- `retrieval/`：tree-sitter、embedding、FAISS 和检索。
- `evaluation/`：benchmark、metrics 和结果输出。
- `webui.py`：本地 UI，用于观察任务、步骤、工具调用和测试结果。

## 快速开始

### 1. 安装依赖

参考运行环境是 Python 3.11。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

LLM server 需要单独安装，因为 vLLM、PyTorch、CUDA 和 FlashInfer 都和硬件环境强相关。参考环境使用的是 vLLM `0.27.1`。

### 2. 启动 vLLM

CodePilot 只调用 OpenAI-compatible 接口，不负责启动或管理模型服务。

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct
```

如果模型已经下载，也可以直接传本地 snapshot 路径。

```bash
vllm serve /path/to/Qwen2.5-Coder-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct
```

启动后可通过以下命令确认服务可用：

```bash
curl http://127.0.0.1:8001/v1/models
```

### 3. 配置 `config.yaml`

```yaml
vllm:
  server_url: "http://localhost:8001/v1"
  context_window: 4096

model:
  name: "Qwen/Qwen2.5-Coder-7B-Instruct"

paths:
  workspace: "./workspace"
  embedding_model: "./models/embedding-model"
  faiss_index: "./retrieval/index.faiss"

retrieval:
  device: "cpu"
```

`config.yaml` 里的 model 名称要和 vLLM 的 served name 对上。`context_window` 低于 server 的最大长度，用于控制工具日志带来的上下文膨胀。

### 4. 运行任务

目标仓库放入 `workspace/`，或改成其他可丢弃 checkout。

```bash
python main.py --task "Fix the login endpoint when password is empty"
```

```bash
python main.py \
  --task "Fix the failing order total test" \
  --max-iterations 10
```

### 5. 查看 Web UI

```bash
python webui.py
```

打开 `http://127.0.0.1:8080` 后，可以查看任务、计划、工具名、工具参数、工具结果、反思消息、修改文件和测试状态。

## 可选 Retrieval

retrieval 不是核心依赖，但在仓库变大后会有帮助。它会把相关代码片段先找出来，再喂给 planner 和 reflection。

```bash
python -m retrieval.build_index \
  --root workspace \
  --index retrieval/index.faiss \
  --device cpu
```

```bash
python main.py \
  --task "Fix the caching bug in the user repository" \
  --with-retriever
```

## 测试和 Benchmark

```bash
python -m pytest -q
```

```bash
python -m evaluation.benchmark
```

benchmark 结果会保存到 `evaluation/results.json`。

深度评测脚本支持 40 个任务、基线和消融对比：

```bash
python -m evaluation.build_benchmark_40
python -m evaluation.experiments
```

默认会输出 `full`、`no_debug_reflection`、`no_recovery`、`one_shot` 四组结果，并生成分组统计。


## License

见 [LICENSE](LICENSE)。
