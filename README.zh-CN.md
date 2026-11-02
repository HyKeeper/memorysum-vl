# memorysum-vl

**给 Agent 用的、可自托管的多模态记忆服务。** 用 `Add` 灌入对话（文字 + 截图），
用 `Search` 检索（纯文字、纯图片或图文混合）。一个 Python 进程、一个 SQLite 文件、
三个由本地 [llama.cpp](https://github.com/ggml-org/llama.cpp) 服务的小模型——
没有独立向量数据库，不调用任何外部 API，数据不出本机。

[English](README.md)

```
POST /v1/add       消息 + 图片 ─┐
                                ├─► SQLite（原文 · 原图 · FTS5 · sqlite-vec）
POST /v1/search    问题 + 图片 ─┘        ▲            ▲             ▲
                                         │            │             │
                                   Rax-4.5    Qwen3-VL-Embedding-2B
                                （图 → 文字）    （文本与图片同一向量空间）
                                                        │
                                                bge-reranker-v2-m3
                                                   （交叉编码器精排）
```

## 为什么这么做

Agent 记忆的难点往往不是「检索不到文字」，而是**答案只存在于某张截图里**，
而那张图躺在三周前的会话里。常见实现要么只做关键词、要么只做向量切片，
这个项目两条都留，并让它们互相校验：

* **图片转成文字，但像素不丢。** 每张图只描述一次（按 `sha256` 缓存），原始字节留在库里，
  命中时可以直接把图片交给上层的答案模型。
* **多路召回、按名次融合。** 三路词面（分词、字符 trigram、中文 jieba）+ 每个查询向量
  对应的 `dense_text`/`dense_image` + 多选题的逐选项召回，用 RRF 融合，
  分数不可比的问题直接消失。
* **精排决定头部。** `bge-reranker-v2-m3` 对融合后的头部逐对打分，再把精排名次与
  RRF 名次融合一次，多路共识仍然算数。
* **分数是标定过的。** 返回 `sigmoid(重排 logit)`，`0.02` 就真的代表「没什么相关」，
  于是才能做弃答（`data: []`）而不是硬凑 100 条。
* **降级是显式的。** 任意模型服务挂掉，`Add`/`Search` 仍然 `200`，
  受影响的单元被标成 `caption_state` / `embedding_state = missing`，
  不会假装一切正常。

## 用到的模型

三个模型全部跑在一张 16 GB 显卡上（常驻约 12.4 GB）。

| 角色 | 模型 | 权重 | 本项目的 GGUF |
|---|---|---|---|
| 图片转文字 | **Rax-4.5**（1.9B VLM，`qwen35` 架构） | [raxcore-dev/Rax-4.5](https://huggingface.co/raxcore-dev/Rax-4.5) | [mradermacher/Rax-4.5-GGUF](https://huggingface.co/mradermacher/Rax-4.5-GGUF) · `Rax-4.5.Q8_0.gguf` + `Rax-4.5.mmproj-f16.gguf` |
| 向量 | **Qwen3-VL-Embedding-2B**（2048 维，文本与图片同空间） | [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | [DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF](https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF) · `Qwen.Qwen3-VL-Embedding-2B.f16.gguf` + `mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf` |
| 精排 | **bge-reranker-v2-m3**（交叉编码器） | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | [gpustack/bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)（Q8_0；本项目实测用 f16） |

推理运行时：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)（CUDA 版）。

文件大小、完整启动参数、一行下载命令，以及五个真实踩坑（缺 mmproj、多模态占位符随机化、
图片 token 撑爆 ubatch、推理模型正文为空、HF 镜像）见
**[docs/models.md](docs/models.md)**。

## 快速开始

前置：Linux、Python 3.12+、16 GB 显存、已编译好的 CUDA 版 `llama.cpp`。

```bash
# 1. 代码与依赖
git clone https://github.com/HyKeeper/memorysum-vl.git
cd memorysum-vl
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 2. 下载权重（约 6.6 GB，已存在会跳过）
export MODEL_DIR=~/models
./deploy/llama/download-models.sh

# 3. 起三个模型服务
./deploy/llama/start-reranker.sh  &
./deploy/llama/start-captioner.sh &
./deploy/llama/start-embedder.sh   &

# 4. 配置 token
cp .env.example .env
mkdir -p .secrets
python -c "import secrets; print(secrets.token_urlsafe(24))" > .secrets/api_token
chmod 600 .secrets/api_token

# 5. 启动
./run.sh start && ./run.sh status
```

```bash
TOKEN=$(cat .secrets/api_token)

curl -s -X POST localhost:8003/v1/add \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"request_id":"r1","user_id":"u1","session_id":"s1",
       "messages":[{"role":"user","content":"备用钥匙在蓝色抽屉里"}]}'

curl -s -X POST localhost:8003/v1/search \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","query":"钥匙放哪了？","top_k":5}'
```

Docker、systemd、参数调优、备份与排障见 **[docs/deployment.md](docs/deployment.md)**。

## 接口一览

| 方法 | 路径 | 鉴权 | |
|---|---|---|---|
| `POST` | `/v1/add` | Bearer | `{request_id, user_id, session_id, messages[]}` → `{success: true, ...}` |
| `POST` | `/v1/search` | Bearer | `{user_id, query, top_k, options?}` → `{data: [{id, content, score, created_at}]}` |
| `GET` | `/healthz` | 免鉴权 | `{status: "ok", track: "multimodal"}` |

完整契约、字段约束与错误码见 **[docs/api.md](docs/api.md)**；
可直接跑的示例见 `examples/quickstart.sh` 与 `examples/client.py`。

## 一次检索发生了什么

```
查询文本 + 查询图片
  → 查询图片先描述（按 sha256 缓存）  → 弃答判定
  → 词面：fts_word · fts_trigram · fts_zh · options
  → 向量：查询文本 / 每张图 / 图文联合，各走 dense_text 与 dense_image
  → RRF 融合（k 自适应、按路加权）→ 取 max(rerank_top_n, top_k)
  → bge-reranker-v2-m3 精排头部
  → 精排名次与 RRF 名次再融合 → sigmoid 标定分数
  → 弱证据收口 · 邻接消息 · Session 标签 · 可选原图
  → top_k
```

分层、存储表结构、以及每一步为什么这么做：**[docs/architecture.md](docs/architecture.md)**。

## 目录结构

```
src/memorysum_vl/
├── api/        FastAPI 路由、Bearer 鉴权、ASGI 请求日志、协议映射
├── services/   AddService / SearchService —— 只做编排
├── domain/     记忆模型 + 端口（MemoryStore / Embedder / Captioner / Reranker）
├── adapters/   sqlite_store、repository（FTS5 + sqlite-vec）、三个 llama.cpp 客户端
└── core/       配置、依赖装配、错误信封、日志、RRF、中文分词、图片处理
deploy/         llama.cpp 启动脚本、Dockerfile、compose 与 systemd 模板
docs/           架构 · 模型 · 接口 · 部署
examples/       curl 与 Python 客户端
scripts/        smoke_api.py（端到端自检）、backfill_transcripts.py（修复工具）
tests/          端到端测试（假模型服务，无需 GPU）
```

刻意做成六边形结构：`domain/ports.py` 是 services 唯一认识的接缝，
把 SQLite 换成 Postgres、把某个模型换掉，都不需要动 API 层。

## 开发

```bash
make install     # 可编辑安装 + dev 依赖
make test        # pytest：22 个端到端用例，假模型服务，无需 GPU
make lint        # ruff check
make fmt         # ruff format
make run         # 前台跑 uvicorn（:8003）
```

测试全程不打模型：`tests/conftest.py` 提供进程内的假 embedder/captioner/reranker，
整条 Add → Search 链路（含降级分支）一秒内跑完。

## 实测数字

RTX 5060 Ti 16 GB，单进程 + 三个模型服务（开发机数据，按数量级看即可）：

| 场景 | 结果 |
|---|---|
| `Search`，`top_k=20`，热态 | 单请求 **1.5–2.0 s** |
| `Search`，`top_k=100`，16 并发 | p50 ≈ 37 s（瓶颈在精排排队，GPU 打满） |
| `Add`，1 张图，3 个 caption 槽位 | 约 7 s |
| 评测时的库规模 | 499k 记忆单元、56k 图片、SQLite 30 GB |
| 内测 164 题上的证据命中 | 多跳 33/36、单跳 47/49 |

## 已知限制

* **单写者存储。** SQLite + WAL 在几亿 token 量级以内很好用，再往上就不合适了；
  存储实现了端口，接一个 Postgres/`pgvector` 适配器是最自然的下一步。
* **一次只服务一条赛道。** `MEMORYSUM_TRACK` 决定契约（`multimodal` / `text` / `code`），
  一个实例只服务其中一条。
* **描述质量决定召回上限。** 1.9B 的视觉模型抄不好密集文档，
  `scripts/backfill_transcripts.py` 的存在就是因为描述有损、值得重做。
* **鉴权只有共享 Bearer token**（见[部署说明第 9 节](docs/deployment.md#9-security-notes)）。

## 许可

MIT，见 [LICENSE](LICENSE)。© 2026 HyKeeper
