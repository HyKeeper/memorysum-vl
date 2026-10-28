# memorysum-vl

按评测协议实现的 Add / Search 记忆服务骨架。协议层的字段、校验、序列化与 Bearer 认证已就位，记忆写入与检索的算法尚未实现。

## 目录结构

```
src/memorysum_vl/
├── main.py              应用工厂，装配路由与容器
├── core/                配置、依赖容器、错误映射、Bearer 认证、媒体与限额
├── schemas/             HTTP 契约：Add、Search、内容分片、时间戳、错误信封
├── domain/              领域模型与端口（MemoryStore 协议）
├── services/            Add / Search 用例编排
├── adapters/stores/     端口实现与工厂
└── api/                 路由、依赖注入、Bearer 校验、协议与领域模型的双向映射
```

依赖方向为 `api → services → domain`，`adapters` 实现 `domain` 的端口，`core` 提供跨层基础设施。

## 认证

除 `/healthz` 外的所有接口都要求 `Authorization: Bearer <token>`。缺失或错误返回 401 并带 `WWW-Authenticate: Bearer`。

token 来源按下列顺序解析，启动时读取一次，不满足即启动失败：

1. `MEMORYSUM_AUTH_TOKEN_FILE`：文件路径，取第一条非空且不以 `#` 开头的行
2. `MEMORYSUM_AUTH_TOKEN`：直接给值

```
mkdir -p .secrets
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > .secrets/api_token
chmod 600 .secrets/api_token
export MEMORYSUM_AUTH_TOKEN_FILE=.secrets/api_token
```

`.secrets/` 已在 `.gitignore` 中。

## 运行

```
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" --index-url https://pypi.org/simple
MEMORYSUM_AUTH_TOKEN_FILE=.secrets/api_token .venv/bin/uvicorn memorysum_vl.main:app --app-dir src --reload
```

## 接口

| 方法 | 路径 | 认证 | 说明 |
| --- | --- | --- | --- |
| POST | `/v1/add` | Bearer | 写入一段会话消息，成功时原样回显 `request_id`、`user_id`、`session_id` |
| POST | `/v1/search` | Bearer | 按 `top_k` 返回记忆列表，`content` 随赛道决定是字符串还是有序内容分片 |
| GET | `/healthz` | 无 | 存活检查，返回当前赛道 |

赛道由 `MEMORYSUM_TRACK` 决定，取值为 `text`、`code`、`multimodal`。文本与代码赛道的 `content` 为字符串，多模态赛道为有序 `ContentPart[]`。

## 校验

请求校验失败返回 422。内容分片、单图 10 MiB、单次请求图片合计 30 MiB、`top_k` 上界 100 均在 `schemas` 层拦截。尚未实现的用例返回 501 与统一错误信封 `{"error": {"code": ..., "message": ...}}`。

认证先于请求体校验执行，未通过认证的请求不会进入校验环节。

## 待实现

`InMemoryMemoryStore.write` 与 `InMemoryMemoryStore.retrieve` 是当前唯一的空实现，也是接入记忆抽取、索引与检索的入口。
