from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Track(StrEnum):
    TEXT = "text"
    CODE = "code"
    MULTIMODAL = "multimodal"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEMORYSUM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "memorysum-vl"
    app_version: str = "0.1.0"
    api_prefix: str = "/v1"
    track: Track = Track.MULTIMODAL
    log_level: str = "INFO"
    auth_token: SecretStr | None = None
    auth_token_file: Path | None = None

    # 日志
    log_format: str = "text"
    log_file: Path | None = None
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5
    log_request_body_limit: int = 0  # 0 = 全量打印
    log_response_body_limit: int = 0  # 0 = 全量打印
    log_quiet_paths: str = ""

    # 存储：SQLite 里同时放原文、图片字节、FTS5 词面索引与 sqlite-vec 向量
    db_path: Path = Path("data/memorysum.db")
    # 送给模型前把图片长边压到这个像素，Qwen3-VL 按 32px 切 patch，1280 约 1024 token
    image_max_side: int = 1280

    # 嵌入：Qwen3-VL-Embedding-2B（llama.cpp /v1/embeddings，文本与图片同空间）
    embedding_base_url: str = "http://127.0.0.1:7998"
    embedding_model: str = "qwen3-vl-embedding-2b"
    embedding_dim: int = 2048
    embedding_timeout: float = 30.0
    embedding_batch_size: int = 8
    embedding_concurrency: int = 4
    embedding_retries: int = 2
    embedding_instruction: str = "Represent the user's input."
    embedding_chat_template: bool = True
    embedding_media_marker: str = "<__media__>"
    # 超长查询（文档级题面）在送模型前保留头尾，避免撞上 ubatch/上下文上限
    embedding_max_query_chars: int = 1200
    rerank_max_query_chars: int = 6000

    # 图片描述：Rax-4.5（llama.cpp /v1/chat/completions，关掉思维链直接出描述）
    caption_base_url: str = "http://127.0.0.1:7996"
    caption_model: str = "rax-4.5"
    caption_timeout: float = 90.0
    caption_concurrency: int = 3
    caption_max_tokens: int = 320
    caption_retries: int = 1
    caption_cache: bool = True

    # 重排：bge-reranker-v2-m3（llama.cpp /rerank）
    rerank_base_url: str = "http://127.0.0.1:7997"
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_enabled: bool = True
    rerank_timeout: float = 20.0
    rerank_retries: int = 1
    rerank_top_n: int = 60
    rerank_doc_chars: int = 1200

    # 融合：每路召回多少候选、RRF 的 k、以及按路名覆盖权重
    route_pool: int = 120
    rrf_k: int = 20
    rrf_k_adaptive: bool = True
    fusion_weight: float = 2.0
    route_weights: str = ""
    max_query_images: int = 4

    # 弃答：问题实词在语料里一个都没出现时返回空列表，而不是硬凑最像的记忆。
    # 默认关闭：实测交空 data 反而少得分（9 题 3→2），保留为显式开关。
    abstain_enabled: bool = False
    abstain_min_terms: int = 2
    abstain_mass_ratio: float = 0.7

    # 选项召回：多选题的每个选项都是一条查询
    options_enabled: bool = True
    options_per_option: int = 2
    max_options: int = 8

    # 返回内容：默认把命中记忆里的原图一起回传（图里的字往往不在描述里）
    return_images: bool = True
    response_image_max_side: int = 1600
    response_image_quality: int = 92
    max_returned_images: int = 40
    response_image_passthrough_bytes: int = 4 * 1024 * 1024
    max_response_image_bytes: int = 24 * 1024 * 1024
    content_include_timestamp: bool = True
    content_include_role: bool = True
    # 返回内容里带上「Session N (日期)」：会话 id 是哈希，答案模型只能靠序号+时间定位
    content_session_label: bool = True
    # 弱证据收口：首条命中置信度低于阈值时只返回前 N 条（0 = 不收口）
    weak_evidence_threshold: float = 0.05
    weak_evidence_limit: int = 0
    # 每条命中附带同会话相邻消息（多跳证据往往就在相邻几轮里）
    neighbor_expand: int = 2
    neighbor_expand_chars: int = 1600

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
            raise ValueError(f"unsupported MEMORYSUM_LOG_LEVEL: {value!r}")
        return level

    @field_validator("log_format")
    @classmethod
    def _check_log_format(cls, value: str) -> str:
        style = value.strip().lower()
        if style not in {"text", "json"}:
            raise ValueError(f"unsupported MEMORYSUM_LOG_FORMAT: {value!r}")
        return style

    @field_validator("log_file")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("rrf_k")
    @classmethod
    def _check_rrf_k(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"MEMORYSUM_RRF_K must be >= 1, got {value}")
        return value

    @field_validator("embedding_dim", "route_pool", "rerank_top_n")
    @classmethod
    def _check_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"value must be >= 1, got {value}")
        return value

    @property
    def quiet_paths(self) -> frozenset[str]:
        """命中这些路径且成功时，请求日志降级为 DEBUG。"""
        return frozenset(part.strip() for part in self.log_quiet_paths.split(",") if part.strip())

    @property
    def parsed_route_weights(self) -> dict[str, float]:
        """`dense_text=1.2,lexical=0.8` → dict；空值表示全 1.0。"""
        weights: dict[str, float] = {}
        for chunk in self.route_weights.split(","):
            name, _, raw = chunk.partition("=")
            name = name.strip()
            if not name or not raw.strip():
                continue
            try:
                weights[name] = float(raw)
            except ValueError as exc:
                raise ValueError(f"invalid route weight {chunk!r}") from exc
        return weights


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
