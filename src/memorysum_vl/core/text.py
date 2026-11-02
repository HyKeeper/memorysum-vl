"""检索文本的规范化与 FTS 查询构造。

两套词面索引共用同一份原文，但用途不同：

* 词索引吃 `expand_for_word_index`，把中文连续串补成二元组，让 `unicode61`
  这种按空格切分的分词器也能命中中文；
* 三元组索引吃原文，交给 FTS5 的 `trigram` 分词器做子串匹配，专治拼写片段、
  编号、人名这类「字符对得上但词对不上」的查询。
"""

from __future__ import annotations

import re
from typing import Final

try:  # 中文分词：没有 jieba 就退回纯二元组，功能不丢
    import logging as _logging

    import jieba
    import jieba.posseg as pseg

    jieba.setLogLevel(_logging.ERROR)
    _JIEBA_READY: Final[bool] = True
except ImportError:  # pragma: no cover - 依赖已声明，兜底路径只做保险
    _JIEBA_READY = False

# 选项前缀（`A.` / `A)` / `(A)`），检索时去掉，只留选项正文
OPTION_LABEL: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:\(?[A-Za-z0-9]{1,3}\)?\s*[.、:：)]\s*|\(?[A-Za-z]\)\s*)"
)

# 拉丁词（含数字、下划线、内部连字符）与连续 CJK 串各算一个切分单元
_TOKEN: Final[re.Pattern[str]] = re.compile(
    r"[0-9A-Za-z]+(?:[._'\-][0-9A-Za-z]+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+"
)
_CJK: Final[re.Pattern[str]] = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+"
)

# 只过滤拉丁停用词；中文虚词单字在二元组里本来就会稀释，不做特殊处理
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)

MAX_TERMS: Final[int] = 48

# 中文功能词：只用于「弃答判定」取实词，不影响 FTS 取词
_ZH_STOPWORDS: Final[frozenset[str]] = frozenset(
    """我 你 他 她 它 我们 你们 他们 自己 的 了 是 在 有 和 与 或 就 都 也 还 又 很 太 更 最
    吗 呢 吧 啊 呀 嘛 哦 嗯 这 那 这个 那个 这些 那些 这里 那里 一个 一下 一些 什么 怎么
    怎样 多少 哪 哪个 哪些 哪里 为什么 因为 所以 但是 不过 如果 然后 而且 已经 现在
    时候 之前 之后 以前 以后 上次 下次 这次 那次 开始 最近 当时 其实 应该 觉得 知道
    可能 需要 可以 没有 不是 还是 就是 有点 一直 比较 非常 特别 一起 这样 那样 怎么样
    是不是 有没有 能不能 要不要""".split()  # noqa: SIM905
)


def tokens(text: str) -> list[str]:
    return [match.group(0) for match in _TOKEN.finditer(text)]


def is_cjk(token: str) -> bool:
    return bool(_CJK.fullmatch(token))


def cjk_bigrams(text: str) -> list[str]:
    """把连续中文串切成二元组；单字串原样保留。"""
    grams: list[str] = []
    for token in tokens(text):
        if not is_cjk(token):
            continue
        if len(token) == 1:
            grams.append(token)
            continue
        grams.extend(token[index : index + 2] for index in range(len(token) - 1))
    return grams


def cjk_trigrams(text: str) -> list[str]:
    grams: list[str] = []
    for token in tokens(text):
        if not is_cjk(token) or len(token) < 3:
            continue
        grams.extend(token[index : index + 3] for index in range(len(token) - 2))
    return grams


def expand_for_word_index(text: str) -> str:
    """原文 + 中文二元组，写入词索引列。"""
    grams = cjk_bigrams(text)
    if not grams:
        return text
    return f"{text}\n{' '.join(dict.fromkeys(grams))}"


def strip_option_label(text: str) -> str:
    return OPTION_LABEL.sub("", text, count=1).strip()


def chinese_terms(text: str, *, limit: int = MAX_TERMS) -> list[str]:
    """中文检索词：jieba 词 + 字符二元组。

    二元组兜住词典外的人名/地名，jieba 词兜住「阿拉斯加」「寄养」这类多字词——
    unicode61 会把连续中文整段当一个 token，只有把它们拆开写进索引才搜得到。
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip().lower()
        if len(cleaned) < 2 or cleaned in seen or not is_cjk(cleaned):
            return
        seen.add(cleaned)
        terms.append(cleaned)

    if _JIEBA_READY:
        for token in jieba.cut(text, HMM=False):
            add(token)
            if len(terms) >= limit:
                return terms[:limit]
    for token in tokens(text):
        if not is_cjk(token):
            continue
        if len(token) == 1:
            continue
        for index in range(len(token) - 1):
            add(token[index : index + 2])
            if len(terms) >= limit:
                return terms[:limit]
    return terms


def content_terms(text: str, *, limit: int = 24) -> list[str]:
    """问题里的实词，只用于弃答判定。

    中文保留单字名词（狗/猫/车/书），过滤功能词；英文取 ≥2 字母的单词并去停用词。
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip().lower()
        if not cleaned or cleaned in seen:
            return
        if cleaned.isascii():
            if len(cleaned) < 2 or cleaned in _STOPWORDS or cleaned.isdigit():
                return
        elif is_cjk(cleaned):
            if cleaned in _ZH_STOPWORDS:
                return
        else:
            return
        seen.add(cleaned)
        terms.append(cleaned)

    if _JIEBA_READY:
        for token in jieba.cut(text, HMM=False):
            if is_cjk(token):
                add(token)
    for token in tokens(text):
        if token.isascii():
            add(token)
    return terms[:limit]


# 名词/实体类词性：弃答判定只看「这条记忆里提没提过这些东西」
_NOUN_FLAGS: Final[frozenset[str]] = frozenset({"n", "nr", "ns", "nt", "nz", "vn", "j", "l", "i"})


def salient_terms(text: str, *, limit: int = 20) -> list[str]:
    """问题里的名词/实体词；一个名词都没有时退回实词。

    弃答判定关心的是「有没有关于这个东西的记忆」，所以名词比动词更有判别力。
    """
    terms: list[str] = []
    seen: set[str] = set()
    if _JIEBA_READY:
        for word, flag in pseg.cut(text):
            cleaned = word.strip().lower()
            if flag in _NOUN_FLAGS and len(cleaned) >= 2 and cleaned not in seen:
                seen.add(cleaned)
                terms.append(cleaned)
                if len(terms) >= limit:
                    return terms
    for token in tokens(text):
        if not token.isascii():
            continue
        cleaned = token.lower()
        if len(cleaned) >= 4 and cleaned not in _STOPWORDS and cleaned not in seen:
            seen.add(cleaned)
            terms.append(cleaned)
            if len(terms) >= limit:
                return terms
    return terms if len(terms) >= 2 else content_terms(text, limit=limit)


def mixed_terms(text: str, *, limit: int = 512) -> list[str]:
    """中英混排的实词：jieba/中文二元组 + 英文单词。

    memorysum 的实测（AGENTS #12）：走 jieba 的 Chinese FTS 对英文同样有效，
    召回甚至优于纯英文 FTS——因为停用词过滤 + 长词优先，噪声更小。
    """
    terms = chinese_terms(text, limit=limit)
    seen = set(terms)
    for token in tokens(text):
        if not token.isascii() or len(token) < 2:
            continue
        lowered = token.lower()
        if lowered in _STOPWORDS or lowered.isdigit() or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(lowered)
        if len(terms) >= limit:
            break
    return terms


def zh_index_text(text: str) -> str:
    """写进中文 FTS 列的形态：中英实词空格分隔。"""
    return " ".join(mixed_terms(text))


def camel_expansions(text: str) -> str:
    """`getUserById` → `get user by id`，让按词拆写的查询也能命中代码标识符。"""
    parts: list[str] = []
    for token in tokens(text):
        if not token.isascii() or token.islower() or token.isupper():
            continue
        pieces = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+", token)
        if len(pieces) > 1:
            parts.append(" ".join(piece.lower() for piece in pieces))
    return " ".join(dict.fromkeys(parts))


def query_terms(text: str, *, limit: int = MAX_TERMS) -> list[str]:
    """查询词：去停用词、保留原始词与中文二元组，去重保序。"""
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens(text):
        candidates = [token] if not is_cjk(token) else cjk_bigrams(token)
        for candidate in candidates:
            lowered = candidate.lower()
            if not is_cjk(candidate) and (lowered in _STOPWORDS or len(candidate) < 2):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            terms.append(candidate)
            if len(terms) >= limit:
                return terms
    return terms


def match_expression(terms: list[str]) -> str | None:
    """把词列表拼成 FTS5 的 MATCH 表达式，逐词加引号避免语法字符干扰。"""
    if not terms:
        return None
    quoted = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
    return " OR ".join(quoted)


def word_query(text: str) -> str | None:
    return match_expression(query_terms(text))


def zh_query(text: str, *, limit: int = MAX_TERMS) -> str | None:
    return match_expression(mixed_terms(text, limit=limit))


def trigram_query(text: str, *, limit: int = MAX_TERMS) -> str | None:
    """三元组分词器只认长度 ≥3 的子串，中文则补滑窗三元组。"""
    candidates: list[str] = []
    for token in tokens(text):
        if is_cjk(token):
            candidates.extend(cjk_trigrams(token) if len(token) >= 3 else [])
        elif len(token) >= 3:
            candidates.append(token)
    cleaned = [item for item in dict.fromkeys(candidates)][:limit]
    return match_expression(cleaned)


def clip_middle(text: str, limit: int) -> str:
    """超长输入保留头尾：开头是问题、结尾常带选项/约束，中间才是可丢的部分。"""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n…\n{text[-tail:]}"


def truncate(text: str, limit: int) -> str:
    """按字符截断，尽量不切断代理对。"""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit]
