import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path

from memorysum_vl.core.config import Settings


class AuthConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ApiToken:
    secret: str

    def matches(self, candidate: str) -> bool:
        return hmac.compare_digest(_digest(self.secret), _digest(candidate))


def load_api_token(settings: Settings) -> ApiToken:
    secret = _resolve_secret(settings)
    if not secret:
        raise AuthConfigurationError(
            "MEMORYSUM_AUTH_TOKEN_FILE or MEMORYSUM_AUTH_TOKEN must provide a bearer token"
        )
    return ApiToken(secret=secret)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _resolve_secret(settings: Settings) -> str:
    if settings.auth_token_file is not None:
        return _read_token_file(settings.auth_token_file)
    if settings.auth_token is not None:
        return settings.auth_token.get_secret_value().strip()
    return ""


def _read_token_file(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuthConfigurationError(f"cannot read bearer token file {path}: {exc}") from exc
    for line in content.splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("#"):
            return candidate
    raise AuthConfigurationError(f"bearer token file {path} does not contain a token")
