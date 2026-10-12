"""Provider boundary with safe diagnostic logging and predictable failures."""

import logging
from contextlib import contextmanager

import httpx
from google.genai.errors import APIError

logger = logging.getLogger(__name__)


class ProcessingError(RuntimeError):
    pass


@contextmanager
def provider_call(operation: str):
    try:
        yield
    except (APIError, httpx.HTTPError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        # Do not log raw provider messages, URLs, keys, resumes or notes.
        logger.warning("%s failed: exception=%s status=%s", operation,
                       type(exc).__name__, getattr(exc, "code", None))
        raise ProcessingError(f"{operation} unavailable or returned invalid output") from exc
