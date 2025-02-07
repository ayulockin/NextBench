import weave

from .cacher import cacher, DiskCacheBackend
from .core_types import RequestResult
from .parsers import parse_math_answer, parse_code_answer


@weave.op()
def preprocess_example(example):
    return {
        "prompt": example["question"]
    }

__all__ = [
    "RequestResult",
    "cacher",
    "DiskCacheBackend",
    "parse_math_answer",
    "parse_code_answer",
    "preprocess_example",
]
