import re
import weave


@weave.op()
def parse_math_answer(completion: str) -> str:
    return re.search(r'\\boxed{(.*)}', completion).group(1)


@weave.op()
def parse_code_answer(completion: str) -> str:
    return re.search(r'```python\n(.*)\n```', completion).group(1)

