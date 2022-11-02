from typing import Any, Dict, Union
from collections import defaultdict


def handle_fallback_per_split(input: Union[Any, Dict[str, Any]]):
    if input is None:
        result = defaultdict(lambda: (lambda x: x))
    elif not isinstance(input, dict):
        result = defaultdict(lambda: input)
    else:
        result = dict(input)
        fallbacks = {
            "test": "validation",
            "validation": "train",
        }
        for split in fallbacks.keys():
            if split in input:
                continue

            fallback_from = split
            while fallback_from is not None:
                fallback_to = fallbacks.get(fallback_from)
                if fallback_to is not None and fallback_to in input:
                    result[split] = input[fallback_to]
                    break
                else:
                    fallback_from = fallback_to
    return result
