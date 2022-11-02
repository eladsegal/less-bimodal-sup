from typing import Any
from dataclasses import dataclass


@dataclass
class Result:
    log_name: str
    value: Any
