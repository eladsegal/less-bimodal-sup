from typing import Any, Mapping


class KeyMapping(dict):
    def apply(self, key: str, batch: Mapping[str, Any]):
        targets = self[key]
        if isinstance(targets, Mapping):
            return {targets[target]: batch[target] for target in targets if target in batch}
        else:
            return {target: batch[target] for target in targets if target in batch}
