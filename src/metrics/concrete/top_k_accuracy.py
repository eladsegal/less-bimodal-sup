from typing import List, Union
import torch
from src.metrics import Metric


class TopKAccuracy(Metric):
    def __init__(self, k: Union[int, List[int]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ks = k if isinstance(k, list) else [k]
        self._max_k = max(self._ks)

        for k in self._ks:
            self.add_state(f"top_{k}_accuracy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        labels = labels.unsqueeze(-1)
        top_k_indices = torch.topk(logits, k=self._max_k, dim=-1).indices
        for k in self._ks:
            setattr(
                self,
                f"top_{k}_accuracy",
                getattr(self, f"top_{k}_accuracy") + torch.sum(top_k_indices[:, :k] == labels),
            )

        self.count += len(logits)

    def compute(self):
        results = {}
        for k in self._ks:
            count = self.count.item()
            results[f"top_{k}_accuracy"] = getattr(self, f"top_{k}_accuracy").item() / count if count > 0 else 0
        return results
