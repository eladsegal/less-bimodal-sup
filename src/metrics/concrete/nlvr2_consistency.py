from typing import List, Dict
import torch
from src.metrics import Metric


class Nlvr2Consistency(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("consistency", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: List[str], references: List[Dict[str, str]]):
        identifiers = [example["identifier"] for example in references]
        labels = [example["label"] for example in references]
        consistency_dict = {}
        for identifier, prediction, reference in zip(identifiers, predictions, labels):
            anon_label = identifier.split("-")
            anon_label[2] = ""
            anon_label = "-".join(anon_label)
            if not anon_label in consistency_dict:
                consistency_dict[anon_label] = True
            if prediction.lower() != reference.lower():
                consistency_dict[anon_label] = False

        self.count += len(consistency_dict)
        for identifier, consistent in consistency_dict.items():
            if consistent:
                self.consistency += 1

    def compute(self):
        count = self.count.item()
        return self.consistency.item() / count if count > 0 else 0
