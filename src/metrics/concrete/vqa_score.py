from typing import List
from collections import Counter

import torch
from src.metrics import Metric


class VqaScore(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: List[str], answers_list: List[List[str]]):
        score = 0
        for prediction, answers in zip(predictions, answers_list):
            answer_count = Counter(answers)
            score += get_score(answer_count[prediction])

        self.score += score
        self.count += len(predictions)

    def compute(self):
        count = self.count.item()
        return self.score.item() / count if count > 0 else count


class VqaScoreByLabels(VqaScore):
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        # TODO: use torchmetrics to_onehot?
        predicted_label_indices = torch.argmax(logits, -1)
        one_hots = torch.zeros(*targets.size()).to(targets)
        one_hots.scatter_(1, predicted_label_indices.view(-1, 1), 1)
        scores = one_hots * targets

        self.score += scores.sum()
        self.count += len(logits)


def get_score(occurences: int):
    # This is the score when:
    # - the accuracy for a set of annotations is min(1, # of agreements / 3)
    # - averaging the accuracy of 10 choose 9 combinations of the human annotations
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0
