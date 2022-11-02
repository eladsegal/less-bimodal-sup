from typing import List
import torch
from src.metrics import Metric


class Accuracy(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("accuracy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: List[str], references: List[str]):
        accuracies = [float(prediction == reference) for prediction, reference in zip(predictions, references)]
        self.accuracy += sum(accuracies)
        self.count += len(predictions)

    def compute(self):
        count = self.count.item()
        return self.accuracy.item() / count if count > 0 else 0


class AccuracyByLabels(Accuracy):
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        predictions = logits.argmax(dim=-1)

        labels = labels.reshape(-1)
        predictions = predictions.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]

        self.accuracy += torch.sum(predictions == labels)
        self.count += predictions.numel()
