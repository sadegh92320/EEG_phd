import torch

class ComputeMetrics():
    """Compute all the metrics of interest to evaluate performance of network"""
    def __init__(self):
        self.tp = None
        self.tn = None
        self.fn = None
        self.fp = None

    def get_TP_TN_FN_FP(self, prediction, true_labels):
        self.tp = ((prediction == 1) & (true_labels == 1)).sum().item()
        self.tn = ((prediction == 0) & (true_labels == 0)).sum().item()
        self.fn = ((prediction == 0) & (true_labels == 1)).sum().item()
        self.fp = ((prediction == 1) & (true_labels == 0)).sum().item()

        return self

    def get_acc(self, prediction, true_labels):
        return (self.tp + self.tn)/(self.tp + self.tn + self.fn + self.fp)

    def get_precision(self, prediction, true_labels):
        return self.tp/(self.fp + self.tp)

    def get_recall(self, prediction, true_labels):
        return self.tp/(self.fn + self.tp)

    def getF1(self, prediction, true_labels):
        r = self.get_recall(prediction, true_labels)
        p = self.get_precision(prediction, true_labels)
        return (2*p*r)/(p+r)

    def get_ROC(self, prediction, true_labels):
        pass

    def get_confusion_matrix(self, prediction, true_labels):
        pass

class Averager():
    def __init__(self) -> None:
        pass

if __name__ == "__main__":
    metric = ComputeMetrics()
    prediction = torch.tensor([1,0,0,1,0,1,1])
    labels = torch.tensor([0,1,0,1,0,1,1])
    metric.get_TP_TN_FN_FP(prediction, labels)
