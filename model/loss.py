import torch


class Loss(torch.nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.crit = torch.nn.BCELoss()

    def forward(self, predictions, truth):
        loss = self.crit(predictions, truth)
        return loss
