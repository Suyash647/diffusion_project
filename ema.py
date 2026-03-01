import copy
import torch


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.shadow.parameters(), model.parameters()):
                ema_param.data = (
                    self.decay * ema_param.data +
                    (1 - self.decay) * param.data
                )

    def state_dict(self):
        return self.shadow.state_dict()