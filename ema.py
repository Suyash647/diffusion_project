import copy


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model)
        self.decay = decay

    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data = self.decay * ema_p.data + (1 - self.decay) * model_p.data