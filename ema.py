class EMA:
    
    def __init__(self, model, decay=0.999):

        self.decay = decay
        self.model = model
        self.shadow = {}

        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):

        for name, param in self.model.named_parameters():

            self.shadow[name] = (
                self.decay * self.shadow[name]
                + (1 - self.decay) * param.data
            )

    def apply_shadow(self):

        for name, param in self.model.named_parameters():
            param.data = self.shadow[name]