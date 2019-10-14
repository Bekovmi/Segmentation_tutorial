from catalyst.dl import SupervisedRunner as Runner


class ModelRunner(Runner):
    def __init__(self, model=None, device=None):
        super().__init__(
            model=model, device=device, input_key="image", output_key="logits"
        )
