from core.models.mlp import MLP

class Model1300VA(MLP):
    def __init__(self, layer_sizes):
        super().__init__(
            layer_sizes = layer_sizes
        )
