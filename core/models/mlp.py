from core.layers.hidden import Hidden
from core.layers.activation import ReLU
from core.layers.activation import Linear
from core.layers.output import Output

class MLP:
    def __init__(self, layer_sizes):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes minimal harus memiliki 2 elemen.")
        if layer_sizes[-1] != 1:
            raise ValueError("Output layer harus berukuran 1 untuk regresi.")

        self.layer_sizes = list(layer_sizes)
        self.layers = []

        input_size = self.layer_sizes[0]

        for hidden_size in self.layer_sizes[1:-1]:
            self.layers.append(Hidden(
                input_size, hidden_size
            ))

            self.layers.append(ReLU())

            input_size = hidden_size

        self.layers.append(
            Output(input_size)
        )

        self.layers.append(
            Linear()
        )


    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
            # if isinstance(layer, Hidden):
            #     print('output preactivation', output)

        return output

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient

    def get_trainable_layers(self):
        trainable = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                trainable.append(layer)
        return trainable
