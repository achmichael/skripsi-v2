import numpy as np

def clip_gradients(model, clip_value):
    """
        Membatasi setiap elemen gradien bobot dan bias
        pada rentang [-clip_value, clip_value].
        Dipanggil setelah backward propagation dan
        sebelum optimizer melakukan update parameter.
    """

    if clip_value <= 0:
        raise ValueError("Clip value harus lebih besar dari 0")

    for layer in model.layers:
        if not hasattr(layer, "weights"):
            continue

        if hasattr(layer, "dweights") and layer.dweights is not None:
            np.clip(
                layer.dweights,
                -clip_value,
                clip_value,
                out=layer.dweights
            )

        if hasattr(layer, "dbiases") and layer.dbiases is not None:
            np.clip(
                layer.dbiases,
                -clip_value,
                clip_value,
                out=layer.dbiases
            )
