import torch
import torch.nn as nn


class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, lablel) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"])):
    plt.subplot(1, 2, i + 1)
    plt.plot(x, y)
    plt.title(f"{lablel} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{lablel}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
