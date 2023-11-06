import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_hidden_layers: int = 1,
            dropout_rate: float = 0.0,
            bias: bool = True
    ) -> None:
        super().__init__()

        assert num_hidden_layers >= 1

        self.layers = nn.ModuleList([])

        dims = [input_dim] + [hidden_dim] * num_hidden_layers
        for in_dim, out_dim in zip(dims, dims[1:]):
            self.layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
