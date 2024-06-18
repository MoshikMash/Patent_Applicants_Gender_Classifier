import torch
import torch.nn as nn

class CustomPooler(nn.Module):
    def __init__(self, input_size):
        super(CustomPooler, self).__init__()
        self.fc_layers = nn.ModuleList()
        # Calculate the number of layers required
        num_layers = int(torch.ceil(torch.log2(torch.tensor(input_size / 2))))
        in_features = input_size
        for _ in range(num_layers):
            out_features = max(in_features // 2, 1)  # Ensure at least 1 neuron
            self.fc_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        # Final linear layer with output size 1
        self.output_layer = nn.Linear(in_features, 1)

    def forward(self, x):
        for layer in self.fc_layers:
            x = torch.relu(layer(x))
        # Apply sigmoid to the output of the final linear layer
        x = self.output_layer(x)
        x = torch.squeeze(x, dim=-1)  # Remove the last dimension (which should be 1)
        return x
