import pickle as pk
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ANN_Topology(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(ANN_Topology, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        layers = [nn.Linear(self.input_size, self.hidden_layers[0])]
        layers += [nn.InstanceNorm1d(hidden_layers[0])]
        layers += [nn.SiLU()]

        for i in range(1, len(self.hidden_layers)):
            layers.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
            layers.append(nn.InstanceNorm1d(hidden_layers[i]))
            layers.append(nn.SiLU())

        # Add the final layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, input):
        for idx, layer in enumerate(self.layers):
            #print(f"Before layer {idx}: {input.shape}")
            input = layer(input)
            #print(f"After layer {idx}: {input.shape}")
        return input

    def predict(self, x):
        x = self.forward(x)
        return x



def load_attrs(filepath):
    ''' In this case the filepath must contain the complete directory including the file'''

    with open(filepath, 'rb') as f:
        attrs = pk.load(f)
    return attrs


def load_model_instance(filepath, attrs):
    ''' In this case the filepath must contain the complete directory including the file'''
    input_size = attrs['input_size']
    output_size = attrs['output_size']
    hidden_layers = attrs['hidden_layers']
    model_state = torch.load(filepath, map_location=torch.device('cpu'))

    # In this case it doesn't matter if ANN_topology or PINN_topology is used as they

    model = ANN_Topology(input_size, hidden_layers, output_size)

    # Handle "module." prefix in state_dict keys
    if any(key.startswith("module.") for key in model_state.keys()):
        model_state = {key[len("module."):]: value for key, value in model_state.items()}

    model.load_state_dict(model_state)
    return model
