#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class CNNDQN(ModelBase):
    def __init__(self, cnn_parameters, layers, activations) -> None:
        super().__init__()
        print(cnn_parameters)
        if type(cnn_parameters) is dict:
            cnn_parameters = json.loads(json.dumps(cnn_parameters))
        self.conv_dims = cnn_parameters.conv_dims
        self.conv_height_kernels = cnn_parameters.conv_height_kernels
        self.conv_width_kernels = cnn_parameters.conv_width_kernels

        self.conv_layers: nn.ModuleList = nn.ModuleList()
        self.pool_layers: nn.ModuleList = nn.ModuleList()

        for i, _ in enumerate(self.conv_dims[1:]):
            self.conv_layers.append(
                nn.Conv2d(
                    self.conv_dims[i],
                    self.conv_dims[i + 1],
                    kernel_size=(
                        self.conv_height_kernels[i],
                        self.conv_width_kernels[i],
                    ),
                )
            )
            nn.init.kaiming_normal_(self.conv_layers[i].weight)
            if cnn_parameters.pool_types[i] == "max":
                self.pool_layers.append(
                    nn.MaxPool2d(kernel_size=cnn_parameters.pool_kernels_strides[i])
                )
            else:
                assert False, "Unknown pooling type".format(layers)

        input_size = (
            cnn_parameters.num_input_channels,
            cnn_parameters.input_height,
            cnn_parameters.input_width,
        )
        conv_out = self.conv_forward(torch.ones(1, *input_size))
        self.fc_input_dim = int(np.prod(conv_out.size()[1:]))
        layers[0] = self.fc_input_dim
        self.feed_forward = FullyConnectedNetwork(layers, activations)

    def conv_forward(self, input) -> torch.FloatTensor:
        x = input
        for i, _ in enumerate(self.conv_layers):
            x = F.relu(self.conv_layers[i](x))
            x = self.pool_layers[i](x)
        return x

    def forward(self, input) -> torch.FloatTensor:
        """ Forward pass for generic convnet DNNs. Assumes activation names
        are valid pytorch activation names.
        :param input image tensor
        """
        x = self.conv_forward(input)
        x = x.view(-1, self.fc_input_dim)
        return self.feed_forward.forward(x)

    
class FullyConnectedDQN(ModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
        use_batch_norm=False,
        dropout_ratio=0.0,
    ):
        super().__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        assert action_dim > 0, "action_dim must be > 0, got {}".format(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
        )
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [action_dim],
            activations + ["linear"],
            use_batch_norm=use_batch_norm,
            dropout_ratio=dropout_ratio,
        )

    def get_distributed_data_parallel_model(self):
        return _DistributedDataParallelFullyConnectedDQN(self)

    def input_prototype(self):
        return rlt.PreprocessedState.from_tensor(torch.randn(1, self.state_dim))

    def forward(self, input: rlt.PreprocessedState):
        q_values = self.fc(input.state.float_features)
        return rlt.AllActionQValues(q_values=q_values)


class _DistributedDataParallelFullyConnectedDQN(ModelBase):
    def __init__(self, fc_dqn):
        super().__init__()
        self.state_dim = fc_dqn.state_dim
        self.action_dim = fc_dqn.action_dim
        current_device = torch.cuda.current_device()
        self.data_parallel = DistributedDataParallel(
            fc_dqn.fc, device_ids=[current_device], output_device=current_device
        )
        self.fc_dqn = fc_dqn

    def input_prototype(self):
        return rlt.PreprocessedState.from_tensor(torch.randn(1, self.state_dim))

    def cpu_model(self):
        return self.fc_dqn.cpu_model()

    def forward(self, input):
        q_values = self.data_parallel(input.state.float_features)
        return rlt.AllActionQValues(q_values=q_values)
