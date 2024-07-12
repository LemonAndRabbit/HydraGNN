import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .Base import Base

# For compatibility with function requires model.module 
class ModuleWrapper:
    def __init__(self, model):
        self.module = model

# For compatibility with function requires data in torch_geometric format
class DataWrapper:
    def __init__(self, pipe_out, label):
        self.y = label[0]
        self.y_loc = label[1]
        self.batch = pipe_out[4]

class ConvPipe(nn.Module):
    def __init__(self, conv_instance):
        super().__init__()
        self.conv = conv_instance

    def forward(self, input_tuple):
        c, pos = self.conv(
            x=input_tuple[0], 
            pos=input_tuple[1], 
            edge_index=input_tuple[2],
        )
        return (c, pos,) + input_tuple[2:]

class ConvPipeEdgeAttr(ConvPipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_tuple):
        c, pos = self.conv(
            x=input_tuple[0], 
            pos=input_tuple[1], 
            edge_index=input_tuple[2],
            edge_attr=input_tuple[3],
        )
        return (c, pos,) + input_tuple[2:]

class FeatPipe(nn.Module):
    def __init__(self, feat_instance, activation_function):
        super().__init__()
        self.feat_layer = feat_instance
        self.activation_function = activation_function

    def forward(self, input_tuple):
        x = self.activation_function(self.feat_layer(input_tuple[0]))
        return (x,) + input_tuple[1:]

class MeanPipe(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_tuple):
        x_graph = input_tuple[0].mean(dim=0, keepdim=True)
        return input_tuple[:5] + (x_graph, ) + input_tuple[6:]

class GlobalMeanPipe(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_tuple):
        x_graph = global_mean_pool(input_tuple[0], input_tuple[4])
        return input_tuple[:5] + (x_graph, ) + input_tuple[6:]

class HeadPipe(nn.Module):
    def __init__(
            self, 
            head_dims, 
            heads_NN, 
            head_type, 
            graph_shared, 
            node_NN_type, 
            use_edge_attr, 
            activation_function
        ):
        super().__init__()
        self.head_dims = head_dims
        self.heads_NN = heads_NN
        self.head_type = head_type
        self.graph_shared = graph_shared
        self.node_NN_type = node_NN_type
        self.use_edge_attr = use_edge_attr
        self.activation_function = activation_function
    def forward(self, input_tuple):
        outputs = []
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                x_graph_head = self.graph_shared(input_tuple[5])
                outputs.append(headloc(x_graph_head))
            else:
                if self.node_NN_type == "conv":
                    for conv, batch_norm in zip(headloc[0::2], headloc[1::2]):
                        if self.use_edge_attr:
                            c, pos = conv(x=input_tuple[0], pos=input_tuple[1], edge_index=input_tuple[2], edge_attr=input_tuple[3])
                        else:
                            c, pos = conv(x=input_tuple[0], pos=input_tuple[1], edge_index=input_tuple[2])
                        c = batch_norm(c)
                        x = self.activation_function(c)
                    x_node = x
                else:
                    x_node = headloc(x=input_tuple[0], batch=input_tuple[4])
                outputs.append(x_node)
        return input_tuple[:6] + tuple(outputs)
        

# Class to wrap the original base model into sequential layers
class BasePipe(Base):
    def __init__(self, base_instance, input_w_batch):
        self.__dict__.update(base_instance.__dict__)
        self.input_w_batch = input_w_batch
        if not hasattr(self, "node_NN_type"):
            self.node_NN_type = None
    
    def to_layers(self):
        # input_tuples: (x, pos, edge_index, edge_attr, batch, x_graph, **outputs)
        layers = []
        
        ### encoder part ####
        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if self.use_edge_attr:
                layers.append(ConvPipeEdgeAttr(conv))
            else:
                layers.append(ConvPipe(conv))
            layers.append(FeatPipe(feat_layer, self.activation_function))
        #### multi-head decoder part####
        if not self.input_w_batch:
            layers.append(MeanPipe())
        else:
            layers.append(GlobalMeanPipe())
        layers.append(HeadPipe(
            self.head_dims,
            self.heads_NN,
            self.head_type,
            self.graph_shared,
            self.node_NN_type,
            self.use_edge_attr,
            self.activation_function
        )) # TODO: when the number of heads is large, we may need to slice the last layer into multiple layers

        return layers

    def get_pipeloss(self, get_head_indices_func):
        moduled_model = ModuleWrapper(model = self)
        def pipeloss(pipe_out, label):
            # pipe_out: (x, pos, edge_index, edge_attr, batch, x_graph, **outputs)
            # label: (y, y_loc)
            pred = pipe_out[6:]
            y = label[0]
            head_index = get_head_indices_func(moduled_model, DataWrapper(pipe_out, label))

            loss, tasks_loss = self.loss(pred, y, head_index)
            return loss

        return pipeloss