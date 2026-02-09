from torch.nn.functional import dropout
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from math import pi, log
from functools import wraps
from typing import *
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
class moleculeGAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=3, num_layers=4, dropout=0.5, layer_norm=True):
        super(moleculeGAT, self).__init__()
        self.molecule_atomencoder = nn.Embedding(512 * 9 + 1, in_channels, padding_idx=0)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gat_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(in_channels, out_channels, heads=num_heads, concat=False))  # concat=False
            else:
                self.gat_layers.append(
                    GATConv(out_channels, out_channels, heads=num_heads, concat=False))  # concat=False

        self.dropout_layer = nn.Dropout(p=dropout)
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(num_layers)])  # out_channels
        else:
            self.layer_norms = None

        self.pool = global_mean_pool

    def forward(self, drug_batch):
        x, edge_index, batch = self.molecule_atomencoder(drug_batch.x.long()), drug_batch.edge_index, drug_batch.batch
        x = torch.mean(x, dim=-2)  # Aggregate features for drug nodes
        layer_outputs = []
        xx = self.pool(x, batch)
        layer_outputs.append(xx.unsqueeze(1))
        for i in range(self.num_layers):
            x = self.gat_layers[i](x, edge_index)
            x = self.dropout_layer(x)
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            xx = self.pool(x, batch)
            layer_outputs.append(xx.unsqueeze(1))  
        out = torch.cat(layer_outputs, dim=1) 
        out = out.mean(dim=1)
        return out


class proteinGAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=3, num_layers=4, dropout=0.5, layer_norm=True):
        super(proteinGAT, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gat_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(in_channels, out_channels, heads=num_heads, concat=False))  # concat=False
            else:
                self.gat_layers.append(
                    GATConv(out_channels, out_channels, heads=num_heads, concat=False))  # concat=False

        self.dropout_layer = nn.Dropout(p=dropout)
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(num_layers)]) 
        else:
            self.layer_norms = None

        self.pool = global_mean_pool

    def forward(self, protein_batch):
        x, edge_index, batch = protein_batch.x, protein_batch.edge_index, protein_batch.batch
        layer_outputs = []
        xx = self.pool(x, batch)
        layer_outputs.append(xx.unsqueeze(1))
        for i in range(self.num_layers):
            x = self.gat_layers[i](x, edge_index)
            x = self.dropout_layer(x)
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            xx = self.pool(x, batch)
            layer_outputs.append(xx.unsqueeze(1)) 
        out = torch.cat(layer_outputs, dim=1) 
        out = out.mean(dim=1)
        return out
