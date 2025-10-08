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

        # 修改：设置 GATConv 层的 concat=False
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
        # print(f'drug_batch.x shape: {drug_batch.x.shape}')
        x, edge_index, batch = self.molecule_atomencoder(drug_batch.x.long()), drug_batch.edge_index, drug_batch.batch
        x = torch.mean(x, dim=-2)  # Aggregate features for drug nodes
        # print(f'x shape: {x.shape}')
        # print(f'edge_index shape: {edge_index.shape}')
        # print(f'batch: {batch}')
        # print(f'batch shape: {batch.shape}')

        layer_outputs = []
        xx = self.pool(x, batch)
        layer_outputs.append(xx.unsqueeze(1))
        # 遍历每一层
        for i in range(self.num_layers):
            # 使用GATConv进行图卷积
            x = self.gat_layers[i](x, edge_index)
            x = self.dropout_layer(x)
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            # print(f'xx shape (after GAT layer): {x.shape}')
            xx = self.pool(x, batch)
            # print(f'xx shape (after pooling): {xx.shape}')
            layer_outputs.append(xx.unsqueeze(1))  # 增加维度以便后续拼接
        out = torch.cat(layer_outputs, dim=1)  # [batch_size, num_layers, out_channels]
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

        # 修改：设置 GATConv 层的 concat=False
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

    def forward(self, protein_batch):
        x, edge_index, batch = protein_batch.x, protein_batch.edge_index, protein_batch.batch
        # print(f'x shape: {x.shape}')
        # print(f'edge_index shape: {edge_index.shape}')
        layer_outputs = []
        xx = self.pool(x, batch)
        layer_outputs.append(xx.unsqueeze(1))
        # 遍历每一层
        for i in range(self.num_layers):
            # 使用GATConv进行图卷积
            x = self.gat_layers[i](x, edge_index)
            x = self.dropout_layer(x)
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            # print(f'xx shape (after GAT layer): {x.shape}')
            xx = self.pool(x, batch)
            # print(f'xx shape (after pooling): {xx.shape}')
            layer_outputs.append(xx.unsqueeze(1))  # 增加维度以便后续拼接
        out = torch.cat(layer_outputs, dim=1)  # [batch_size, num_layers, out_channels]
        # print(f'out shape (after concat): {out.shape}')
        out = out.mean(dim=1)
        return out