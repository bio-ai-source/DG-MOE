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
from .GAT import moleculeGAT, proteinGAT
from .net import MoE_Fusion
class DTI(nn.Module):
    def __init__(
            self,
            drug_in_channels,
            drug_out_channels,
            protein_in_channels,
            protein_out_channels,
            num_heads,
            num_layers,
            dropout,
            layer_output,
            p_cum
    ):
        super(DTI, self).__init__()
        self.drug_gat = moleculeGAT(
            in_channels=drug_in_channels,
            out_channels=drug_out_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout = dropout
        )
        self.protein_gat = proteinGAT(
            in_channels=protein_in_channels,
            out_channels=protein_out_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        self.fusion = MoE_Fusion(
            input_dims=[1024, 1280, 3072, 3072, 512],
            num_experts=6,
            top_k=2,
            expert_dim=512,
            t=p_cum
        )
        # mlp=3
        self.layer_output = layer_output
        self.W_out = nn.ModuleList([nn.Linear(
            512, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256)])
        self.W_interaction = nn.Linear(256, 2)
    def forward(self, drug_batch, protein_batch):
        zero_tensor = torch.zeros(len(drug_batch.label), 512).to('cuda')
        drug_batch =  drug_batch.to('cuda')
        protein_batch = protein_batch.to('cuda')
        protein_graph = self.protein_gat(protein_batch)
        drug_graph = self.drug_gat(drug_batch)
        batch = drug_graph.size(0)
        drug_embedding = drug_batch.dt.view(batch, 3072)
        protein_embedding = protein_batch.pt.view(batch, 3072)
        gene_embedding = protein_batch.gt.view(batch, 512)
        tensors = [drug_graph, protein_graph, drug_embedding, protein_embedding, gene_embedding]
        x, probs = self.fusion(*tensors)
        for j in range(self.layer_output):
            x = torch.tanh(self.W_out[j](x))
        predicted = self.W_interaction(x)
        return probs, predicted
