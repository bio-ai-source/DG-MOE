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
        # tensors = [drug_graph, protein_graph, drug_embedding, protein_embedding, gene_embedding, zero_tensor]
        tensors = [drug_graph, protein_graph, drug_embedding, protein_embedding, gene_embedding]

        # print('drug_graph: ', drug_graph.shape)
        # print('protein_graph: ', protein_graph.shape)
        # print('drug_embedding: ', drug_embedding.shape)
        # print('protein_embedding: ', protein_embedding.shape)
        # print('gene_embedding: ', gene_embedding.shape)


        x, probs = self.fusion(*tensors)
        # print(f'x shape : {x.shape}')
        # print(f'probs: ', probs.shape)
        # print(f'topk_indices : {topk_indices}')
        #
        # print(f'topk_indices : {topk_indices.shape}')

        for j in range(self.layer_output-1):
            x = torch.tanh(self.W_out[j](x))
        predicted = self.W_interaction(x)
        # print(f'predicted shape : {predicted.shape}')
        return predicted, probs, probs

if __name__ == "__main__":
    batch_size = 3
    # Generate protein data (2 samples)


    protein_pt = torch.randn(3072)  # Protein feature matrix of shape [241, 1280]
    protein_gt = torch.randn(512)
    protein_x1 = torch.randn(241, 1280)  # Protein feature matrix of shape [241, 1280]
    protein_edge_index1 = torch.randint(0, 241, (2, 2436))  # Protein edge indices of shape [2, 2436]
    protein_edge_index2 = torch.randint(0, 241, (2, 2436))  # Protein edge indices of shape [2, 2436]
    protein_data1 = Data(x=protein_x1, edge_index=protein_edge_index1, pt = protein_pt, gt = protein_gt)
    protein_data2 = Data(x=protein_x1, edge_index=protein_edge_index2, pt = protein_pt, gt = protein_gt)
    protein_data3 = Data(x=protein_x1, edge_index=protein_edge_index2, pt = protein_pt, gt = protein_gt)

    embedding_size = 512 * 9  # Valid indices range from 0 to 512*9-1
    drug_x1 = torch.randint(0, embedding_size, (20, 9), dtype=torch.long)  # Drug feature matrix with integer indices
    drug_edge_index1 = torch.randint(0, 20, (2, 44))  # Drug edge indices of shape [2, 44]
    drug_data1 = Data(x=drug_x1, edge_index=drug_edge_index1, dt = protein_pt)  # Drug feature matrix with valid indices
    drug_data2 = Data(x=drug_x1, edge_index=drug_edge_index1, dt = protein_pt)  # Drug feature matrix with valid indices

    protein_data_list = [protein_data1, protein_data2, protein_data3]
    drug_data_list = [drug_data1, drug_data2, drug_data2]

    protein_loader = DataLoader(protein_data_list, batch_size=batch_size)
    drug_loader = DataLoader(drug_data_list, batch_size=batch_size)

    # Example of iterating over the DataLoader for proteins
    for drug_batch, protein_batch in zip(drug_loader, protein_loader):
        config = {
            'drug_in_channels': 1024,
            'drug_out_channels': 1024,
            'protein_in_channels': 1280,
            'protein_out_channels': 1280,
            'num_heads': 3,
            'num_layers': 4,
            'dropout': 0.5,
            'layer_output': 3
        }
        model = DTI(**config)

        # model = DTI(drug_in_channels=1024,
        #             drug_out_channels=1024,
        #             protein_in_channels=1280,
        #             protein_out_channels=1280,
        #             num_heads=3,
        #             num_layers=4,
        #             dropout=0.5
        #             )
        x1, x2, result = model(drug_batch, protein_batch)
        print(f'x1 shape: {x1.shape}')
        print(f'x2 shape: {x2.shape}')
        print(f'result: {result}')
        print(f'result shape: {result.shape}')

    # 对于1个drug和protein, shape类似如下:
    # protein x shape: torch.Size([241, 1280])
    # protein edge_index shape: torch.Size([2, 2436])
    # drug node_feat shape: torch.Size([20, 9, 512])
    # drug edge_index shape: torch.Size([2, 44])
