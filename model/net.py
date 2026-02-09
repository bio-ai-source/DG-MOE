import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import sort_edge_index


class ModalCompressor(nn.Module):
    def __init__(self, input_dim, unified_dim=256):
        super().__init__()
        self.input_dim = int(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, 2 * unified_dim),
            nn.GELU(),
            nn.Linear(2 * unified_dim, unified_dim)
        )
        self.norm = nn.LayerNorm(unified_dim)
        self.residual = nn.Linear(self.input_dim, unified_dim) if self.input_dim >= unified_dim \
            else lambda x: F.pad(x, (0, unified_dim - self.input_dim))

    def forward(self, x):
        return self.norm(self.proj(x) + self.residual(x))
        
class GlobalGating(nn.Module):
    def __init__(self, feat_dim: int = 256, num_experts: int = 6, mlp_hidden: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_experts = num_experts
        self.mlp_g = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, num_experts)  
        )

    def forward(self, context: torch.Tensor):
        B, M, D = context.shape
        assert D == self.feat_dim, f"feat_dim mismatch: got D={D}, expected {self.feat_dim}"
        assert M == 5, f"expected 5 modalities, got {M}"

        C_avg = context.mean(dim=1)

        # 全局 gating logits: [B, 6]
        g_global = self.mlp_g(C_avg)
        return g_global
        
class LocalGating(nn.Module):
    def __init__(self, feat_dim: int = 256, pool_out: int = 8, mlp_hidden: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.pool_out = pool_out
        self.mode_pairs = [(0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)]
        self.register_buffer("pairs_idx",
                             torch.tensor(self.mode_pairs, dtype=torch.long))
        self.mlp_L = nn.Sequential(
            nn.Linear(pool_out, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1)
        )


    def forward(self, context: torch.Tensor):
        B, M, D = context.shape
        assert D == self.feat_dim, f"feat_dim mismatch: got D={D}, expected {self.feat_dim}"
        x = context.reshape(B * M, 1, D)                     
        salient = F.adaptive_max_pool1d(x, self.pool_out)   
        salient = salient.squeeze(1)                         
        c = self.mlp_L(salient).view(B, M)                   
        i_idx = self.pairs_idx[:, 0]                        
        j_idx = self.pairs_idx[:, 1]                      
        g_local = c[:, i_idx] + c[:, j_idx]                  
        ci = c[:, i_idx]
        cj = c[:, j_idx]
        weight = torch.stack([ci, cj], dim=-1)
        return g_local, weight


class DualGating(nn.Module):
    def __init__(self, t):
        super().__init__()

        self.global_function = GlobalGating()
        self.local_function = LocalGating()
        self.temperature = nn.Parameter(torch.tensor(t))

        self.w_alpha = nn.Parameter(torch.tensor(-0.0 / 2.0))
        self.w_beta = nn.Parameter(torch.tensor(-0.0 / 2.0))

        self.W_gate = nn.Parameter(torch.randn(6, 1))  
        self.b_gate = nn.Parameter(torch.randn(1)) 

    def forward(self, context):
        g_global = self.global_function(context)
        g_local, weight = self.local_function(context)

        lambda_x = torch.sigmoid(torch.matmul(g_global, self.W_gate) + self.b_gate)  # [B, 1]

        gfinal = lambda_x * g_global + (1 - lambda_x) * g_local

        gfinal = F.softmax(gfinal / torch.exp(self.temperature), dim=-1)

        return gfinal


class ModalAwareExpert(nn.Module):
    def __init__(self, input_dim1, input_dim2, expert_dim=512):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(input_dim1, expert_dim // 2), nn.GELU(), nn.LayerNorm(expert_dim // 2))
        self.net2 = nn.Sequential(nn.Linear(input_dim2, expert_dim // 2), nn.GELU(), nn.LayerNorm(expert_dim // 2))
        self.fusion = nn.Sequential(nn.Linear(expert_dim, expert_dim), nn.ELU(), nn.LayerNorm(expert_dim))

    def forward(self, x1, x2):
        return self.fusion(torch.cat([self.net1(x1), self.net2(x2)], dim=-1))


class MoE_Fusion(nn.Module):
    def __init__(self, input_dims, num_experts=6, top_k=2, expert_dim=512,t=0.5):
        super().__init__()
        self.top_k = 6
        self.compressors = nn.ModuleList([ModalCompressor(dim) for dim in input_dims])
        self.mode_pairs = [(0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)]
        self.experts = nn.ModuleList([
            ModalAwareExpert(input_dims[i], input_dims[j], expert_dim)
            for i, j in self.mode_pairs
        ])
        self.gating = DualGating(t)
    def forward(self, *modes):
        compressed = [comp(m) for comp, m in zip(self.compressors, modes)]
        z_stack = torch.stack(compressed, dim=1)
        gate = self.gating(z_stack)
        expert_outputs = []
        for expert, (i, j) in zip(self.experts, self.mode_pairs):
            expert_outputs.append(expert(modes[i], modes[j]))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, D]
        y = (expert_outputs * gate.unsqueeze(-1)).sum(dim=1)
        # print(y.shape)
        # print(y)
        return y, gate
        return self.refinement(fused), gate

