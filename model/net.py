import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import sort_edge_index


class ModalCompressor(nn.Module):
    """模态特征压缩模块（含残差连接）"""

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
    """
    Global Uncertainty Assessment (Global Gating)
    - 输入 context: [B, M(=5), D]
    - 输出:
        g_global: [B, 6]  —— 与 6 个专家一一对应的全局 gating logits
    """
    def __init__(self, feat_dim: int = 256, num_experts: int = 6, mlp_hidden: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_experts = num_experts

        # 由 C_avg 直接映射到每个专家的全局 logit（长度=6）
        self.mlp_g = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, num_experts)  # 输出 [B, 6]
        )

    def forward(self, context: torch.Tensor):
        """
        context: [B, 5, D]
        """
        B, M, D = context.shape
        assert D == self.feat_dim, f"feat_dim mismatch: got D={D}, expected {self.feat_dim}"
        assert M == 5, f"expected 5 modalities, got {M}"

        # C_avg = 所有模态特征均值: [B, D]
        C_avg = context.mean(dim=1)

        # 全局 gating logits: [B, 6]
        g_global = self.mlp_g(C_avg)
        return g_global
        
class LocalGating(nn.Module):
    """
    Local Utility Assessment (Local Gating)
    - 输入 context: [B, M, D] (batch, 模态数, 特征维度)
    - 输出:
        g_local: [B, 6]   每个样本 6 个专家的局部门控分数
        c: [B, M]         每个模态的可信度标量
    """
    def __init__(self, feat_dim: int = 256, pool_out: int = 8, mlp_hidden: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.pool_out = pool_out

        # 固定 6 个专家的模态对
        self.mode_pairs = [(0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 4)]
        self.register_buffer("pairs_idx",
                             torch.tensor(self.mode_pairs, dtype=torch.long))

        # MLP_L: salient feature -> scalar credibility
        self.mlp_L = nn.Sequential(
            nn.Linear(pool_out, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1)
        )


    def forward(self, context: torch.Tensor):
        """
        context: [B, M, D]
        """
        B, M, D = context.shape
        assert D == self.feat_dim, f"feat_dim mismatch: got D={D}, expected {self.feat_dim}"

        # ---- Step 1: 逐模态显著特征提取 ----
        x = context.reshape(B * M, 1, D)                     # [B*M, 1, D]
        salient = F.adaptive_max_pool1d(x, self.pool_out)    # [B*M, 1, pool_out]
        salient = salient.squeeze(1)                         # [B*M, pool_out]

        # 逐模态可信度标量
        c = self.mlp_L(salient).view(B, M)                   # [B, M]

        # ---- Step 2: 根据专家模态对求 logit ----
        i_idx = self.pairs_idx[:, 0]                         # [6]
        j_idx = self.pairs_idx[:, 1]                         # [6]
        g_local = c[:, i_idx] + c[:, j_idx]                  # [B, 6]

        ci = c[:, i_idx]
        cj = c[:, j_idx]
        weight = torch.stack([ci, cj], dim=-1)

        # print('g_local： ', g_local)
        # print('weight： ', weight.shape)

        return g_local, weight

class DualGating(nn.Module):
    """双重动态门控机制"""

    def __init__(self, t):
        super().__init__()

        self.global_function = GlobalGating()
        self.local_function = LocalGating()
        self.temperature = nn.Parameter(torch.tensor(t))

        self.w_alpha = nn.Parameter(torch.tensor(-0.0 / 2.0))
        self.w_beta = nn.Parameter(torch.tensor(-0.0 / 2.0))

    def forward(self, context):
        g_global = self.global_function(context)
        g_local, weight = self.local_function(context)
        # print(g_global)
        # print(g_local)
        alpha = torch.sigmoid(self.w_alpha - self.w_beta)
        # print(alpha)
        # print(alpha.shape)
        # print(alpha)
        g = alpha * g_global + (1.0 - alpha) * g_local     # [B, 6]
        # print(g)
        g = F.softmax(g / torch.exp(self.temperature), dim=-1)
        # print(torch.exp(self.temperature))
        # print(g)
        return g




class ModalAwareExpert(nn.Module):
    """模态感知专家"""

    def __init__(self, input_dim1, input_dim2, expert_dim=512):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(input_dim1, expert_dim // 2), nn.GELU(), nn.LayerNorm(expert_dim // 2))
        self.net2 = nn.Sequential(nn.Linear(input_dim2, expert_dim // 2), nn.GELU(), nn.LayerNorm(expert_dim // 2))
        self.fusion = nn.Sequential(nn.Linear(expert_dim, expert_dim), nn.ELU(), nn.LayerNorm(expert_dim))

    def forward(self, x1, x2):
        return self.fusion(torch.cat([self.net1(x1), self.net2(x2)], dim=-1))


class MoE_Fusion(nn.Module):
    """完整的DG-MoE模型"""

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
        # 特征压缩
        compressed = [comp(m) for comp, m in zip(self.compressors, modes)]
        z_stack = torch.stack(compressed, dim=1)
        # 门控计算
        gate = self.gating(z_stack)
        # 预先计算所有专家输出
        expert_outputs = []
        for expert, (i, j) in zip(self.experts, self.mode_pairs):
            expert_outputs.append(expert(modes[i], modes[j]))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, D]
        y = (expert_outputs * gate.unsqueeze(-1)).sum(dim=1)
        # print(y.shape)
        # print(y)
        return y, gate
        # 加权融合
        # fused = (selected * sorted_indices.unsqueeze(-1)).sum(dim=1)
        # fused = selected.sum(dim=1) / K
        # fused=selected.sum(dim=1)
        # print(selected.sum(dim=1))
        fused = selected.sum(dim=1)
        # print(selected.shape)
        # print(selected.sum(dim=1) / K)
        return self.refinement(fused), gate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# 测试验证
if __name__ == "__main__":
    # set_seed(0)
    model = MoE_Fusion(input_dims=[1024, 1280, 3072, 3072, 512], t=-3.0).to('cpu')
    test_inputs = [torch.randn(32, dim) for dim in [1024, 1280, 3072, 3072, 512]]
    for a in test_inputs:
        print(f'a shape ：{a.shape}')

    output,sadad = model(*test_inputs)
    print(f"输出维度: {output.shape}")  # torch.Size([32, 256])
