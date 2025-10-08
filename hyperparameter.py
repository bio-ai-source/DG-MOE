# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""
import torch
from datetime import datetime
class hyperparameter():
    def __init__(self):
        self.current_time = None
        self.type = 'cold'
        self.DATASET = 'DrugBank' # Luo11 Luo110
        self.skip_train = False
        # self.skip_train = False

        self.p_cum = -2.25
        self.K_Fold = 5
        self.epochs = 10 #500
        self.Learning_rate = 0.05 # 5e-5
        self.batch = 32
        self.seed_cross = 0
        self.weight_decay = 5e-4
        self.device = 'cuda'
        # GAT
        self.num_heads=3
        self.dropout = 0.5
        self.num_layers = 4
        # MLP
        self.mlp_layer_output = 4

