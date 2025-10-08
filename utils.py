import random
import numpy as np
import torch
# from pandas.conftest import datapath
from sklearn.model_selection import KFold
import numpy as np
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.loader import DataLoader
import torch.optim as optim
from collections import defaultdict
from torch.optim import Optimizer
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import itertools

def generate_negative_samples(dataset):
    # 收集所有唯一的药物和蛋白质名称
    drugs = set()
    proteins = set()
    existing_pairs = set()
    
    for sample in dataset:
        drug, protein, _ = sample
        drugs.add(drug)
        proteins.add(protein)
        existing_pairs.add((drug, protein))
    
    # 生成所有可能的药物-蛋白质组合
    all_pairs = itertools.product(drugs, proteins)
    
    # 过滤已存在的组合并生成负样本
    new_negatives = [
        [drug, protein, 0]
        for drug, protein in all_pairs
        if (drug, protein) not in existing_pairs
    ]
    
    return new_negatives


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k  # How often to perform the slow parameter update
        self.alpha = alpha  # Smoothing factor
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)  # Step of the inner optimizer (SGD)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)  # Update the slow parameters
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0  # Reset the counter after `k` steps
        return loss
    def zero_grad(self):
        self.optimizer.zero_grad()

def get_name():
    drug_name = np.loadtxt('datasets/d.txt', dtype=str, delimiter=' ')
    protein_name = np.loadtxt('datasets/p.txt', dtype=str, delimiter=' ')
    drug_map = {idx : line for idx, line in enumerate(drug_name)}  # 行号从1开始
    protein_map = {idx : line for idx, line in enumerate(protein_name)}  # 行号从1开始
    return drug_map, protein_map

def load_txt_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # 按空格分隔每行的值
            drug_idx, protein_idx, label = map(int, line.split())
            # 将样本作为元组（drug_idx, protein_idx, label）加入到列表中
            data.append((drug_idx, protein_idx, label))
    return data

def get_kfold_data(i, datasets, k):
    fold_size = len(datasets) // k
    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]
    return trainset, validset

def get_data(dataname, datapath):
    dataset = []
    with open(datapath + dataname + '.txt', 'r') as f:
        data_list = f.read().strip().split('\n')
        for data in data_list:
            if dataname == 'DrugBank':
                parts = data.split()  # 假设每行数据用制表符分隔
            else:
                parts = data.split()  # 假设每行数据用制表符分隔
            drug_name, protein_name, label = parts[0], parts[1], parts[-1]
            dataset.append((drug_name, protein_name, label))  # 将标识符和标签作为元组添加到列表中
    random.shuffle(dataset)
    return dataset


def get_cold_data(dataset):
    drugs, proteins = [], []
    for pair in dataset:
        drugs.append(pair[0])
        proteins.append(pair[1])
    drugs = list(set(drugs))
    proteins = list(set(proteins))
    return drugs, proteins

def get_embedding():
    df = pd.read_csv('./datasets/d_features.csv', header=None)
    drug_embedding, protein_embedding, gene_embedding = {}, {}, {}
    for _, row in df.iterrows():
        drug_name = row[1]
        feature_vector = row[3:].values  #
        drug_embedding[drug_name] = feature_vector
    df = pd.read_csv('./datasets/p_features.csv', header=None)
    for _, row in df.iterrows():
        protein_name = row[0]
        feature_vector = row[2:].values
        protein_embedding[protein_name] = feature_vector
    df = pd.read_csv('./datasets/g_features.csv', header=None)
    for _, row in df.iterrows():
        protein_name = row[0]
        feature_vector = row[2:].values  #
        gene_embedding[protein_name] = feature_vector
    return drug_embedding, protein_embedding, gene_embedding


def save_predict_result(DATASET, type, predict_result):
    df = pd.DataFrame(predict_result, columns=["Drug Name", "Protein Name", "Interaction Label", "Predicted Score"])
    file_path = f'results/{DATASET}/{type}/predict_result.csv'
    file_exists = os.path.isfile(file_path)
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)


def DI_DataLoader(data, molecule_graph, protein_graph, molecule_embedding, protein_embedding, gene_embedding):
    data_samples = []
    for sample in data:
        drug_name, protein_name, label = sample
        dg = molecule_graph[drug_name]
        dt = torch.tensor(np.array(molecule_embedding[drug_name], dtype=np.float32), dtype=torch.float)
        label = torch.tensor(int(label))
        # pg = protein_graph[protein_name][1]
        # pt = torch.tensor(np.array(protein_embedding[protein_name], dtype=np.float32), dtype=torch.float)
        data_samples.append(Data(
            x = dg.x,
            edge_index= dg.edge_index,
            edge_attr= dg.edge_attr,
            dt = dt,
            drug_name = drug_name,
            protein_name = protein_name,
            label = label
            )
        )
    return data_samples

def PI_DataLoader(data, molecule_graph, protein_graph, molecule_embedding, protein_embedding, gene_embedding):
    data_samples = []
    for sample in data:

        drug_name, protein_name, label = sample
        # dg = molecule_graph[drug_name][1]
        # dt = torch.tensor(np.array(molecule_embedding[drug_name], dtype=np.float32), dtype=torch.float)
        #
        # protein_id = protein_graph[protein_name][0]
        label = torch.tensor(int(label))

        pg = protein_graph[protein_name]
        pt = torch.tensor(np.array(protein_embedding[protein_name], dtype=np.float32), dtype=torch.float)
        gt = torch.tensor(
            np.array(gene_embedding.get(protein_name, np.zeros(512, dtype=np.float32)), dtype=np.float32),
            dtype=torch.float
        )

        data_samples.append(Data(
            x = pg.x,
            edge_index= pg.edge_index,
            pt = pt,
            gt = gt,
            drug_name = drug_name,
            protein_name = protein_name,
            label = label
            )
        )
    return data_samples

def get_dataloader(train_data, valid_data, test_data,  molecule_graph, protein_graph, molecule_text_embedding, protein_text_embedding, gene_embedding, batch):
    train_drug_samples = DI_DataLoader(data=train_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                       molecule_embedding=molecule_text_embedding,
                                       protein_embedding=protein_text_embedding,
                                       gene_embedding=gene_embedding)
    train_protein_samples = PI_DataLoader(data=train_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                          molecule_embedding=molecule_text_embedding,
                                          protein_embedding=protein_text_embedding,
                                       gene_embedding=gene_embedding)
    valid_drug_samples = DI_DataLoader(data=valid_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                       molecule_embedding=molecule_text_embedding,
                                       protein_embedding=protein_text_embedding,
                                       gene_embedding=gene_embedding)
    valid_protein_samples = PI_DataLoader(data=valid_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                          molecule_embedding=molecule_text_embedding,
                                          protein_embedding=protein_text_embedding,
                                       gene_embedding=gene_embedding)
    test_drug_samples = DI_DataLoader(data=test_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                      molecule_embedding=molecule_text_embedding,
                                      protein_embedding=protein_text_embedding,
                                       gene_embedding=gene_embedding)
    test_protein_samples = PI_DataLoader(data=test_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                         molecule_embedding=molecule_text_embedding,
                                         protein_embedding=protein_text_embedding,
                                        gene_embedding=gene_embedding)

    train_drug_loader = DataLoader(train_drug_samples, batch_size=batch, shuffle=False)
    train_protein_loader = DataLoader(train_protein_samples, batch_size=batch, shuffle=False)
    valid_drug_loader = DataLoader(valid_drug_samples, batch_size=batch, shuffle=False)
    valid_protein_loader = DataLoader(valid_protein_samples, batch_size=batch, shuffle=False)
    test_drug_loader = DataLoader(test_drug_samples, batch_size=batch, shuffle=False)
    test_protein_loader = DataLoader(test_protein_samples, batch_size=batch, shuffle=False)
    return train_drug_loader, train_protein_loader, valid_drug_loader, valid_protein_loader, test_drug_loader, test_protein_loader




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
