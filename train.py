import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, \
    recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.functional import softmax
import numpy as np
import pandas as pd


# Model, loss, and metrics calculation remains the same as earlier.
def train(fold, epoch, model, train_drug_loader, train_protein_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    train_labels = []
    train_preds = []
    train_scores = []
    pbar = tqdm(
        zip(train_drug_loader, train_protein_loader),
        total=len(train_drug_loader),  # 关键点：明确总批次数
        desc=f"Fold {fold}, Epoch {epoch} training",
        unit="batch",
        dynamic_ncols=True,
        leave=False,  # 进度条完成后自动消失
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for batch_idx, (drug_batch, protein_batch) in enumerate(pbar):
        drug_batch.to(device)
        protein_batch.to(device)
        label = drug_batch.label.to(device)

        optimizer.zero_grad()
        predicted, indices, gates = model(drug_batch, protein_batch)
        
        # print(f'combined shape: {combined_embedding.shape}')
        # print(f'predicted shape: {predicted.shape}')
        loss = criterion(predicted, label)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        loss.backward()
        clip_grad_norm_(parameters=model.parameters(), max_norm=5)
        optimizer.step()
        # 这里是用来存储每个批次的标签和预测分数
        interaction = label.cpu().detach().numpy()
        ys = softmax(predicted, 1).cpu().detach().numpy()

        # 更新 train_labels, train_preds, train_scores
        train_labels.extend(interaction)
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))

        train_preds.extend(predicted_labels)
        train_scores.extend(predicted_scores)


    pbar.clear()
    pbar.refresh()

    # 计算指标
    fpr, tpr, thresholds = roc_curve(train_labels, train_scores)  # T,S
    train_acc = accuracy_score(train_labels, train_preds)  # T,Y
    train_auc = roc_auc_score(train_labels, train_scores)  # T,S
    train_aupr = average_precision_score(train_labels, train_scores)
    precision = precision_score(train_labels, train_preds, zero_division=1)  # T,Y
    recall = recall_score(train_labels, train_preds)  # T,Y

    f1 = f1_score(train_labels, train_preds)
    MCC = matthews_corrcoef(train_labels, train_preds)

    return total_loss / len(
        train_drug_loader), train_labels, train_scores, train_acc, train_auc, train_aupr, precision, recall, f1, MCC


def valid(fold, epoch, model, valid_drug_loader, valid_protein_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    valid_labels = []
    valid_preds = []
    valid_scores = []
    total_loss = 0
    with torch.no_grad():  # 在验证时不需要计算梯度
        pbar = tqdm(
            zip(valid_drug_loader, valid_protein_loader),
            total=len(valid_drug_loader),  # 确保总批次数正确
            desc=f"Fold {fold}, Epoch {epoch} validating",
            unit="batch",
            dynamic_ncols=True,
            leave=False,  # 进度条完成后自动消失
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        for batch_idx, (drug_batch, protein_batch) in enumerate(pbar):
            drug_batch.to(device)
            protein_batch.to(device)
            label = drug_batch.label.to(device)

            predicted, indices, gates = model(drug_batch, protein_batch)

            # 计算损失
            loss = criterion(predicted, label)
            total_loss += loss.item()
            # 这里是用来存储每个批次的标签和预测分数
            interaction = label.cpu().detach().numpy()
            ys = softmax(predicted, 1).cpu().detach().numpy()

            # 更新 valid_labels, valid_preds, valid_scores
            valid_labels.extend(interaction)
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            valid_preds.extend(predicted_labels)
            valid_scores.extend(predicted_scores)

        pbar.clear()
        pbar.refresh()

        # 计算指标
        valid_acc = accuracy_score(valid_labels, valid_preds)  # T,Y
        valid_auc = roc_auc_score(valid_labels, valid_scores)  # T,S
        valid_aupr = average_precision_score(valid_labels, valid_scores)
        precision = precision_score(valid_labels, valid_preds, zero_division=1)  # T,Y
        recall = recall_score(valid_labels, valid_preds)  # T,Y
        f1 = f1_score(valid_labels, valid_preds)
        MCC = matthews_corrcoef(valid_labels, valid_preds)

    return total_loss / len(
        valid_drug_loader), valid_labels, valid_scores, valid_acc, valid_auc, valid_aupr, precision, recall, f1, MCC

epochs = 0
def test(fold, model, test_drug_loader, test_protein_loader, device):
    model.eval()
    test_labels = []
    test_preds = []
    test_scores = []
    expert_selection_data = []

    results = []
    with torch.no_grad():  # No gradients needed for evaluation
        pbar = tqdm(
            zip(test_drug_loader, test_protein_loader),
            total=len(test_drug_loader),  # 关键点：明确总批次数
            desc=f"Fold {fold}, Testing",
            unit="batch",
            dynamic_ncols=True,
            leave=False,  # 进度条完成后自动消失
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        for batch_idx, (drug_batch, protein_batch) in enumerate(pbar):
            drug_batch.to(device)
            protein_batch.to(device)
            label = drug_batch.label.to(device)

            predicted, indices, gates = model(drug_batch, protein_batch)

            # 这里是用来存储每个批次的标签和预测分数
            interaction = label.cpu().detach().numpy()
            ys = softmax(predicted, 1).cpu().detach().numpy()

            # 更新 test_labels, test_preds, test_scores
            test_labels.extend(interaction)
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))

            drug_names = []  # Store drug names
            protein_names = []  # Store protein names
            interaction_labels = []  # Store drug-target interaction labels
            drug_names.extend(drug_batch.drug_name)
            protein_names.extend(protein_batch.protein_name)
            interaction_labels.extend(drug_batch.label)
            for drug_name, protein_name, label, score in zip(drug_names, protein_names, interaction, predicted_scores):
                results.append([drug_name, protein_name, label, score])
            test_preds.extend(predicted_labels)
            test_scores.extend(predicted_scores)
            batch_size = len(drug_batch.drug_name)
            topk_indices_cpu = gates.cpu().detach().numpy()  # 转换为numpy数组
            topk_gates_cpu = gates.cpu().detach().numpy()  # 转换为numpy数组

            pre = predicted.cpu().detach().numpy()
            for i in range(batch_size):
                # 构建[DB0021.P21312,0,1,1]格式的数据
                drug_protein = f"{drug_batch.drug_name[i]}.{protein_batch.protein_name[i]}"
                label_val = drug_batch.label[i].item()
                pred = predicted_scores[i]
                index1 = topk_indices_cpu[i].tolist()
                index2 = topk_gates_cpu[i].tolist()

                expert_selection_data.append([
                    drug_protein,
                    label_val,
                    pred,
                    index1,
                    index2
                ])

        df = pd.DataFrame(expert_selection_data,
                          columns=["drug_protein", "label", "predict", "expert_idx1", "expert_idx2"])
        pbar.clear()
        pbar.refresh()
        fpr, tpr, thresholds = roc_curve(test_labels, test_scores)  # T,S
        test_acc = accuracy_score(test_labels, test_preds)  # T,Y
        test_auc = roc_auc_score(test_labels, test_scores)  # T,S
        test_aupr = average_precision_score(test_labels, test_scores)
        precision = precision_score(test_labels, test_preds, zero_division=1)  # T,Y
        recall = recall_score(test_labels, test_preds)  # T,Y
        f1 = f1_score(test_labels, test_preds)
        MCC = matthews_corrcoef(test_labels, test_preds)

    return df, results, test_labels, test_scores, fpr, tpr, test_acc, test_auc, test_aupr, precision, recall, f1, MCC

