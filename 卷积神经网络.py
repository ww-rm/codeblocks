# %%
import random

import numpy as np
import torch
from sklearn.datasets import load_files
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%


class MyDataset(Dataset):
    def __init__(self, path):
        data = load_files(path, load_content=False, shuffle=False)
        self.inputs = data.get("filenames")
        self.targets = data.get("target")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        input_ = read_image(self.inputs[item], ImageReadMode.GRAY).float().to(DEVICE)
        target = torch.tensor([self.targets[item]]).long().to(DEVICE)

        return (input_, target)

# %%


class MyNeuralNetwork(nn.Module):
    def __init__(self, input_channels=1, output_size=10):
        super().__init__()
        self.btnorm = nn.BatchNorm2d(input_channels)
        self.convpooling = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(),
            nn.Linear(128*3*3, output_size)
        )

    def forward(self, inputs):
        outputs = self.btnorm(inputs)
        outputs = self.convpooling(outputs)
        outputs = self.fc(outputs)
        return outputs

# %%


def collate_fn(data):
    """正确组合样本"""
    inputs, targets = map(list, zip(*data))
    inputs = torch.stack(inputs)
    targets = torch.cat(targets)

    return (inputs, targets)

# %%


def eval_metrics(y_true, y_pred):
    """获得指标结果"""
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro")
    r = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=4)

    return (acc, p, r, f1, report)

# %%


def train(model, train_data_loader, criterion, optimizer):
    """训练模型"""
    model.train()
    loss_list = []
    pred_list = []
    true_list = []
    for inputs, targets in train_data_loader:
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        pred_list.append(outputs.argmax(dim=-1).cpu().numpy())
        true_list.append(targets.cpu().numpy())

    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(true_list)

    loss = np.mean(loss_list)
    result = eval_metrics(y_true, y_pred)

    return (loss, *result)

# %%


def evaluate(model, eval_data_loader, criterion):
    """测试模型"""
    model.eval()
    loss_list = []
    pred_list = []
    true_list = []
    with torch.no_grad():
        for inputs, targets in eval_data_loader:
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss_list.append(loss.item())
            pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
            true_list.append(targets.cpu().numpy())

    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(true_list)

    loss = np.mean(loss_list)
    result = eval_metrics(y_true, y_pred)

    return (loss, *result)


# %%
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    train_dataset = MyDataset("./data/digit_data/train/")
    test_dataset = MyDataset("./data/digit_data/test/")

    model = MyNeuralNetwork().to(DEVICE)

    # 训练参数
    learning_rate = 1e-3
    batch_size = 500
    epochs = 25

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    best_f1 = 0
    for i in range(epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        *train_metrics, _ = train(model, train_dataloader, loss_fn, optimizer)
        *evaluate_metrics, _ = evaluate(model, test_dataloader, loss_fn)
        print("Train Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(*train_metrics))
        print("Eval  Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(*evaluate_metrics))
        if best_f1 < evaluate_metrics[3]:
            best_f1 = evaluate_metrics[3]
            torch.save(model.state_dict(), "./data/model/cnn.pt")

    print("Done!")
    # 打印最优一轮测试结果
    model.load_state_dict(torch.load("./data/model/cnn.pt"))
    *_, report = evaluate(model, test_dataloader, loss_fn)
    print(report)

# %%
