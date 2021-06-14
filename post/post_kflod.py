import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from ogb.lsc import MAG240MDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch.optim import Adam
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--commit', type=str, default='rgat')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--root', type=str, default='.')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--hidden_channels', type=int, default=2048)
parser.add_argument('--out_channels', type=int, default=153)
args = parser.parse_args()


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        hidden_channels = args.hidden_channels
        out_channels = args.out_channels
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, inps):
        logits = self.mlp(inps)
        return logits


commit = args.commit
device = torch.device(f'cuda:{args.device}')

# set seeds
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if not os.path.exists('./saved'):
    os.makedirs('./saved/random', exist_ok=True)
    os.makedirs('./saved/finetune', exist_ok=True)

if not os.path.exists('./post_results'):
    os.makedirs('./post_results', exist_ok=True)

dataset = MAG240MDataset(root=args.root)

split_dict = dataset.get_idx_split()
valid_idx = split_dict['valid']  # numpy array storing indices of validation paper nodes
test_idx = split_dict['test']

paper_label = dataset.paper_label
valid_label = paper_label[valid_idx]
test_label = paper_label[test_idx]

train_X = np.load(f'./results/{commit}/valid_preds.npy')
train_Y = valid_label

test_X = np.load(f'./results/{commit}/test_preds.npy')

mlp_parm = torch.load(f'./results/{commit}/mlp.pt', map_location=lambda storage, loc: storage.cuda(args.device))
# k-fold
kf = KFold(n_splits=5, shuffle=False)  # init KFold
train_X = torch.from_numpy(train_X).float().to(device)
train_Y = torch.from_numpy(train_Y).long().to(device)

test_X = torch.from_numpy(test_X).float().to(device)

# base result
model = Model()
model.to(device)
model.mlp.load_state_dict(mlp_parm)
model.eval()
with torch.no_grad():
    logits = model(test_X)
    logits = torch.softmax(logits, dim=1)
base_test_logits = logits

with torch.no_grad():
    logits = model(train_X)
    logits = torch.softmax(logits, dim=1)
base_valid_logits = logits

base_acc = accuracy_score(train_Y.cpu().numpy(), (base_valid_logits).argmax(1).cpu().numpy())
print(f'base valid acc: {base_acc}')

# finetune result
for idx, (train_index, valid_index) in enumerate(kf.split(train_X)):
    model = Model(args)
    model.to(device)
    model.mlp.load_state_dict(mlp_parm)
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()

    model.to(device)
    train_inps = train_X[train_index]
    train_labs = train_Y[train_index]

    valid_inps = train_X[valid_index]
    valid_labs = train_Y[valid_index]

    pbar = tqdm(range(1000))
    best_acc = 0
    for i in pbar:
        model.train()
        optim.zero_grad()
        logits = model(train_inps)
        loss = ce(logits, train_labs)
        loss.backward()
        optim.step()

        if i % 1 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(valid_inps).cpu().numpy()
            valid_acc = accuracy_score(valid_labs.cpu().numpy(), logits.argmax(1))
            if valid_acc > best_acc:
                torch.save(model.state_dict(), f'./saved/finetune/{commit}_{idx}.pt')
            best_acc = max(valid_acc, best_acc)
            pbar.set_postfix(valid_acc=valid_acc, best_acc=best_acc)

finetune_test_logits = None
finetune_valid_logits = torch.zeros(train_X.shape[0], args.out_channels).to(device)
for idx, (train_index, valid_index) in enumerate(kf.split(train_X)):
    mlp_parm = torch.load(f'../saved/finetune/{commit}_{idx}.pt')
    model = Model()
    model.to(device)
    model.load_state_dict(mlp_parm)

    model.eval()

    with torch.no_grad():
        logits = model(train_X[valid_index])
        logits = torch.softmax(logits, dim=1)
    finetune_valid_logits[valid_index] = logits

    with torch.no_grad():
        logits = model(test_X)
        logits = torch.softmax(logits, dim=1)
    finetune_test_logits = logits if finetune_test_logits is None else logits + finetune_test_logits

finetune_acc = accuracy_score(train_Y.cpu().numpy(), (finetune_valid_logits).argmax(1).cpu().numpy())
print(f'finetune valid acc: {finetune_acc}')

# random result
for idx, (train_index, valid_index) in enumerate(kf.split(train_X)):  # 调用split方法切分数据
    model = Model(args)
    model.to(device)
#     model.mlp.load_state_dict(mlp_parm)
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()

    model.to(device)
    train_inps = train_X[train_index]
    train_labs = train_Y[train_index]

    valid_inps = train_X[valid_index]
    valid_labs = train_Y[valid_index]

    pbar = tqdm(range(600))
    best_acc = 0
    for i in pbar:
        model.train()
        optim.zero_grad()
        logits = model(train_inps)
        loss = ce(logits, train_labs)
        loss.backward()
        optim.step()

        if i % 1 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(valid_inps).cpu().numpy()
            valid_acc = accuracy_score(valid_labs.cpu().numpy(), logits.argmax(1))
            if valid_acc > best_acc:
                torch.save(model.state_dict(), f'./saved/random/{commit}_{idx}.pt')
            best_acc = max(valid_acc, best_acc)
            pbar.set_postfix(valid_acc=valid_acc, best_acc=best_acc)

random_test_logits = None
random_valid_logits = torch.zeros(train_X.shape[0], args.out_channels).to(device)
for idx, (train_index, valid_index) in enumerate(kf.split(train_X)):
    mlp_parm = torch.load(f'../saved/random/{commit}_{idx}.pt')
    model = Model()
    model.to(device)
    model.load_state_dict(mlp_parm)

    model.eval()

    with torch.no_grad():
        logits = model(train_X[valid_index])
        logits = torch.softmax(logits, dim=1)
    random_valid_logits[valid_index] = logits

    with torch.no_grad():
        logits = model(test_X)
        logits = torch.softmax(logits, dim=1)
    random_test_logits = logits if random_test_logits is None else logits + random_test_logits

random_acc = accuracy_score(train_Y.cpu().numpy(), (random_valid_logits).argmax(1).cpu().numpy())
print(f'random valid acc: {random_acc}')

# save result
result_dict = {
    'valid': {
        'base': base_valid_logits,
        'random': random_valid_logits,
        'finetune': finetune_valid_logits
    },
    'test': {
        'base': base_test_logits,
        'random': random_test_logits,
        'finetune': finetune_test_logits
    }
}


with open(f'./post_result/{commit}_result.pkl', 'wb') as f:
    pickle.dump(result_dict, f)
