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
parser.add_argument('--commit', type=str, default='base')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--online', action='store_true')
parser.add_argument('--root', type=str, default='.')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--hidden_channels', type=int, default=2048)
args = parser.parse_args()


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        hidden_channels = args.hidden_channels
        out_channels = 153
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
online = args.online
phase = 'online' if online else 'offline'

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

valid = np.load(f'./results/{commit}/valid_preds.npy')
test = np.load(f'./results/{commit}/test_preds.npy')

if online:
    train_X = valid
    train_Y = valid_label

    test_X = test
    test_Y = test_label
else:
    train_X = valid[:100000]
    train_Y = valid_label[:100000]

    test_X = valid[100000:]
    test_Y = valid_label[100000:]

mlp_parm = torch.load(f'./results/{commit}/mlp.pt', map_location=lambda storage, loc: storage.cuda(args.device))
# k-fold
kf = KFold(n_splits=5, shuffle=False)  # init KFold
train_X = torch.from_numpy(train_X).float().to(device)
train_Y = torch.from_numpy(train_Y).long().to(device)

test_X = torch.from_numpy(test_X).float().to(device)
test_Y = torch.from_numpy(test_Y).long().to(device)

# base result
model = Model(args)
model.to(device)
model.mlp.load_state_dict(mlp_parm)
model.eval()
with torch.no_grad():
    logits = model(test_X)
    logits = torch.softmax(logits, dim=1)
base_test_logits = logits

base_acc = accuracy_score(test_Y.cpu().numpy(), (base_test_logits).argmax(1).cpu().numpy())
print(f'base_acc: {base_acc}')

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
                torch.save(model.state_dict(), f'./saved/finetune/{phase}_{commit}_{idx}.pt')
            best_acc = max(valid_acc, best_acc)
            pbar.set_postfix(valid_acc=valid_acc, best_acc=best_acc)

finetune_test_logits = None
for idx in range(5):
    parms = torch.load(f'./saved/finetune/{phase}_{commit}_{idx}.pt')
    model = Model(args)
    model.to(device)
    model.load_state_dict(parms)

    model.eval()
    with torch.no_grad():
        logits = model(test_X)
        logits = torch.softmax(logits, dim=1)
    finetune_test_logits = logits if finetune_test_logits is None else logits + finetune_test_logits

finetune_acc = accuracy_score(test_Y.cpu().numpy(), (finetune_test_logits).argmax(1).cpu().numpy())
print(f'finetune_acc: {finetune_acc}')

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
                torch.save(model.state_dict(), f'./saved/random/{phase}_{commit}_{idx}.pt')
            best_acc = max(valid_acc, best_acc)
            pbar.set_postfix(valid_acc=valid_acc, best_acc=best_acc)

random_test_logits = None
for idx in range(5):
    parms = torch.load(f'./saved/random/{phase}_{commit}_{idx}.pt')
    model = Model(args)
    model.to(device)
    model.load_state_dict(parms)

    model.eval()
    with torch.no_grad():
        logits = model(test_X)
        logits = torch.softmax(logits, dim=1)
    random_test_logits = logits if random_test_logits is None else logits + random_test_logits

random_acc = accuracy_score(test_Y.cpu().numpy(), (random_test_logits).argmax(1).cpu().numpy())
print(f'random_acc: {random_acc}')

# save result
result_dict = {
    'base': base_test_logits,
    'random': random_test_logits,
    'finetune': finetune_test_logits
}


with open(f'./post_result/{phase}_{commit}_result.pkl', 'wb') as f:
    pickle.dump(result_dict, f)
