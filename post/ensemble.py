import argparse
import pickle

import numpy as np
import torch
from ogb.lsc import MAG240MDataset
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--online', action='store_true')
parser.add_argument('--root', type=str, default='.')
parser.add_argument('--hidden_channels', type=int, default=2048)
args = parser.parse_args()


device = torch.device(f'cuda:{args.device}')
online = args.online
phase = 'online' if online else 'offline'
dataset = MAG240MDataset(root=args.root)

commits = ['2048_hidden', 'sgc_2048', 'gat']

split_dict = dataset.get_idx_split()
valid_idx = split_dict['valid']  # numpy array storing indices of validation paper nodes
test_idx = split_dict['test']

paper_label = dataset.paper_label
valid_label = paper_label[valid_idx]
test_label = paper_label[test_idx]

if online:
    test_Y = test_label
else:
    test_Y = valid_label[100000:]

results = []
for commit in commits:
    with open(f'../post_result/{phase}_{commit}_result.pkl', 'rb') as f:
        result = pickle.load(f)
    results.append(result)

# weight
weight_result = []
weights = [0.2, 0.4, 0.4]
for result in results:
    for idx, value in enumerate(result.values()):
        weight_result.append(weights[idx] * value)
weight_result = sum(weight_result)

final_acc = accuracy_score(test_Y, (weight_result).argmax(1).cpu().numpy())
print(f'final acc: {final_acc}')

# save result
y_pred = weight_result.argmax(1)
assert y_pred.shape == (146818, )

if isinstance(y_pred, torch.Tensor):
    y_pred = y_pred.cpu().numpy()
y_pred = y_pred.astype(np.short)

np.savez_compressed('./y_pred_mag240m', y_pred=y_pred)
