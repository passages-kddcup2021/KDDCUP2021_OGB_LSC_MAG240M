import argparse
import pickle

import numpy as np
import torch
from ogb.lsc import MAG240MDataset
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--root', type=str, default='.')
parser.add_argument('--hidden_channels', type=int, default=2048)
args = parser.parse_args()


device = torch.device(f'cuda:{args.device}')
dataset = MAG240MDataset(root=args.root)

commits = ['rgat', 'sgc_rgat']

split_dict = dataset.get_idx_split()
valid_idx = split_dict['valid']
test_idx = split_dict['test']

paper_label = dataset.paper_label
train_Y = paper_label[valid_idx]
test_Y = paper_label[test_idx]

# load results
results = []
for commit in commits:
    with open(f'../post_result/{commit}_result.pkl', 'rb') as f:
        result = pickle.load(f)
    results.append(result)

# weights
weights = [0.2, 0.4, 0.4]

weight_valid_result = []
for result in results:
    for idx, value in enumerate(result['valid'].values()):
        weight_valid_result.append(weights[idx] * value)
weight_valid_result = sum(weight_valid_result)

final_acc = accuracy_score(train_Y, (weight_valid_result).argmax(1).cpu().numpy())
print(f'final valid acc: {final_acc}')

# save result
weight_test_result = []
for result in results:
    for idx, value in enumerate(result['valid'].values()):
        weight_test_result.append(weights[idx] * value)
weight_test_result = sum(weight_test_result)
y_pred = weight_test_result.argmax(1)
assert y_pred.shape == (146818, )

if isinstance(y_pred, torch.Tensor):
    y_pred = y_pred.cpu().numpy()
y_pred = y_pred.astype(np.short)

np.savez_compressed('./saved/y_pred_mag240m', y_pred=y_pred)
