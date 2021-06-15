import argparse

import numpy as np
import torch
import torch.nn.functional as F
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from torch.nn import BatchNorm1d, Identity, Linear, ModuleList
from torch.utils.data import DataLoader
from tqdm import tqdm


class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = True, relu_last: bool = False):
        super(MLP, self).__init__()

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.batch_norms = ModuleList()
        for _ in range(num_layers - 1):
            norm = BatchNorm1d(hidden_channels) if batch_norm else Identity()
            self.batch_norms.append(norm)

        self.weight_layer = Linear(768, 1)
        self.dropout = dropout
        self.relu_last = relu_last

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        x = x.transpose(0, 1)
        all_attention_feat = self.weight_layer(x).squeeze(dim=2).T
        attention_score = F.softmax(all_attention_feat, dim=1)
        del all_attention_feat
        x = (x.T * attention_score).T
        x = x.sum(dim=0)

        for lin, batch_norm in zip(self.lins[:-1], self.batch_norms):
            x = lin(x)
            if self.relu_last:
                x = batch_norm(x).relu_()
            else:
                x = batch_norm(x.relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def train(model, x_train, y_train, batch_size, optimizer):
    model.train()

    total_loss = 0
    for idx in DataLoader(range(y_train.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()

    return total_loss / y_train.size(0)


@torch.no_grad()
def test(model, x_eval, y_eval):
    model.eval()
    y_pred = model(x_eval).argmax(dim=-1)
    return evaluator.eval({'y_true': y_eval, 'y_pred': y_pred})['acc']


def get_embedding(model, dataset):

    np_feat_list = []
    num_papers = dataset.num_papers
    eval_batch_size = 380000
    for i in range(5):
        path = f'{dataset.dir}/{i}_step_paper_feature.npy'
        mat = np.load(path, mmap_mode='r')
        np_feat_list.append(mat)

    model.eval()
    with torch.no_grad():
        save_list = []
        for idx in tqdm(DataLoader(range(num_papers), eval_batch_size, shuffle=False)):
            paper_feat_list = []
            for i in range(5):
                paper_feat_list.append(np_feat_list[i][idx])
            all_paper_feat = np.stack(paper_feat_list, axis=0)
            del paper_feat_list
            paper_feat = torch.from_numpy(all_paper_feat).to(torch.float).cuda()

            all_attention_feat = model.weight_layer(paper_feat).squeeze(dim=2).T
            attention_score = F.softmax(all_attention_feat, dim=1)
            del all_attention_feat
            paper_feat = (paper_feat.T * attention_score).T
            del attention_score
            paper_feat = paper_feat.sum(dim=0)
            np_feat = paper_feat.cpu().numpy()
            del paper_feat
            save_list.append(np_feat)

        outpath = f'{dataset.dir}/att_node_feat.npy'
        paper_feat = np.concatenate(save_list, axis=0)

        print("saving embeddings")
        np.save(outpath, paper_feat)
        print("save done")
        del paper_feat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=2048)
    parser.add_argument('--num_layers', type=int, default=2),
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_last', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=38000)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(42)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = MAG240MDataset(args.root)
    evaluator = MAG240MEvaluator()

    train_list = []
    val_list = []
    k = 3
    for i in tqdm(range(0, k + 1)):
        path = f'{dataset.dir}/labeled_node_feat/train_val_feat{i}.npz'
        data = np.load(path)
        train_list.append(data['train'])
        val_list.append(data['val'])
    x_train = np.stack(train_list, axis=1)
    x_val = np.stack(val_list, axis=1)

    x_train = torch.from_numpy(x_train).to(torch.float).cuda()
    x_valid = torch.from_numpy(x_val).to(torch.float).cuda()

    print(x_train.shape)

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')

    y_train = torch.from_numpy(dataset.paper_label[train_idx])
    y_train = y_train.to(torch.long).cuda()
    y_valid = torch.from_numpy(dataset.paper_label[valid_idx])
    y_valid = y_valid.to(torch.long).cuda()

    num_features = 768
    num_paper_features = num_features
    num_classes = 153

    model = MLP(num_paper_features, args.hidden_channels,
                num_classes, args.num_layers, args.dropout,
                not args.no_batch_norm, args.relu_last)
    model = torch.nn.DataParallel(model).cuda()

    print('number of GPUs available:{}'.format(torch.cuda.device_count()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f'#Params: {num_params}')

    pbar = tqdm(range(args.epochs))
    best_valid_acc = 0
    for epoch in pbar:
        loss = train(model, x_train, y_train, args.batch_size, optimizer)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                valid_acc = test(model, x_valid, y_valid)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), f'{dataset.dir}/saved/mlp.pth')
                    print("best_valid_result", best_valid_acc)

                pbar.set_postfix(valid_acc=valid_acc, best_acc=best_valid_acc)

    print(f'Valid: {best_valid_acc: .4f}')
    get_embedding(model, dataset)
