import os.path as osp

import numpy as np
import torch
from ogb.lsc import MAG240MDataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, to_torch_sparse
from tqdm import tqdm
import argparse


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk), 'get'):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk), 'save'):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]


def save_laplacian(directory):
    dataset = MAG240MDataset(directory)

    num_papers = dataset.num_papers
    row, col, _ = torch.load(f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()

    paper_adj_t = SparseTensor(row=row, col=col, sparse_sizes=(num_papers, num_papers), is_sorted=True)
    index = torch.stack([row, col], dim=0)
    del row, col

    paper_matrix = gcn_norm(paper_adj_t, add_self_loops=False)
    del paper_adj_t

    val = paper_matrix.storage.value()
    laplacian_mat = to_torch_sparse(index, val, num_papers, num_papers)
    del index, val, paper_matrix

    path = f'{dataset.dir}/laplacian.pt'
    torch.save(laplacian_mat, path)


def read_mat(directory):
    k = 6
    dataset = MAG240MDataset(directory)
    train_idx = dataset.get_idx_split('train')
    val_idx = dataset.get_idx_split('valid')

    for i in tqdm(range(k)):
        file_name = f'{dataset.dir}/{i}_step_paper_feature.npy'
        outpath = f'{dataset.dir}/labeled_node_feat/train_val_feat_{i}.npz'
        mat = np.load(file_name, mmap_mode='r')
        train = mat[train_idx]
        val = mat[val_idx]
        np.savez(outpath, train=train, val=val)
        del train, val, mat


def feature_transformation(directory):
    print("running")
    dataset = MAG240MDataset(directory)
    num_papers = dataset.num_papers

    print(num_papers)

    print("reading laplacian matrix...")
    laplacian = torch.load(f'{dataset.dir}/laplacian.pt')
    print("reading laplacian matrix done...")

    path = osp.join(dataset.dir, 'processed', 'paper', 'node_feat.npy')
    x = np.memmap(path, dtype=np.float16, mode='r', shape=(num_papers, 768))
    print("shape of node_feat is", x.shape)

    k = 3
    num_features = 768
    dim_chunk_size = 128
    in_x = x

    outpath = osp.join(dataset.dir, 'processed', 'paper', 'node_feat_copy.npy')
    dst = np.memmap(outpath, dtype=np.float16, mode='w+', shape=(num_papers, 768))
    for idx in tqdm(range(k + 1), desc='k'):
        for i in tqdm(range(0, num_features, dim_chunk_size), desc='feature'):
            j = min(i + dim_chunk_size, num_features)
            inputs = get_col_slice(
                in_x, start_row_idx=0,
                end_row_idx=num_papers,
                start_col_idx=i, end_col_idx=j)
            inputs = torch.from_numpy(inputs).to(torch.float32)
            outputs = laplacian.matmul(inputs).numpy().astype(np.float16)
            del inputs
            save_col_slice(
                x_src=outputs, x_dst=dst, start_row_idx=0,
                end_row_idx=num_papers,
                start_col_idx=i, end_col_idx=j)
            del outputs
        if idx == 0:
            del x, in_x
            print("delete x done")
        in_x = dst
        outpath = f'{dataset.dir}/{idx + 1}_step_paper_feature.npy'
        print(f"saving {idx + 1}-step matrix")
        np.save(outpath, dst)
        print("save done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.')
    args = parser.parse_args()
    save_laplacian(args.root)
    feature_transformation(args.root)
    read_mat(args.root)
