import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from .sc_dataloader import DataLoader
from scipy.spatial import Delaunay
from typing import Union
from torch import Tensor

IndexType = Union[slice, Tensor, np.ndarray]


class BuildSimplex(torch.nn.Module):

    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, data):
        """
        Returns:
            features of simplices and the adjacent matrix
        """
        pos = data.pos.numpy()
        assert pos.shape[1] == self.order

        sc_i = Delaunay(pos)
        s3_i = sc_i.simplices
        s2_i, am2_i = extrace_ls_from_hs(s3_i)
        s1_i, am1_i = extrace_ls_from_hs(s2_i)
        s0_i= sc_i.points
        am0_i = get_adj_mat(s0_i, s1_i)
        adj_mat = Data()
        # the k-simplices in a batch can be concat like other tensors, but the adj matrix in a batch can't,
        # therefore I use an additional Data to save the adj matrix to easier collate
        # give the k-simplex feature, barycenter

        adj_mat.am0 = torch.tensor(am0_i, dtype=torch.int32)

        data.s1 = torch.tensor(s1_i, dtype=torch.int32)
        pos1 = (pos[s1_i[:,0],:]+ pos[s1_i[:,1],:])/2.0         #barycentre coordinate
        ab1 = -pos[s1_i[:,0],:]+ pos[s1_i[:,1],:]               # vector AB
        z1 = np.linalg.norm(ab1,2,axis=-1)                      #length of the edge
        ori1 = ab1/np.expand_dims(z1, -1)                       #normailized orientation vector
        adj_mat.am1 = torch.tensor(am1_i, dtype=torch.int32)    #adj matrix

        data.s2 = torch.tensor(s2_i, dtype=torch.int32)
        pos2 = (pos[s2_i[:, 0], :] + pos[s2_i[:, 1], :]+ pos[s2_i[:, 2], :]) / 3.0          #barycentre coordinate
        ab2 = -pos[s2_i[:, 0], :] + pos[s2_i[:, 1], :]                                      #vector AB
        ac2 = -pos[s2_i[:, 0], :] + pos[s2_i[:, 2], :]                                      #vector AC
        ori2 = np.cross(ab2,ac2)
        z2 = np.linalg.norm(ori2,2,axis=-1)/2.0                                             #area of triangle
        ori2 = ori2/np.expand_dims(z2, -1)                                                  #normailized normal vector
        adj_mat.am2 = torch.tensor(am2_i, dtype=torch.int32)

        data.s3 = torch.tensor(s3_i, dtype=torch.int32)
        pos3 = (pos[s3_i[:, 0], :] + pos[s3_i[:, 1], :] + pos[s3_i[:, 2], :]+ pos[s3_i[:, 3], :]) / 4.0
        ab3 = -pos[s3_i[:, 0], :] + pos[s3_i[:, 1], :]                                      #vector AB
        ac3 = -pos[s3_i[:, 0], :] + pos[s3_i[:, 2], :]                                      #vector AC
        ad3 = -pos[s3_i[:, 0], :] + pos[s3_i[:, 3], :]                                      #vector AD
        z3 = np.abs(np.expand_dims(np.cross(ab3,ac3),-2) @ np.expand_dims(ad3,-1))/6.0      #volume of tetrahedron
        z3 = np.squeeze(z3)
        data.z1 = torch.tensor(z1, dtype=torch.float32)
        data.ori1 = torch.tensor(ori1, dtype=torch.float32)
        data.z2 = torch.tensor(z2, dtype=torch.float32)
        data.ori2 = torch.tensor(ori2, dtype=torch.float32)
        data.z3 = torch.tensor(z3, dtype=torch.float32)

        data.pos1 = torch.tensor(pos1, dtype=torch.float32)
        data.pos2 = torch.tensor(pos2, dtype=torch.float32)
        data.pos3 = torch.tensor(pos3, dtype=torch.float32)
        ret = {'data': data, 'adj_mat': adj_mat}
        return ret

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation.value}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"

def extrace_ls_from_hs(t):
    envelope = []
    ind_list = []
    for tet in t:
        ind = []
        for i in range(tet.shape[0]):
            res = np.delete(tet,i).tolist()
            res.sort()
            # if face has already been encountered, then it's not on the envelope
            # the magic of hashsets makes that check O(1) (eg. extremely fast)
            if res in envelope:
                ind.append(envelope.index(res))
                continue

                # if not encoutered yet, add it flipped
            else:
                envelope.append(res)
                ind.append(len(envelope)-1)
        ind_list.append(ind)
    # there is now only faces encountered once (or an odd number of times for paradoxical meshes)
    # another important task is generate the adjacency matrix
    am = np.zeros([len(envelope),len(envelope)],dtype=int)

    for ind in ind_list:
        for i in range(len(ind)):
            am[ind[i], ind[:i]+ind[i+1:]]=1

    # check if adj matrix is diagonal
    assert (am==am.T).all()
    return np.array(envelope), am

def get_adj_mat(points,edges):
    am = np.zeros([points.shape[0],points.shape[0]],dtype=int)

    for ind in edges:
        ind = ind.tolist()
        am[ind, ind[::-1]]=1
    return am


class QM9SC(InMemoryDataset):

    def __init__(self, root='dataset/', transform=None, pre_transform=None, pre_filter=None):

        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.folder = osp.join(root, 'qm9')

        super(QM9SC, self).__init__(self.folder, transform, pre_transform, pre_filter)
        # self.process()
        # self.process_sc()

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return 'qm9_sc_pyg.pt'



    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):

        data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        R = data['R']
        Z = data['Z']
        N = data['N']
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z, split)
        target = {}

        for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']:
            target[name] = np.expand_dims(data[name], axis=-1)
        # y = np.expand_dims([data[name] for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']], axis=-1)



        data_list = []
        num_invalid = 1
        for i in tqdm(range(len(N))):
            # if R_qm9[i].shape[0]<4:
            #     print("The num of atoms in the molecule is less than 4. Omit this molecule.")
            #     continue
            R_i = torch.tensor(R_qm9[i], dtype=torch.float32)

            try:
                sc_i = Delaunay(R_qm9[i])
            except:
                print(f"{num_invalid} molecules is invalid to build simplicial complex. Omit the molecules.")
                print(Z_qm9[i])
                num_invalid+=1
                continue

            z_i = torch.tensor(Z_qm9[i], dtype=torch.int64)
            y_i = [torch.tensor(target[name][i], dtype=torch.float32) for name in
                   ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']]
            data = Data(pos=R_i, z=z_i, y=y_i[0], mu=y_i[0], alpha=y_i[1], homo=y_i[2], lumo=y_i[3], gap=y_i[4],
                        r2=y_i[5], zpve=y_i[6], U0=y_i[7], U=y_i[8], H=y_i[9], G=y_i[10], Cv=y_i[11])

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


    # def process_sc(self):
    #
    #     data = np.load(osp.join(self.raw_dir, self.raw_file_names))
    #
    #     R = data['R']
    #     Z = data['Z']
    #     N = data['N']
    #     split = np.cumsum(N)
    #     R_qm9 = np.split(R, split)
    #     Z_qm9 = np.split(Z, split)
    #     target = {}
    #
    #     for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']:
    #         target[name] = np.expand_dims(data[name], axis=-1)
    #     # y = np.expand_dims([data[name] for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']], axis=-1)
    #
    #     data_list = []
    #     num_invalid = 1
    #     for i in tqdm(range(len(N))):
    #         # if R_qm9[i].shape[0]<4:
    #         #     print("The num of atoms in the molecule is less than 4. Omit this molecule.")
    #         #     continue
    #         try:
    #             sc_i = Delaunay(R_qm9[i])
    #         except:
    #             print(f"{num_invalid} molecules is invalid to build simplicial complex. Omit the molecules.")
    #             print(Z_qm9[i])
    #             num_invalid+=1
    #             continue
    #
    #         s3_i = sc_i.simplices
    #         s2_i, am2_i = extrace_ls_from_hs(s3_i)
    #         s1_i, am1_i = extrace_ls_from_hs(s2_i)
    #         s0_i= sc_i.points
    #         am0_i = get_adj_mat(s0_i, s1_i)
    #
    #         z_i = torch.tensor(Z_qm9[i], dtype=torch.int32)
    #
    #         data = Data(s0=s0_i, am0=am0_i, s1=s1_i, am1=am1_i, s2=s2_i,am2=am2_i, s3=s3_i)
    #
    #         data_list.append(data)
    #
    #     if self.pre_filter is not None:
    #         data_list = [data for data in data_list if self.pre_filter(data)]
    #     if self.pre_transform is not None:
    #         data_list = [self.pre_transform(data) for data in data_list]
    #
    #     data, slices = self.collate(data_list)
    #
    #     print('Saving...')
    #     torch.save((data, slices), self.processed_sc_file_names)



    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union['Dataset', Data]:

        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            if self.transform is None:
                return data
            else:
                data = self.transform(data)
                return data

        else:
            return self.index_select(idx)

if __name__ == '__main__':
    dataset = QM9SC(root='/home/lanhai/restore/dataset/QM9')

    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    target = 'mu'
    dataset.data.y = dataset.data[target]
    print(dataset.data.y.shape)
    print(dataset.data.y)
    print(dataset.data.mu)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    print(split_idx)
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
        split_idx['test']]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))
    print(data)