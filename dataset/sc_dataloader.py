from collections.abc import Mapping, Sequence
from typing import List, Optional, Union

import numpy as np
import torch.utils.data

from torch_geometric.data import Batch, Dataset,Data
from torch_geometric.data.data import BaseData
from utils.graph_plot import plot_3d_graph

class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        if 'edg_ind' in batch[0].keys():
            edg_ind_batch = [item['edg_ind'] for item in batch]
            batch = [item['data'] for item in batch]

            sc_d = Data()

            # for item in zip(batch,edg_ind_batch):
            #     points= item[0]['pos']
            #     edges = item[1]['am0']
            #     plot_3d_graph(points.numpy(), edges.numpy())

            for k in edg_ind_batch[0].keys:
                if k=='am0':
                    sc_l = [item['z'].shape[0] for item in batch]
                elif k == 'am1':
                    sc_l = [item['z1'].shape[0] for item in batch]
                elif k == 'am2':
                    sc_l = [item['z2'].shape[0] for item in batch]
                sc_l.insert(0, 0)
                sc_l = np.cumsum(sc_l)
                res = []
                for i,item in enumerate(edg_ind_batch):
                    res.append(item[k]+sc_l[i])

                sc_d[k] = torch.concat(res, axis = 1)

            self.follow_batch = ['z1','z2','z3']
            batch_data = Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
            ret = {'batch_data': batch_data, 'batch_adj': sc_d}

            # points = batch['pos']
            # edges = sc_d['am0']
            # plot_3d_graph(points.numpy(), edges.numpy())

            return ret
        else:
            sc_batch = [item['adj_mat'] for item in batch]
            batch = [item['data'] for item in batch]

            batch = Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
            sc_d = Data()

            for k in sc_batch[0].keys:
                sc_l = [item[k].shape[0] for item in sc_batch]
                sc_l.insert(0,0)
                sc_l = np.cumsum(sc_l)
                res = torch.zeros([sc_l[-1],sc_l[-1]], dtype=torch.int32)
                for i, data in enumerate(sc_batch):
                    res[sc_l[i]:sc_l[i+1],sc_l[i]:sc_l[i+1]] = data[k]
                assert (res==res.T).all()
                sc_d[k] = res
            ret = {'batch_data':batch,'batch_adj':sc_d}
            return ret


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )
