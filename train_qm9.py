from dataset.PygQM93D import QM93D
from model.schnet import SchNet
from model.mpnn import MPNN
from utils.eval import ThreeDEvaluator
from utils.train import Train
from utils.logger import create_logger
import torch
import os
import argparse
from datetime import datetime
from torch_geometric.data import DataLoader

def init_arg():
    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("--model_type",
                        default="schnet",
                        help="Which variant to use.")

    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers")

    parser.add_argument("--num_layers", default=4, type=int,
                        help="num_layers")
    parser.add_argument("--num_heads", default=2, type=int,
                        help="num_heads")

    parser.add_argument("--embedding_dim", default=128, type=int,
                        help="embedding_dim")
    parser.add_argument('-p', '--positional-embedding',
                        type=str.lower,
                        choices=['learnable', 'sine', 'none'],
                        default='sine', dest='positional_embedding')
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--pretrain", default=None, type=str,
                        help="path of pretrain model")

    parser.add_argument("--learning_rate", default=0.0005, type=float,
                        help="The initial learning rate for SGD.")

    parser.add_argument("--weight_decay", default=0.05, type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument('--clip-grad-norm', default=0., type=float,
                        help='gradient norm clipping (default: 0 (disabled))')

    parser.add_argument("--epoch", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup", default= 0, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--warmup_lr", default=0.001, type=float,
                        help="lr of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    return args


def main():
    args = init_arg()

    ctime = datetime.now()
    run_path = './runs/' + ctime.strftime("%Y-%b-%d_%H:%M:%S") + '_' + args.model_type + '_qm9'
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    logger = create_logger(output_dir=run_path,  name=args.model_type)


    # Load the dataset and split
    dataset = QM93D(root='/home/lanhai/restore/dataset/QM9')
    target = 'U0'
    dataset.data.y = dataset.data[target]
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.eval_batch_size, shuffle=False)


    # Define model, loss, and evaluation
    model = MPNN(energy_and_force=False,
                   cutoff=10.0,
                   num_layers=6,
                   hidden_channels=128,
                   out_channels=1,
                   num_filters=128,
                   num_gaussians=50)

    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()

    logger.info(model)
    # Train and evaluate
    run3d = Train(run_path, logger)
    run3d.run('cuda', train_loader, valid_loader, test_loader, model, loss_func, evaluation,
              epochs=args.epoch,  lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15)



if __name__ == '__main__':
    main()
