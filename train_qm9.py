from dig.threedgraph.dataset.PygQM9SC import QM9SC,BuildSimplex
from dig.threedgraph.method import SphereNet,SMPNN
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run
import torch

# Load the dataset and split
dataset = QM9SC(root='/home/lanhai/restore/dataset/QM9', transform=BuildSimplex(3))
target = 'U0'
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

# Define model, loss, and evaluation
model = SMPNN()
# model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
#                   hidden_channels=128, out_channels=1, int_emb_size=64,
#                   basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
#                   num_spherical=3, num_radial=6, envelope_exponent=5,
#                   num_before_skip=1, num_after_skip=2, num_output_layers=3)
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate
run3d = run()
run3d.run('cuda', train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          epochs=20, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15)