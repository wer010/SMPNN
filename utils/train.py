import time
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class Train():
    r"""
    The base script for running different 3DGN methods.
    """

    def __init__(self,root_dir,logger):
        self.root_dir = root_dir
        self.logger = logger
        self.save_dir = os.path.join(self.root_dir,'ckpt')


    def run(self, device, train_loader, valid_loader, test_loader, model, loss_func, evaluation, epochs=500,
            lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0,
            energy_and_force=False, p=100):

        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)


        best_valid = float('inf')
        best_test = float('inf')

        if self.save_dir != '':
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        writer = SummaryWriter(log_dir=self.root_dir)

        for epoch in range(1, epochs + 1):
            self.logger.info("\n=====Epoch {}".format(epoch))

            self.logger.info('\nTraining...')
            model.train()
            loss_accum = 0
            for step, batch_data in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                if isinstance(batch_data,dict):
                    for key in batch_data:
                        batch_data[key] = batch_data[key].to(device)
                else:
                    batch_data = batch_data.to(device)
                out = model(batch_data)
                if energy_and_force:
                    force = - \
                    grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out), create_graph=True,
                         retain_graph=True)[0]
                    e_loss = loss_func(out, batch_data.y.unsqueeze(1))
                    f_loss = loss_func(force, batch_data.force)
                    loss = e_loss + p * f_loss
                else:
                    loss = loss_func(out, batch_data.y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                loss_accum += loss.detach().cpu().item()

            train_mae = loss_accum / (step + 1)

            self.logger.info('\n\nEvaluating...')
            valid_mae = self.val(model, valid_loader, energy_and_force, p, evaluation, device)

            self.logger.info('\n\nTesting...')
            test_mae = self.val(model, test_loader, energy_and_force, p, evaluation, device)

            self.logger.info({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae})


            writer.add_scalar('train_mae', train_mae, epoch)
            writer.add_scalar('valid_mae', valid_mae, epoch)
            writer.add_scalar('test_mae', test_mae, epoch)

            if valid_mae < best_valid:
                best_valid = valid_mae
                best_test = test_mae
                if self.save_dir != '':
                    self.logger.info('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid,
                                  'num_params': num_params}
                    torch.save(checkpoint, os.path.join(self.save_dir, 'best_checkpoint.pt'))

            scheduler.step()

        self.logger.info(f'Best validation MAE so far: {best_valid}')
        self.logger.info(f'Test MAE when got best validation result: {best_test}')

        writer.close()

    def val(self, model, data_loader, energy_and_force, p, evaluation, device):
        r"""
        The script for validation/test.

        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy. (default: :obj:`100`)
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.

        :rtype: Evaluation result. ( :obj:`mae`)

        """
        model.eval()

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)

        if energy_and_force:
            preds_force = torch.Tensor([]).to(device)
            targets_force = torch.Tensor([]).to(device)

        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out = model(batch_data)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out), create_graph=True,
                              retain_graph=True)[0]
                preds_force = torch.cat([preds_force, force.detach_()], dim=0)
                targets_force = torch.cat([targets_force, batch_data.force], dim=0)
            preds = torch.cat([preds, out.detach_()], dim=0)
            targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)

        input_dict = {"y_true": targets, "y_pred": preds}

        if energy_and_force:
            input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']
            self.logger.info({'Energy MAE': energy_mae, 'Force MAE': force_mae})
            return energy_mae + p * force_mae

        return evaluation.eval(input_dict)['mae']
