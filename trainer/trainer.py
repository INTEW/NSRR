import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision import utils as vutils
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import pytorch_colors as colors

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            # self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        for batch_idx, (img_view, img_depth, img_flow, img_view_truth) in enumerate(self.data_loader):
            # print("img_view.shape = ", img_view.shape)
            img_view = img_view.to(self.device)
            img_depth = img_depth.to(self.device)
            img_flow = img_flow.to(self.device)
            img_view_truth = img_view_truth.to(self.device)
            target = img_view_truth[:,:,0,:,:]

            self.optimizer.zero_grad()
            output = self.model(img_view, img_depth, img_flow)
            # vutils.save_image(target[0], './output/target.png'.format(epoch))
            vutils.save_image(output[0], './output/output.png'.format(epoch))
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

    # def _valid_epoch(self, epoch):
    #     """
    #     Validate after training an epoch
    #     :param epoch: Integer, current training epoch.
    #     :return: A log that contains information about validation
    #     """
    #     self.model.eval()
    #     self.valid_metrics.reset()
    #     with torch.no_grad():
    #         # for batch_idx, (data, target) in enumerate(self.valid_data_loader):
    #         #     data, target = data.to(self.device), target.to(self.device)

    #         #     output = self.model(data)
    #         #     loss = self.criterion(output, target)

    #         #     self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
    #         #     self.valid_metrics.update('loss', loss.item())
    #         #     for met in self.metric_ftns:
    #         #         self.valid_metrics.update(met.__name__, met(output, target))
    #         #     self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

    #         for batch_idx, (img_view, img_depth, img_flow, img_view_truth) in enumerate(self.valid_data_loader):
    #             img_view = img_view.to(self.device)
    #             img_depth = img_depth.to(self.device)
    #             img_flow = img_flow.to(self.device)
    #             img_view_truth = img_view_truth.to(self.device)
    #             target = img_view_truth[0].unsqueeze(0)

    #             self.optimizer.zero_grad()
    #             output = self.model(img_view, img_depth, img_flow)
    #             loss = self.criterion(output, img_view_truth, 1.0)

    #             self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
    #             self.valid_metrics.update('loss', loss.item())
    #             for met in self.metric_ftns:
    #                 self.valid_metrics.update(met.__name__, met(output, target))
    #             self.writer.add_image('input', make_grid(img_view.cpu(), nrow=8, normalize=True))

    #     # add histogram of model parameters to the tensorboard
    #     for name, p in self.model.named_parameters():
    #         self.writer.add_histogram(name, p, bins='auto')
    #     return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
