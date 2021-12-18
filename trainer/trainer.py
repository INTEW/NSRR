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

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        # it = iter(self.data_loader)
        # item = next(it)
        # print('----------------------------------')
        # print(len(item))
        for batch_idx, (img_view, img_depth, img_flow, img_view_truth) in enumerate(self.data_loader):
            img_view = img_view.to(self.device)
            img_depth = img_depth.to(self.device)
            img_flow = img_flow.to(self.device)
            img_view_truth = img_view_truth.to(self.device)
            target = img_view_truth[0].unsqueeze(0)
            # vutils.save_image(target, './target/target_{}.png'.format(epoch))
            # vutils.save_image(img_view[0], './target/img_view_{}.png'.format(epoch))
            self.optimizer.zero_grad()
            output = self.model(img_view, img_depth, img_flow)
            # print(output.shape)
            # print(target.shape)
            loss = self.criterion(output, target, 0.1)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                # print("img_view:", img_view.shape)
                # print("img_depth:", img_depth.shape)
                # print("img_flow:", img_flow.shape)
                # print("img_truth:", img_view_truth.shape)
                self.train_metrics.update(met.__name__, met(output, target))
            #print("loss = ", loss)
            #vutils.save_image(img_view[0], './output/{}.png'.format(batch_idx))
            vutils.save_image(output, './output/epoch_{}_id_{}.png'.format(epoch,batch_idx))
            # vutils.save_image(target, './target/{}.png'.format(batch_idx))
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(img_view.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            #     data, target = data.to(self.device), target.to(self.device)

            #     output = self.model(data)
            #     loss = self.criterion(output, target)

            #     self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            #     self.valid_metrics.update('loss', loss.item())
            #     for met in self.metric_ftns:
            #         self.valid_metrics.update(met.__name__, met(output, target))
            #     self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            for batch_idx, (img_view, img_depth, img_flow, img_view_truth) in enumerate(self.valid_data_loader):
                img_view = img_view.to(self.device)
                img_depth = img_depth.to(self.device)
                img_flow = img_flow.to(self.device)
                img_view_truth = img_view_truth.to(self.device)
                target = img_view_truth[0].unsqueeze(0)

                self.optimizer.zero_grad()
                output = self.model(img_view, img_depth, img_flow)
                loss = self.criterion(output, img_view_truth, 1.0)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(img_view.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
