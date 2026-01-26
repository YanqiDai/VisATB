import copy
import random
import math
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.distributed as dist

import csv

torch.autograd.set_detect_anomaly(True)

class WeightMethod:
    """
    Abstract class for weighting methods.
    """
    def __init__(self, num_tasks, device):
        super().__init__()
        self.num_tasks = num_tasks
        self.device = device
        self.weights = None
    
    def get_task_sample_num(self, task_ids):
        """
        Get the number of samples for each task.
        Input:
            task_ids: [batch_size], task id for each sample.
        Output:
            task_sample_num: [num_task], number of samples for each task.
        """
        task_sample_num = torch.zeros(self.num_tasks).to(self.device)
        for task_id in range(self.num_tasks):
            task_sample_num[task_id] = torch.sum(task_ids == task_id)
        return task_sample_num

    def get_task_token_num(self, losses, task_ids):
        """
        Get the number of tokens for each task.
        Input:
            losses: [batch_size, seq_len], losses for each token, 0 for padding tokens.
            task_ids: [batch_size], task id for each sample.
        Output:
            task_token_num: [num_task], number of tokens for each task.
        """
        mask = losses != 0
        task_token_num = torch.zeros(self.num_tasks).to(self.device)
        for task_id in range(self.num_tasks):
            task_token_num[task_id] = torch.sum(mask[task_ids == task_id])
        return task_token_num
    
    def get_mean_loss_wo_zero(self, losses):
        """
        Get the mean loss without zero.
        Input:
            losses: [batch_size, seq_len], losses for each token, 0 for padding tokens.
        Output:
            mean_loss: [1], mean loss without zero.
        """
        if len(losses) == 0:
            return torch.tensor(0.0).to(self.device)
        mask = losses != 0
        mask_sum = torch.sum(mask)
        if mask_sum == 0:
            return torch.tensor(0.0).to(self.device)
        mean_loss = torch.sum(losses) / mask_sum
        return mean_loss
    
    def get_mean_task_losses_wo_zero(self, losses, task_ids):
        """
        Get the mean loss for each task without zero.
        Input:
            losses: [batch_size, seq_len], losses for each token, 0 for padding tokens.
            task_ids: [batch_size], task id for each sample.
        Output:
            mean_task_losses: [num_task], mean loss for each task without zero.
        """
        mean_task_losses = torch.zeros(self.num_tasks).to(self.device)
        for task_id in range(self.num_tasks):
            task_losses = losses[task_ids == task_id]
            mean_task_losses[task_id] = self.get_mean_loss_wo_zero(task_losses)
        return mean_task_losses
    
    def get_weighted_loss_task_balancing(self, mean_task_losses, weights):
        """
        Get the weighted loss by task balancing.
        Input:
            mean_task_losses: [num_task], mean loss for each task without zero.
            weights: [num_task], task weights.
        Output:
            weighted_loss: [1], weighted loss.
        """
        weighted_loss = torch.sum(mean_task_losses * weights) / torch.sum(weights)
        return weighted_loss
    
    def get_weighted_loss_token_balancing(self, losses, weights, task_ids):
        """
        Get the weighted loss with token balancing. LLAMA default.
        Input:
            losses: [batch_size, seq_len], losses for each token, 0 for padding tokens.
            weights: [num_task], task weights.
            task_ids: [batch_size], task id for each sample.
        Output:
            weighted_loss: [1], weighted loss.
        """
        batch_weights = weights[task_ids]
        weighted_losses = (batch_weights * losses.t()).t()

        mask = losses != 0
        weighted_mask = (batch_weights * mask.t()).t()

        weighted_mask_sum = torch.sum(weighted_mask)
        if weighted_mask_sum == 0:
            return torch.tensor(0.0).to(self.device)
        weighted_loss = torch.sum(weighted_losses) / weighted_mask_sum
        return weighted_loss
    
    def set_weights(self, weights):
        self.weights = weights
    
    @abstractmethod
    def get_weighted_loss(self, losses, task_ids, **kwargs):
        """
        Get the weighted loss.
        Input:
            losses: [batch_size, seq_len], losses for each token, 0 for padding tokens.
            task_ids: [batch_size], task id for each sample.
        """
        pass


class EqualWeighting(WeightMethod):
    "EW baseline"
    def __init__(self, num_tasks, device):
        super().__init__(num_tasks, device)
        self.weights = torch.ones(num_tasks).to(device)
        self.name = "EW"

    def get_weighted_loss(self, losses, task_ids, **kwargs):
        weighted_loss = self.get_weighted_loss_token_balancing(losses, self.weights, task_ids)
        return weighted_loss
    

class EqualWeightingTaskBalancing(WeightMethod):
    "TLA baseline"
    def __init__(self, num_tasks, device):
        super().__init__(num_tasks, device)
        self.weights = torch.ones(num_tasks).to(device)
        self.name = "EWTB"

    def get_weighted_loss(self, losses, task_ids, **kwargs):
        mean_task_losses = self.get_mean_task_losses_wo_zero(losses, task_ids)
        weighted_loss = self.get_weighted_loss_task_balancing(mean_task_losses, self.weights)
        return weighted_loss

    
class RandomLossWeighting(WeightMethod):
    "RLW baseline"
    def __init__(self, num_tasks, device):
        super().__init__(num_tasks, device)
        self.weights = torch.ones(num_tasks).to(device)
        self.name = "RLW"

    def get_weighted_loss(self, losses, task_ids, **kwargs):
        self.weights = (self.num_tasks * F.softmax(torch.randn(self.num_tasks), dim=-1)).to(self.device)
        weighted_loss = self.get_weighted_loss_token_balancing(losses, self.weights, task_ids)
        return weighted_loss


class DynamicWeightAverage(WeightMethod):
    """
    DWA baseline
    Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    """
    def __init__(self, num_tasks, device, iteration_window=25, temp=2.0):
        super().__init__(num_tasks, device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = torch.ones((iteration_window * 2, num_tasks)).to(device)
        self.weights = torch.ones(num_tasks).to(device)
        self.name = "DWA"

    def get_weighted_loss(self, losses, task_ids, **kwargs):
        mean_task_losses = self.get_mean_task_losses_wo_zero(losses, task_ids)

        # update costs - fifo
        cost = mean_task_losses.detach()
        self.costs[:-1, :] = self.costs[1:, :].clone()
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window:, :].mean(0) / self.costs[:self.iteration_window, :].mean(0)
            self.weights = (self.num_tasks * torch.exp(ws / self.temp)) / (torch.exp(ws / self.temp)).sum()

        weighted_loss = self.get_weighted_loss_token_balancing(losses, self.weights, task_ids)
        self.running_iterations += 1
        return weighted_loss
    

class ImprovableGapBalancing(WeightMethod):
    "IGB baseline"
    def __init__(self, num_tasks, device, iteration_window=100):
        super().__init__(num_tasks, device)
        self.iteration_window = iteration_window
        self.running_iterations = 0
        self.costs = torch.zeros((iteration_window, num_tasks)).to(device)
        self.base_task_losses = None
        self.weights = torch.ones(num_tasks).to(device)
        self.name = "IGB"
    
    def get_weighted_loss(self, losses, task_ids, **kwargs):
        mean_task_losses = self.get_mean_task_losses_wo_zero(losses, task_ids)

        if self.running_iterations >= self.iteration_window and self.running_iterations < 2 * self.iteration_window:
            self.costs[self.running_iterations - self.iteration_window, :] = mean_task_losses.detach()
        elif self.running_iterations == 2 * self.iteration_window:
            self.base_task_losses = self.costs.mean(0)
        if self.base_task_losses is not None:
            self.weights = self.num_tasks * F.softmax(mean_task_losses.detach() / self.base_task_losses, dim=-1)
        
        weighted_loss = self.get_weighted_loss_token_balancing(losses, self.weights, task_ids)
        self.running_iterations += 1
        return weighted_loss
