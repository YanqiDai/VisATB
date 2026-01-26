import torch
import torch.nn as nn
import math
import numpy as np
import csv
import torch.nn.functional as F


temperature = 0.5

out_contributions = torch.tensor([0.1637, 0.1903, 0.1952, 0.1433, 0.0768, 0.1316, 0.1582])
out_weights = out_contributions.size(0) * F.softmax(out_contributions / temperature, dim=0)
print(out_weights)

in_contributions = torch.tensor([0.1634, 0.2179, 0.1326, 0.1775, 0.1674, 0.0783, 0.1219])
in_weights = in_contributions.size(0) * F.softmax(-in_contributions / temperature, dim=0)
print(in_weights)

difficulties = torch.tensor([0.7597, 0.5313, 0.8852, 0.7979, 0.5511, 0.8680, 0.1904])
difficulty_weights = difficulties.size(0) * F.softmax((1-difficulties) / temperature, dim=0)
print(difficulty_weights)

print("inter-task contribution weights: ", (out_weights + in_weights) / 2)
print("intra-task difficulty weights: ", difficulty_weights)
print("final weights: ", ((out_weights + in_weights) / 2 + difficulty_weights) / 2)


# vicuna 13b

temperature = 0.5

out_contributions = torch.tensor([0.1885, 0.2192, 0.1561, 0.1390, 0.0409, 0.1519, 0.1248])
out_weights = out_contributions.size(0) * F.softmax(out_contributions / temperature, dim=0)
print(out_weights)

in_contributions = torch.tensor([0.0551, 0.2067, 0.1497, 0.2181, 0.1392, 0.0702, 0.1813])
in_weights = in_contributions.size(0) * F.softmax(-in_contributions / temperature, dim=0)
print(in_weights)

difficulties = torch.tensor([0.6814, 0.5744, 0.8916, 0.8065, 0.5705, 0.8715, 0.1247])
difficulty_weights = difficulties.size(0) * F.softmax((1-difficulties) / temperature, dim=0)
print(difficulty_weights)

print("inter-task contribution weights: ", (out_weights + in_weights) / 2)
print("intra-task difficulty weights: ", difficulty_weights)
print("final weights: ", ((out_weights + in_weights) / 2 + difficulty_weights) / 2)


# llava v1

temperature = 0.5

out_contributions = torch.tensor([0.7601, 0.3496, 1.6727])
out_weights = out_contributions.size(0) * F.softmax(out_contributions / temperature, dim=0)
print(out_weights)

in_contributions = torch.tensor([1.2890, 0.9779, 0.5156])
in_weights = in_contributions.size(0) * F.softmax(-in_contributions / temperature, dim=0)
print(in_weights)

difficulties = torch.tensor([0.8636, 0.3534, 0.9593])
difficulty_weights = difficulties.size(0) * F.softmax((1-difficulties) / temperature, dim=0)
print(difficulty_weights)

print("inter-task contribution weights: ", (out_weights + in_weights) / 2)
print("intra-task difficulty weights: ", difficulty_weights)
print("final weights: ", ((out_weights + in_weights) / 2 + difficulty_weights) / 2)

