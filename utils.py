import numpy as np
import torch
import matplotlib.pyplot as plt
import os.path as osp

from torch import nn

# AutoRec 
class AutoRec(nn.Module) :
    def __init__(self, d, k, dropout_rate) :
        super().__init__()
        self.encoder = nn.Linear(d,k)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(k)  # Added a  normalization layer
        self.decoder = nn.Linear(k,d)

    def forward(self, x):
        encoded = torch.sigmoid(self.encoder(x))
        encoded = self.dropout(encoded)
        encoded = self.batch_norm(encoded)
        decoded = self.decoder(encoded)
        return decoded

# Useful functions

def load_ratings(path, num_users, num_items) :

    fp = open(osp.join(path, "ratings.dat"))
    ratings = np.zeros((num_users,num_items))
    mask_ratings = np.zeros((num_users, num_items))

    lines = fp.readlines()
    for line in lines:
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        ratings[user_idx, item_idx] = int(rating)
        mask_ratings[user_idx, item_idx] = 1
    
    return ratings, mask_ratings

def compute_rmse(pred, true, mask):
    assert pred.shape == true.shape
    loss = torch.sum(((pred - true) ** 2) * mask) / torch.sum(mask)
    return torch.sqrt(loss)

def compute_mse(pred, true, mask) :
    loss = torch.sum(((pred - true) ** 2) * mask) / torch.sum(mask)
    return loss

def get_mask(ratings):
    mask = ~torch.isnan(ratings) 
    ratings = torch.nan_to_num(ratings, nan=0.0)
    return ratings, mask.float()

def adjust_outliers(tensor, low_threshold, high_treshold) :
    tensor[tensor>high_treshold] = high_treshold
    tensor[tensor<low_threshold] = low_threshold
    return tensor

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def split_matrix(matrix, ratio=0.9):
    new_train = np.copy(matrix)
    new_test = np.full(matrix.shape, np.nan)
    
    for i in range(matrix.shape[0]):
        ratings = np.where(~np.isnan(matrix[i]))[0]
        if len(ratings) > 2:
            np.random.shuffle(ratings)
            split_index = int(ratio * len(ratings))
            new_test[i, ratings[split_index:]] = matrix[i, ratings[split_index:]]
            new_train[i, ratings[split_index:]] = np.nan
            
    return new_train, new_test

