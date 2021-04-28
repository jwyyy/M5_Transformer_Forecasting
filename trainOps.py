import torch
import pandas as pd
import numpy as np
from numpy.random import randint

CONST_LEN = 28
CONST_CAT_DIM = 3049+7+3+10+3


def check_tensor(tensor_ls):
    for ts in tensor_ls:
        if torch.isnan(ts).sum():
            raise Exception("tensor is nan!")
        elif torch.isinf(ts).sum():
            raise Exception("tensor is inf")
    return "okay"


def compute_loss(y, pred, scale, mask):
    # print(y.size())
    # print(pred.size())
    diff = (y - pred) * scale
    batch, seq_len = diff.size()
    if mask is None:
        return torch.sum(diff**2) / (batch * seq_len)
    mask = 1 - mask.unsqueeze(-1)
    return torch.sum(torch.matmul(diff**2, mask)) / (batch * torch.sum(mask))


def compute_prediction_loss(y, pred, scale, mask):
    # print(y.size())
    # print(pred.size())
    diff = torch.abs(y - pred) * scale
    batch, seq_len = diff.size()
    if mask is None:
        return torch.sum(diff) / (batch * seq_len)
    mask = 1 - mask.unsqueeze(-1)
    return torch.sum(torch.matmul(diff, mask)) / (batch * torch.sum(mask))


def get_mask(seq_len=4*CONST_LEN, random=False):
    src_mask = [1]*seq_len
    tar_mask = [1]*seq_len
    if not random:
        tar_mask[-CONST_LEN:] = [0] * CONST_LEN
    else:
        pos = randint(0, seq_len-1)
        tar_mask[pos:] = [0] * (seq_len - pos)
    return torch.Tensor(src_mask), torch.Tensor(tar_mask)


def create_small_dataset(data_file, csv_name="small_X.csv", size=1000):
    dat = pd.read_csv(data_file)
    n, _ = dat.shape
    # categorical variables, numerical variables
    cat = dat.iloc[:, :5]
    cat = pd.concat([pd.get_dummies(cat.iloc[:, j]) for j in range(5)], axis=1)
    # print(cat.shape)
    cat_x = pd.concat([cat.iloc[:size, :], dat.iloc[:size, 5:]], axis=1)
    cat_x.to_csv(csv_name, index=False)
    print("A small dataset was created!")


class DataLoader:
    def __init__(self, data_file, batch_size=10, cat_exist=False, split=(8, 1, 1), random_seed=10234):

        dat = pd.read_csv(data_file)
        self.n, _ = dat.shape
        # print("dataset size : ", self.n)
        self.batch_size = batch_size
        self.batch = self.n // batch_size
        self.train_n = round(self.batch * split[0] / sum(split))
        # print("train_n = ", self.train_n)
        self.valid_n = round(self.batch * split[1] / sum(split))
        self.test_n = self.batch - self.train_n - self.valid_n
        # random shuffle dataset rows
        # then do train/valid/test split
        # categorical variables, numerical variables
        # set a random_seed to memorize train/valid/test split
        np.random.seed(random_seed)
        _order = list(range(self.n))
        np.random.shuffle(_order)
        dat = dat.iloc[_order, :]
        if not cat_exist:
            cat, self.dat = dat.iloc[:, :5], dat.iloc[:, 5:]
            self.cat = pd.concat([pd.get_dummies(cat.iloc[:, j]) for j in range(5)], axis=1)
        else:
            self.cat, self.dat = dat.iloc[:, :CONST_CAT_DIM], dat.iloc[:, CONST_CAT_DIM:]

        assert self.cat.shape[1] == CONST_CAT_DIM

        self.train_dat_ = self.dat.iloc[:self.train_n*batch_size, :]
        self.valid_dat_ = self.dat.iloc[(self.train_n * batch_size):(self.train_n + self.valid_n) * batch_size, :]
        self.test_dat_ = self.dat.iloc[(self.train_n + self.valid_n) * batch_size:, :]
        # scaling
        mu = self.train_dat_.mean(axis=0)
        self.mu = mu.tolist()
        scale = self.train_dat_.std(axis=0)
        scale.replace(0, 1.0, inplace=True)
        self.scale = scale.tolist()

        # training dataset
        self.train_dat = (self.train_dat_ - mu) / scale
        self.train_cat = self.cat.iloc[:(self.train_n*batch_size), :]
        # validation dataset
        self.valid_dat = (self.valid_dat_ - mu) / scale
        self.valid_cat = self.cat.iloc[(self.train_n*batch_size):(self.train_n + self.valid_n)*batch_size, :]
        # test dataset
        self.test_dat = (self.test_dat_ - mu) / scale
        self.test_cat = self.cat.iloc[(self.train_n + self.valid_n)*batch_size:, :]
        # print(self.train_n, self.valid_n, self.test_n)

    def shuffle(self):
        # training dataset
        train_size = self.train_dat.shape[0]
        new_order = list(range(train_size))
        np.random.shuffle(new_order)
        self.train_dat = self.train_dat.iloc[new_order, :]
        self.train_cat = self.train_cat.iloc[new_order, :]

    def get_training_batch(self):

        for i in range(1, self.train_n):
            l = self.train_cat.iloc[((i-1)*self.batch_size):(i*self.batch_size), :]
            x = self.train_dat.iloc[((i-1)*self.batch_size):(i*self.batch_size), :-CONST_LEN]
            y = self.train_dat.iloc[((i-1)*self.batch_size):(i*self.batch_size), CONST_LEN:]
            # print(l.shape, x.shape, y.shape)
            yield torch.Tensor(l.to_numpy()), torch.Tensor(x.to_numpy()), torch.Tensor(y.to_numpy())

    def get_validation_batch(self):

        for i in range(1, self.valid_n):
            l = self.valid_cat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :]
            x = self.valid_dat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :-CONST_LEN]
            y = self.valid_dat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), CONST_LEN:]
            # print(l.shape, x.shape, y.shape)
            yield torch.Tensor(l.to_numpy()), torch.Tensor(x.to_numpy()), torch.Tensor(y.to_numpy())

    def get_test_batch(self):

        for i in range(1, self.test_n):
            l = self.test_cat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :]
            x = self.test_dat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :-CONST_LEN]
            y = self.test_dat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), CONST_LEN:]
            # print(l.shape, x.shape, y.shape)
            yield torch.Tensor(l.to_numpy()), torch.Tensor(x.to_numpy()), torch.Tensor(y.to_numpy())

    def get_original_test_batch(self):

        for i in range(1, self.test_n):
            l = self.test_cat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :]
            x = self.test_dat_.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :-CONST_LEN]
            y = self.test_dat_.iloc[((i - 1) * self.batch_size):(i * self.batch_size), CONST_LEN:]
            # print(l.shape, x.shape, y.shape)
            yield torch.Tensor(l.to_numpy()), torch.Tensor(x.to_numpy()), torch.Tensor(y.to_numpy())

    def get_submission_batch(self, dat):
        n, _ = dat.shape
        # get row id
        dat_id = dat.iloc[:, 0].to_list()
        # print(dat_id[:5])
        # get row categorical variables
        dat_cat = dat.iloc[:, 1:6]
        # convert categorical variables to dummies
        cat = pd.concat([pd.get_dummies(dat_cat.iloc[:, j]) for j in range(5)], axis=1)
        # standardize x
        x = (dat.iloc[:, 6:] - self.mu[:-CONST_LEN]) / self.scale[:-CONST_LEN]
        # create y
        dat_y = x.iloc[:, CONST_LEN:]
        y = pd.concat([dat_y, pd.DataFrame(np.zeros((n, 28)))], axis=1)
        i = 0
        while i < n:
            end = min(i + self.batch_size, n)
            yield dat_id[i:end], torch.Tensor(cat.iloc[i:end, :].to_numpy()), \
                  torch.Tensor(x.iloc[i:end, :].to_numpy()), torch.Tensor(y.iloc[i:end, :].to_numpy())
            i = end + 1



