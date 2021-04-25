import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from Model import Transformer
from trainOps import DataLoader, get_mask, create_small_dataset, check_tensor, compute_prediction_loss

# torch.autograd.set_detect_anomaly(True)
# set up GPU
device = torch.device("cuda:0")

# training configuration
epoch = 100
save_model_every = 25
data_split = (96, 2, 2)
random_mask = True
if False:
    data_input = 'small_X.csv'
    batch_size = 16
    cat_exist = True
else:
    data_input = 'valid_X.csv'
    batch_size = 512
    cat_exist = False
    print("training on the full dataset ...")


# model configuration
CONST_LEN = 28
seq_len = 28 * 4
channels = [8, 8, 8, 8]
conv_k = 5
dropout = 0.5
model = Transformer(seq_len, channels, conv_k, dropout)
# send model to GPU
model.to(device)

# training code
loss_train_history = []
loss_valid_history = []

optimizer = Adam(model.parameters(), lr=3e-4)
# create_small_dataset(data_file="valid_X.csv", csv_name="small_X.csv")
dataLoader = DataLoader(data_input, batch_size, cat_exist, data_split)

v_src_mask, v_tar_mask = get_mask(4 * CONST_LEN, random=False)
# send src_mask, tar_mask to GPU
valid_src_mask, valid_tar_mask = v_src_mask.to(device), v_tar_mask.to(device)

for k in range(epoch):

    if k and k % save_model_every == 0:
        checkpoint = {'model': Transformer(seq_len, channels, conv_k, dropout),
                      'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict()}
        torch.save(checkpoint, str(k)+'_'+'checkpoint.pth')

    loss_train = []
    dataLoader.shuffle()
    # set model training state
    model.train()
    for i, (cat, src, tar) in enumerate(dataLoader.get_training_batch()):
        src_mask, tar_mask = get_mask(4 * CONST_LEN, random_mask)
        # send src_mask, tar_mask to GPU
        src_mask, tar_mask = src_mask.to(device), tar_mask.to(device)
        # print("train mini-batch ", i)
        # send tensors to GPU
        # print("train - check input: ", check_tensor([cat, src, tar]))
        cat, src, tar = cat.to(device), src.to(device), tar.to(device)
        # print(src.size())
        out = model.forward(cat, src, tar, src_mask, tar_mask)
        # print("train - check out: ", check_tensor([out]))
        loss = compute_prediction_loss(out, tar, None)

        # record training loss history
        loss_train.append(loss.item())

        # update parameters using backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train_history.append(np.mean(loss_train))

    # model evaluation mode
    loss_valid = []
    with torch.no_grad():
        model.eval()
        for i, (cat, x, y) in enumerate(dataLoader.get_validation_batch()):
            # print("validation mini-batch ", i)
            # send tensors to GPU
            # print("validation - check input: ", check_tensor([cat, src, tar]))
            cat, x, y = cat.to(device), x.to(device), y.to(device)
            valid_y = model.forward(cat, x, y, valid_src_mask, valid_tar_mask)
            valid_loss = compute_prediction_loss(valid_y, y, None)
            # print("valid - check out: ", check_tensor([valid_loss]))
            loss_valid.append(valid_loss.item())

    loss_valid_history.append(np.mean(loss_valid))

    print("epoch:", k,
          "training loss = ", loss_train_history[-1],
          "validation loss = ", loss_valid_history[-1])

plt.plot(list(range(1, epoch+1)), loss_train_history, label='train')
plt.plot(list(range(1, epoch+1)), loss_valid_history, label='valid')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.legend()
plt.savefig('loss_plot.png')


