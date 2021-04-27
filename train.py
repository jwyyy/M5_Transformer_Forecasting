import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from Model import Transformer
from trainOps import DataLoader, get_mask, create_small_dataset, check_tensor, compute_prediction_loss, compute_loss

# torch.autograd.set_detect_anomaly(True)
# set up GPU
device = torch.device("cuda:0")

# training configuration
epoch = 75
save_model_every = 25
data_split = (96, 2, 2)
random_mask = False
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
dropout = 0.3
model = Transformer(seq_len, channels, conv_k, dropout)
# send model to GPU
model.to(device)

# training code
loss_train_history = []
loss_valid_history = []

optimizer = Adam(model.parameters(), lr=3e-4)
# create_small_dataset(data_file="valid_X.csv", csv_name="small_X.csv")
dataLoader = DataLoader(data_input, batch_size, cat_exist, data_split)
scale = torch.Tensor(dataLoader.scale)
scale = scale.to(device)

for k in range(epoch):

    if k and k % save_model_every == 0:
        checkpoint = {'model': Transformer(seq_len, channels, conv_k, dropout),
                      'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict()}
        torch.save(checkpoint, str(k)+'_'+'checkpoint.pth')

    loss_train = []
    loss_valid = []
    dataLoader.shuffle()
    model.train()
    # set model training state
    for i, (cat, src, tar) in enumerate(dataLoader.get_training_batch()):
        src_mask, tar_mask = get_mask(4 * CONST_LEN, random_mask)
        # send src_mask, tar_mask to GPU
        src_mask, tar_mask = src_mask.to(device), tar_mask.to(device)
        # send tensors to GPU
        cat, src, tar = cat.to(device), src.to(device), tar.to(device)
        out = model.forward(cat, src, tar, src_mask, tar_mask)
        loss = compute_prediction_loss(out, tar, scale[CONST_LEN:], tar_mask)

        # record training loss history
        loss_train.append(loss.item())

        # update parameters using backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("training mini-batch ", i,
                  "loss =", loss.item())

    loss_train_history.append(np.mean(loss_train))

    print("last training mini-batch loss  =", loss_train[-1])

    # model evaluation mode

    with torch.no_grad():
        model.eval()
        for i, (cat, x, y) in enumerate(dataLoader.get_validation_batch()):
            # get validation masks
            v_src_mask, v_tar_mask = get_mask(4 * CONST_LEN, random=random_mask)
            # send src_mask, tar_mask to GPU
            valid_src_mask, valid_tar_mask = v_src_mask.to(device), v_tar_mask.to(device)
            cat, x, y = cat.to(device), x.to(device), y.to(device)
            valid_y = model.forward(cat, x, y, valid_src_mask, valid_tar_mask)
            valid_loss = compute_prediction_loss(valid_y, y, scale[CONST_LEN:], valid_tar_mask)
            loss_valid.append(valid_loss.item())

    if len(loss_valid_history) and np.mean(loss_valid) < loss_valid_history[-1]:
        checkpoint = {'model': Transformer(seq_len, channels, conv_k, dropout),
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, 'best_' + 'checkpoint.pth')

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


