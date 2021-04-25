import torch
import numpy as np
from trainOps import DataLoader, compute_loss, get_mask, compute_prediction_loss

# set up GPU
device = torch.device("cuda:1")

# model configuration
CONST_LEN = 28

# load model
# replace x by the epoch number
checkpoint = torch.load('x_checkpoint.pth')
model = checkpoint["model"]
model.load_state_dict(checkpoint["state_dict"])
for parameter in model.parameters():
    parameter.requires_grad = False
# send model to GPU
model.to(device)
model.eval()

# set random seed if used a non default value
dataLoader = DataLoader('valid_X.csv', batch_size=512, cat_exist=False, split=(90, 2, 2))
src_mask, tar_mask = get_mask(4 * CONST_LEN, random=False)
# send src_mask, tar_mask to GPU
src_mask, tar_mask = src_mask.to(device), tar_mask.to(device)
loss_test = []
pred_y = []
mean = torch.Tensor(dataLoader.mean).to(device)
std = torch.Tensor(dataLoader.std_).to(device)
for i, (cat, x, y) in enumerate(dataLoader.get_test_batch()):
    # print("test mini-batch ", i)
    # send tensors to GPU
    cat, x, y = cat.to(device), x.to(device), y.to(device)
    test_y = model.forward(cat, x, y, src_mask, tar_mask)
    pred_y.append(test_y.squeeze(1) * std[CONST_LEN:] + mean[CONST_LEN:])
    test_loss = compute_loss(test_y, y, tar_mask)
    loss_test.append(test_loss.item())

print("Standardized test dataset loss : ", np.mean(loss_test))


loss_pred = []
for i, (cat, x, y) in enumerate(dataLoader.get_original_test_batch()):
    # print("test mini-batch ", i)
    # send tensors to GPU
    _, _, y = cat.to(device), x.to(device), y.to(device)
    flag = torch.round(pred_y[i]) > 0
    y_ = flag * torch.ceil(pred_y[i])
    loss_pred.append(compute_prediction_loss(y_, y, tar_mask).item())
    if i % 24 == 0:
        print("A few checks ...")
        print(pred_y[i][1, -CONST_LEN:])
        print(y[1, -CONST_LEN:])
        print(y_[1, -CONST_LEN:])
        print("============================")

print("Original test dataset loss : ", np.mean(loss_pred))

