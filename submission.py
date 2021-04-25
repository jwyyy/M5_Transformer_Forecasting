import torch
import pandas as pd
import numpy as np
from trainOps import DataLoader, compute_loss, get_mask, compute_prediction_loss

CONST_LEN = 28


def make_prediction(model, dat, src_m, tar_m, datLoader, device):
    # send model to GPU
    model.to(device)
    model.eval()
    # send tensors to GPU
    src_mask, tar_mask = src_m.to(device), tar_m.to(device)
    mean = torch.Tensor(datLoader.mean).to(device)
    std = torch.Tensor(dataLoader.std_).to(device)
    id = []
    pred = []
    for i, (batch_id, cat, x, y) in enumerate(datLoader.get_submission_batch(dat)):
        # print("prediction mini-batch : ", i)
        id.extend(batch_id)
        cat, x, y = cat.to(device), x.to(device), y.to(device)
        out = model.forward(cat, x, y, src_mask, tar_mask)
        v_out = out.squeeze(1) * std[CONST_LEN:] + mean[CONST_LEN:]
        flag = torch.round(v_out) > 0
        v_out = flag * torch.ceil(v_out)
        v_out = v_out.cpu().numpy()
        v_out = v_out[:, -CONST_LEN:]
        # print(v_out[1, 1:])
        # print("=====")
        pred.extend(v_out.tolist())
    # print(id[:5])
    output = pd.DataFrame(pred, index=id, columns=["F"+str(i) for i in range(1, 29)])
    output.index.name = "id"
    return output


def load_model(checkpoint_path):
    # load model
    # replace x by the epoch number
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


# set up GPU
device = torch.device("cpu")
model = load_model("10_checkpoint.pth")
print("Model loaded ...")

dataLoader = DataLoader('small_X.csv', batch_size=16, cat_exist=True, split=(90, 5, 5))
src_mask, tar_mask = get_mask(4 * CONST_LEN, random=False)

print("validation set loaded ...")
dat = pd.read_csv("valid_X_pred.csv", header=None)
valid_pred = make_prediction(model, dat, src_mask, tar_mask, dataLoader, device)
print("validation set prediction done ...")

print("evaluation set loaded ...")
dat_ = pd.read_csv("eval_X_pred.csv", header=None)
eval_pred = make_prediction(model, dat_, src_mask, tar_mask, dataLoader, device)
print("evaluation set prediction done ...")

output = pd.concat([valid_pred, eval_pred], axis=0)
output.to_csv("prediction_.csv")

