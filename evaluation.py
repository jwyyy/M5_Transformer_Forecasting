import pandas as pd
import numpy as np

pred = pd.read_csv("eval_dataset/prediction_.csv", header=None, low_memory=False)
true = pd.read_csv("eval_dataset/sales_test_evaluation.csv", header=None, low_memory=False)

pred2 = pred.iloc[30491:, 1:]
true2 = true.iloc[1:, 5:]

print(pred2.shape, true2.shape)

err = []

for i in range(30490):
    d = np.abs(pred2.iloc[i, :].astype("float") - true2.iloc[i, :].astype("float"))
    err.append(np.mean(d))

print(np.mean(err))