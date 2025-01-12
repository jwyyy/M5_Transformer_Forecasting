import pandas as pd

dat_eval = pd.read_csv("dataset/sales_train_evaluation.csv")
dat_valid = pd.read_csv("dataset/sales_train_validation.csv")
dat_price = pd.read_csv("dataset/selling_price_seq.csv")

seq_valid = dat_valid.iloc[:, 6:]
seq_eval = dat_eval.iloc[:, 6:]

# print(dat_valid.iloc[0:3, 0:10])
# print(dat_eval.iloc[:3, :10])

CONST_LEN = 28


def get_predictor(dat):
    n, m = dat.shape
    x = []
    for i in range(n):
        print(i)
        # cnt = (m - start[i]) // CONST_LEN
        x.append(dat.iloc[i, (-8 * CONST_LEN):].tolist())

    output_x = pd.DataFrame(x)
    output_x = pd.concat([dat.iloc[:, :6], output_x], axis=1)
    output_x.set_index("id", inplace=True)
    return output_x


# each row is a seq of CONST_LEN * 4 observations
# we will use X = 1-4 CONST_LEN observations to predict Y = 2 - 5 CONST_LEN observations
# note: rows{dat_valid} == rows{dat_eval}
def create_data_set(dat, start):

    n, m = dat.shape
    # n = 10  # generate a small dataset for testing
    x = []
    cat = []
    for i in range(n):
        print(i)
        cnt = (m - start[i]) // CONST_LEN
        cat_vec = dat.iloc[i, 1:6].tolist()
        for k in range(cnt-7): # 5/8-1
            # print(sum(dat.iloc[i,:].isnull()))
            if k: x.append(dat.iloc[i, (-(k+8)*CONST_LEN):(-k*CONST_LEN)].tolist())
            else: x.append(dat.iloc[i, (-8*CONST_LEN):].tolist())
            cat.append(cat_vec.copy())
    # print(cat[0], len(cat))
    # print(x[0], len(x))
    output_x = pd.concat([pd.DataFrame(cat), pd.DataFrame(x)], axis=1)
    return output_x


if True:
    start_ls = dat_price.iloc[:, -1].tolist()
    x = create_data_set(dat=dat_valid, start=start_ls)
    if x is not None:
        x.to_csv("train_X.csv", index=False)
    else:
        print("Error: x is None.")

if True:
    x_valid = get_predictor(dat=dat_valid)
    x_eval = get_predictor(dat=dat_eval)
    if x_valid is not None:
        x_valid.to_csv("valid_X_pred.csv", header=False)
    else:
        print("Error: x_valid is None.")

    if x_eval is not None:
        x_eval.to_csv("eval_X.csv", header=False)
    else:
        print("Error: x_eval is None.")










