# %%
import numpy as np
import torch as t
import torch.nn.modules as nn
import pandas as pd
import torch.optim as optim

df = pd.read_csv("hf://datasets/syncora/developer-productivity-simulated-behavioral-data/Developer_Productivity_Synthetic_Syncora.csv")

# %%
df

# %%
X=df.iloc[:,:-1].to_numpy(dtype=np.float32)
Y=df.iloc[:,-1].to_numpy(dtype=np.float32)

# %%
Y
X

# %%
Y=t.from_numpy(Y)
X=t.from_numpy(X)


# %%
X.dtype
Y=Y.reshape(-1,1)

