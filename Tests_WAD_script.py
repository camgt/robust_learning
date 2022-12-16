# %% [markdown]
# # Wasserstein Ascend Descend
# ### (C. and N.) Garcia Trillos


# %%
import sys
import numpy as np
from Robust_nn.WAD import WAD2scale
import matplotlib.pyplot as plt
from utils.utils import read_vision_dataset
from utils.convnet import ConvNet
from utils.convnet_silu import ConvNetSiLU
import os
import torch
import gc
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR


# %% [markdown]
# **Read the DataLoader**

# %%
# Create some networks

# n_nets = 5
n_nets = 4
net_lst = [ConvNet() for i in range(n_nets)]
avg_nets =[ConvNet() for i in range(n_nets*4)]

# adv_net = WAD2scale(net_list = net_lst, avg_nets = avg_nets, 
#                     dataset_name='MNIST',batch_size = 1024, 
#                     device = None , criterion= nn.CrossEntropyLoss(), 
#                     scale_factor=5, num_adverse=2,
#                     kappa = { 'param': 0.2, 'adv': 0.2   },
#                     max_batches= 5)


# adv_net = WAD2scale(net_list = net_lst, avg_nets = avg_nets, 
#                     dataset_name='MNIST',batch_size = 64, 
#                     device = None , criterion= nn.CrossEntropyLoss(), 
#                     scale_factor=3, num_adverse=2,
#                     penalty_coef = 3,
#                     kappa = { 'param': 0.25, 'adv': 0.15  },
#                     max_batches= 80)



                    
adv_net = WAD2scale(net_list = net_lst, avg_nets = avg_nets, 
                    dataset_name='MNIST',batch_size = 64, 
                    device = None , criterion= nn.CrossEntropyLoss(), 
                    scale_factor=5, num_adverse=2,
                    penalty_coef = 10,
                    kappa = { 'param': 0.25, 'adv': 0.25  },
                    max_batches= 5)

# %%
import os
os.listdir('.')

# %%
adv_net.set_optimizer()
adv_net.train(epochs=1)

# %%
adv_net.save_model('Test1_kappaB')

# %%
# adv_net.set_optimizer()
# adv_net.import_model('Test1_kappaB')

# %%
'''
Testing the model 

Peform the following tests:
1. Test for accuracy of the average  model
2. Train directly the model with the whole set of adversaries. Calculate...
3. Create directly the adversarial. 
'''

# %%
adv_net.test_pgd(20)

# %%
# adv_net.test_base()

# %%
val_at_nash = adv_net.test_base()

# %%
val_better_adv = adv_net.test_improve_adversaries(5)

# %%
val_better_model = adv_net.test_improve_model(5)

# %%
#Check the order

print(val_better_model<=val_at_nash<=val_better_adv)

# %%
# Percentage of change

print( 'Relative Interval:', 1 - val_better_model/val_at_nash, val_better_adv/val_at_nash -1    )

# %%
type(val_better_model)

# %%
aux = next(iter(adv_net.trainloader))

# %%
print(aux[1][0])
plt.imshow( np.squeeze(aux[0])[0,:,:])


# %%



