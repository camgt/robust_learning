# %% [markdown]
# # Wasserstein Ascend Descend
# ### (C. and N.) Garcia Trillos

# %%
# %load_ext autoreload
# %autoreload 2

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
n_nets = 2
net_lst = [ConvNet() for i in range(n_nets)]
avg_nets =[ConvNet() for i in range(n_nets*4)]

# adv_net = WAD2scale(net_list = net_lst, avg_nets = avg_nets, 
#                     dataset_name='MNIST',batch_size = 1024, 
#                     device = None , criterion= nn.CrossEntropyLoss(), 
#                     scale_factor=5, num_adverse=2,
#                     kappa = { 'param': 0.2, 'adv': 0.2   },
#                     max_batches= 5)


adv_net = WAD2scale(net_list = net_lst, avg_nets = avg_nets, 
                    dataset_name='MNIST',batch_size = 128, num_workers=0,
                    device = None , criterion= nn.CrossEntropyLoss(), 
                    scale_factor=2, num_adverse=2,
                    kappa = { 'param': 0.2, 'adv': 0.2   },
                    max_batches= 2)

# %%
import os
os.listdir('.')

# %%
adv_net.set_optimizer()
adv_net.train(epochs = 10)

# %%
adv_net.save_model('Test1_kappa0p2-4')

# %%
adv_net.set_optimizer()
adv_net.import_model('Test1_kappa0p2-4')

# %%
'''
Testing the model 

Peform the following tests:
1. Test for accuracy of the average  model
2. Train directly the model with the whole set of adversaries. Calculate...
3. Create directly the adversarial. 
'''

# %%
#adv_net.test_pgd(5)

# %%
adv_net.test_improve_model(5)


# %%
adv_net.test_improve_adversaries(5)

'''

# %% [markdown]
# **Creating and comparing results**

# %%


# %%
# options = {'only o1':(True,False, False), 'only o2':(False, True, False), 'both': (True,True, False), 'none':(False,False,False), 'modO2':(False,True,True) } 
options = {'only o1':(True,False, False), 'both': (True,True, True), 'none':(False,False,False) } 
rvec = [2,4,np.inf]
deltav = [0.2]
mdict = {}
basepath = os.curdir
mpath = os.path.join(basepath,'models')

for i,(k,(o1,o2, mod_o2)) in enumerate(options.items()):
  auxd = {}
  for r in rvec:
    rstr = str(r)
    print('\n*******\n ** Case '+k+', r=',r,'\n ******')
    torch.manual_seed(0)
    np.random.seed(0)
    auxd[rstr] = {}
    network = ConvNet()
    net_RegTrin = OrdTwoL(network, trainloader, testloader,  device='cuda', delta =0.2, r= r, o2=o2, o1=o1, mod_o2=mod_o2) #'cuda'
    net_RegTrin.set_optimizer(optim_alg='Adam', args={'lr':1e-4})
    net_RegTrin.train(epochs=5, delta=deltav)
    auxd[rstr]['train_loss'] = net_RegTrin.train_loss.copy()
    auxd[rstr]['train_acc'] = net_RegTrin.train_acc.copy()
    auxd[rstr]['train_reg'] = net_RegTrin.train_reg.copy()
    auxd[rstr]['test_acc_adv'] = net_RegTrin.test_acc_adv.copy()
    auxd[rstr]['test_acc_clean'] = net_RegTrin.test_acc_clean.copy()
    auxd[rstr]['train_times'] = net_RegTrin.train_times.copy()
    torch.save(network.state_dict(), os.path.join(mpath, k+'_r_'+str(r)+'.pth' ))
    del(network)
    gc.collect()
    torch.cuda.empty_cache()
  mdict[k] = auxd
  with open('tests_all.txt','w') as data: 
      data.write(str(mdict))


'''