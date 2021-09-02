#!/usr/bin/env python
# coding: utf-8

# In[389]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import os, sys
import copy
import socket
from tqdm import tqdm
import torch
import pickle
from torch import optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from mxnet import nd as mnd
import heapq

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from libs import fl, nn, agg, data, poison, log, sim


# In[390]:


# Save Logs To File (info | debug | warning | error | critical) [optional]
log.init("debug")
#log.init("info", "federated.log")
#log.init("debug", "flkafka.log")


# In[391]:


class FedArgs():
    def __init__(self):
        self.num_clients = 50
        self.epochs = 10
        self.local_rounds = 1
        self.client_batch_size = 32
        self.test_batch_size = 128
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.cuda = True
        self.seed = 1
        self.tb = SummaryWriter('../../out/runs/federated/FLTrust', comment="Mnist Centralized Federated training")

fedargs = FedArgs()


# In[392]:


use_cuda = fedargs.cuda and torch.cuda.is_available()
torch.manual_seed(fedargs.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


# In[393]:


print(torch.cuda.is_available())


# In[394]:


host = socket.gethostname()
clients = [host + "(" + str(client + 1) + ")" for client in range(fedargs.num_clients)]


# In[395]:


#Initialize Global and Client models
global_model = nn.ModelMNIST()
client_models = {client: copy.deepcopy(global_model) for client in clients}

# Function for training
def train_model(_model, train_loader, fedargs, device):
    model, loss = fl.client_update(_model,
                                train_loader,
                                fedargs.learning_rate,
                                fedargs.weight_decay,
                                fedargs.local_rounds,
                                device)
    model_update = agg.sub_model(_model, model)
    return model_update, model, loss


# In[396]:


# Load MNIST Data to clients
train_data, test_data = data.load_dataset("mnist")


# In[397]:


# For FLTrust
#############Skip this section for running other averaging
FLTrust = True
root_ratio = 0.002
train_data, root_data = torch.utils.data.random_split(train_data, [int(len(train_data) * (1-root_ratio)), 
                                                              int(len(train_data) * root_ratio)])
root_loader = torch.utils.data.DataLoader(root_data, batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)
print(len(train_data), len(root_data))

#global_model, _ = train_model(global_model, root_loader, fedargs, device)
#client_models = {client: copy.deepcopy(global_model) for client in clients}
#############


# In[398]:


clients_data = data.split_data(train_data, clients)


# In[399]:


# Poison a client
################Skip this section for running without poison
#for client in range(10):
#    clients_data[clients[client]] = poison.label_flip(clients_data[clients[client]], 4, 9, poison_percent = -1)
    
#clients_data[clients[0]] = poison.label_flip(clients_data[clients[0]], 6, 2, poison_percent = 1)
#clients_data[clients[0]] = poison.label_flip(clients_data[clients[0]], 3, 8, poison_percent = 1)
#clients_data[clients[0]] = poison.label_flip(clients_data[clients[0]], 1, 5, poison_percent = 1)


# In[400]:


client_train_loaders, _ = data.load_client_data(clients_data, fedargs.client_batch_size, None, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=fedargs.test_batch_size, shuffle=True, **kwargs)

clients_info = {
        client: {"train_loader": client_train_loaders[client]}
        for client in clients
    }


# In[439]:


global_model = nn.ModelMNIST()
B1 = copy.deepcopy(global_model)
C1 = copy.deepcopy(global_model)
C2 = copy.deepcopy(global_model)

C3 = copy.deepcopy(global_model)
C4 = copy.deepcopy(global_model)
C5 = copy.deepcopy(global_model)

def copy_model(B1, C1, C2, C3, C4, C5, _model):
    B1 = copy.deepcopy(_model)
    C1 = copy.deepcopy(_model)
    C2 = copy.deepcopy(_model)

    C3 = copy.deepcopy(_model)
    C4 = copy.deepcopy(_model)
    C5 = copy.deepcopy(_model)
    
    return B1, C1, C2, C3, C4, C5

B1, C1, C2, C3, C4, C5 = copy_model(B1, C1, C2, C3, C4, C5, global_model)


# In[440]:


t1 = global_model

for i in range(5):
    _B1, B1, _ = train_model(B1, root_loader, fedargs, device)
    print(fl.eval(B1, test_loader, device))

    _C1, C1, _ = train_model(C1, clients_info[list(clients_info.keys())[0]]['train_loader'], fedargs, device)
    _C2, C2, _ = train_model(C2, clients_info[list(clients_info.keys())[1]]['train_loader'], fedargs, device)

    _C3, C3, _ = train_model(C3, clients_info[list(clients_info.keys())[2]]['train_loader'], fedargs, device)
    _C4, C4, _ = train_model(C4, clients_info[list(clients_info.keys())[3]]['train_loader'], fedargs, device)
    _C5, C5, _ = train_model(C5, clients_info[list(clients_info.keys())[4]]['train_loader'], fedargs, device)
    
    fb1, bslist = sim.get_net_arr(_B1)
    fc1, cslist = sim.get_net_arr(_C1)
    fc4, cslist = sim.get_net_arr(_C4)
    fc5, cslist = sim.get_net_arr(_C5)

    np5 = (fc5 - fb1)
    fc11 = copy.deepcopy(fc5)
    for index in heapq.nlargest(int(1199882), range(len(np5)), np5.take):
        #fc11[index] = fc11[index] + (2* np5[index])
        fc11[index] = sim.cosine_coord_vector(fb1, fc11, index)
    print("Here1")
    #np.random.shuffle(np5)

    #V1                       V2
    #1, 2, 3, 4.              2, 4, 6, 8
    #CS = 1

    #np5 = 1, 2, 3, 4
    #np5.shuffle = 3, 4, 1, 2

    #V3 = 4, 6, 4, 6
    #CS = .87

    #fc11 = fb1 + np5
    #cs11 = mnd.dot(mnd.array(fb1), mnd.array(fc11)) / (mnd.norm(mnd.array(fb1)) + 1e-9) / (mnd.norm(mnd.array(fc11)) + 1e-9)
    #print(cs11)

    fc11 = sim.get_arr_net(_C1, fc11, cslist)
    #print(sim.grad_cosine_similarity(_B1, fc11))

    #np.random.shuffle(np5)

    #fc22 = fb1 + np5
    #cs22 = mnd.dot(mnd.array(fb1), mnd.array(fc22)) / (mnd.norm(mnd.array(fb1)) + 1e-9) / (mnd.norm(mnd.array(fc22)) + 1e-9)
    #print(cs22)
    
    np5 = (fc4 - fb1)
    fc22 = copy.deepcopy(fc4)
    for index in heapq.nlargest(int(1199882), range(len(np5)), np5.take):
        #fc22[index] = fc22[index] + (2* np5[index])
        fc22[index] = sim.cosine_coord_vector(fb1, fc22, index)

    print("Here2")
    fc22 = sim.get_arr_net(_C2, fc22, cslist)
    #print(sim.grad_cosine_similarity(_B1, fc22))

    avgargs = {"base_update": _B1}
    global_model = fl.federated_avg({'a': fc11, 'b': fc22, 'c': _C3, 'd': _C4, 'e': _C5}, B1, agg.Rule.FLTrust, **avgargs)
    
    B1, C1, C2, C3, C4, C5 = copy_model(B1, C1, C2, C3, C4, C5, global_model)
    
    print(fl.eval(global_model, test_loader, device))