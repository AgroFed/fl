#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import asyncio, copy, os, pickle, socket, sys, time
from functools import partial
from multiprocessing import Pool, Process
from pathlib import Path
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from libs import agg, data, fl, log, nn, poison, resnet, sim


# In[2]:


# Save Logs To File (info | debug | warning | error | critical) [optional]
log.init("info")
#log.init("info", "federated.log")
#log.init("debug", "flkafka.log")


# In[3]:


class FedArgs():
    def __init__(self):
        self.num_clients = 50
        self.epochs = 50
        self.local_rounds = 1
        self.client_batch_size = 32
        self.test_batch_size = 128
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.cuda = False
        self.seed = 1
        self.loop = asyncio.get_event_loop()
        self.tb = SummaryWriter('../../out/runs/federated/FLTC/mn-sine-500*2-dot-5+1-e-50 (test)', comment="Centralized Federated training")

fedargs = FedArgs()


# In[4]:


use_cuda = fedargs.cuda and torch.cuda.is_available()
torch.manual_seed(fedargs.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


# In[5]:


host = socket.gethostname()
clients = [host + "(" + str(client + 1) + ")" for client in range(fedargs.num_clients)]


# In[6]:


# Initialize Global and Client models
#init_model = resnet.ResNet18() #For cifar
init_model = nn.ModelMNIST() #For mnist and f-mnist
global_model = copy.deepcopy(init_model)

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


# In[7]:


# Load MNIST Data to clients
train_data, test_data = data.load_dataset("mnist")
classes = [label + 1 for label in range(10)]


# <h2>Set right parameters for poisoning here before proceeding, else make all False!</h2>

# In[8]:


# FLTrust
FLTrust = {"is": True,
           "ratio": 0.003,
           "data": None,
           "loader": None,
           "proxy": {"is": False,
                     "ratio": 0.5,
                     "data": None,
                     "loader": None}}

# No of malicious clients
mal_clients = [c for c in range(24)]
corrupt = {"is": False,
           "ratio": 0.006,
           "data": None,
           "loader": None}

# Label Flip
label_flip_attack = {"is": False}
label_flip_attack["source_label"] = 4 if label_flip_attack["is"] else None
label_flip_attack["target_label"] = 6 if label_flip_attack["is"] else None

# Layer replacement attack
layer_replacement_attack = {"is": False}

# Cosine attack
cosine_attack = {"is": True,
                 "args": {"poison_percent": 1, 
                          "scale_dot": 5, 
                          "scale_norm": 500}}

# Sybil attack, for sending same update as base
sybil_attack = {"is": False}


# <h2>Prepare a corrupted Model</h2>

# In[9]:


# Flip all the labels to next label
def poison_labels_to_next(data, classes, poison_percent = 1):
    for index, label in enumerate(classes):
        corrupt_data = poison.label_flip(data, label, classes[(index + 1) % len(classes)], poison_percent)
        
    return corrupt_data
    

if corrupt["is"]:
    train_data, corrupt["data"] = data.random_split(train_data, corrupt["ratio"])
    corrupt["data"] = poison_labels_to_next(corrupt["data"], classes)
    corrupt["loader"] = torch.utils.data.DataLoader(corrupt["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)


# <h2>FLTrust</h2>

# In[10]:


if FLTrust["is"]:
    train_data, FLTrust["data"] = data.random_split(train_data, FLTrust["ratio"])
    FLTrust["loader"] = torch.utils.data.DataLoader(FLTrust["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)
    
    if FLTrust["proxy"]["is"]:
        FLTrust["data"], FLTrust["proxy"]["data"] = data.random_split(FLTrust["data"], FLTrust["proxy"]["ratio"])
        FLTrust["loader"] = torch.utils.data.DataLoader(FLTrust["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)
        FLTrust["proxy"]["loader"] = torch.utils.data.DataLoader(FLTrust["proxy"]["data"], batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)


# In[11]:


# Load client's data
clients_data = data.split_data(train_data, clients)


# <h2>Label Flip Attack</h2>

# In[12]:


if label_flip_attack["is"]:
    for client in mal_clients:
        clients_data[clients[client]] = poison.label_flip(clients_data[clients[client]],
                                                          label_flip_attack["source_label"],
                                                          label_flip_attack["target_label"], -1)

        for index, label in enumerate(classes):
            pass#clients_data[clients[client]] = poison_labels_to_next(clients_data[clients[client]], classes, 1)


# In[13]:


client_train_loaders, _ = data.load_client_data(clients_data, fedargs.client_batch_size, None, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=fedargs.test_batch_size, shuffle=True, **kwargs)

client_details = {
        client: {"train_loader": client_train_loaders[client],
                 "model":  copy.deepcopy(global_model),
                 "model_update": None}
        for client in clients
    }


# In[14]:


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def process(client, epoch, model, train_loader, fedargs, device):
    # Train
    model_update, model, loss = train_model(model, train_loader, fedargs, device)

    log.jsondebug(loss, "Epoch {} of {} : Federated Training loss, Client {}".format(epoch, fedargs.epochs, client))
    log.modeldebug(model_update, "Epoch {} of {} : Client {} Update".format(epoch, fedargs.epochs, client))
    
    return model_update


# In[15]:


import time
start_time = time.time()
    
# Federated Training
for _epoch in tqdm(range(fedargs.epochs)):

    epoch = _epoch + 1
    log.info("Federated Training Epoch {} of {}".format(epoch, fedargs.epochs))

    # Gloabal Model Update
    if epoch > 1:
        # For Tmean and FLTrust, not impacts others as of now
        avgargs = {"beta": 10, 
                   "base_model_update": global_model_update if FLTrust["is"] else None,
                   "base_norm": True}
        
        # Average
        client_model_updates = {client: details["model_update"] for client, details in client_details.items()}
        global_model = fl.federated_avg(client_model_updates, global_model, agg.Rule.FLTC, **avgargs)
        log.modeldebug(global_model, "Epoch {} of {} : Server Update".format(epoch, fedargs.epochs))

        # Test
        global_test_output = fl.eval(global_model, test_loader, device,
                                     label_flip_attack["source_label"],
                                     label_flip_attack["target_label"])
        fedargs.tb.add_scalar("Gloabl Accuracy/", global_test_output["accuracy"], epoch)
        if label_flip_attack["is"]:
            fedargs.tb.add_scalar("Attack Success Rate/", global_test_output["attack"]["attack_success_rate"], epoch)
        log.jsoninfo(global_test_output, "Global Test Outut after Epoch {} of {}".format(epoch, fedargs.epochs))
    
        # Update client models
        for client in clients:
            client_details[client]['model'] = copy.deepcopy(global_model)

    # Clients
    tasks = [process(client, epoch, client_details[client]['model'],
                     client_details[client]['train_loader'],
                     fedargs, device) for client in clients]
    try:
        updates = fedargs.loop.run_until_complete(asyncio.gather(*tasks))
    except KeyboardInterrupt as e:
        print("Caught keyboard interrupt. Canceling tasks...")
        tasks.cancel()
        fedargs.loop.run_forever()
        tasks.exception()

    for client, update in zip(clients, updates):
        client_details[client]['model_update'] = update
    
    if FLTrust["is"]:
        global_model_update, _, _ = train_model(global_model, FLTrust["loader"], fedargs, device)

        # For Attacks related to FLTrust
        base_model_update = global_model_update
        if FLTrust["proxy"]["is"]:
            base_model_update, _, _ = train_model(global_model, FLTrust["proxy"]["loader"], fedargs, device)
            
        if layer_replacement_attack["is"]:
            if corrupt["is"]:
                corrupt_model_update, _, _ = train_model(global_model, corrupt["loader"], fedargs, device)
            for client in mal_clients:
                client_details[clients[client]]['model_update'] = poison.layer_replacement_attack(base_model_update,
                                                                                        corrupt_model_update 
                                                                                                  if corrupt["is"] 
                                                                                                  else client_details[clients[client]]['model_update'],
                                                                                        ['conv1.weight'])

        # For cosine attack, Malicious Clients
        if cosine_attack["is"]:
            b_arr, b_list = sim.get_net_arr(base_model_update)
            
            if epoch % 5 == 0:
                cosine_attack["args"]["scale_dot"] = 1 + cosine_attack["args"]["scale_dot"]
                cosine_attack["args"]["scale_norm"] = 2 * cosine_attack["args"]["scale_norm"]

            with Pool(len(mal_clients)) as p:
                func = partial(poison.model_poison_cosine_coord, b_arr, cosine_attack["args"])
                p_models = p.map(func, [sim.get_net_arr(client_details[clients[client]]['model_update'])[0]
                                        for client in mal_clients])
                p.close()
                p.join()


            for client, (p_arr, _) in zip(mal_clients, p_models):
                client_details[clients[client]]['model_update'] = sim.get_arr_net(client_details[clients[client]]['model_update'],
                                                                        p_arr, b_list)
                
            #plot params changed for only one client
            fedargs.tb.add_scalar("Params Changed for Cosine Attack/", p_models[0][1], epoch)

        # For sybil attack, Malicious Clients
        if sybil_attack["is"]:
            for client in mal_clients:
                client_details[clients[client]]['model_update'] = base_model_update

print(time.time() - start_time)


# In[ ]:


nn.ModelMNIST()

