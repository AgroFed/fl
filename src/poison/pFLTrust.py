#!/usr/bin/env python
# coding: utf-8

# In[46]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import asyncio, copy, os, pickle, socket, sys, time
from pathlib import Path
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from libs import fl, nn, agg, data, poison, log


# In[47]:


# Save Logs To File (info | debug | warning | error | critical) [optional]
log.init("info")
#log.init("info", "federated.log")
#log.init("debug", "flkafka.log")


# In[48]:


class FedArgs():
    def __init__(self):
        self.num_clients = 50
        self.epochs = 25
        self.local_rounds = 1
        self.client_batch_size = 32
        self.test_batch_size = 128
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.cuda = False
        self.seed = 1
        self.loop = asyncio.get_event_loop()
        self.tb = SummaryWriter('../../out/runs/federated/FLTrust(ACC)', comment="Mnist Centralized Federated training")

fedargs = FedArgs()


# In[49]:


use_cuda = fedargs.cuda and torch.cuda.is_available()
torch.manual_seed(fedargs.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


# In[50]:


host = socket.gethostname()
clients = [host + "(" + str(client + 1) + ")" for client in range(fedargs.num_clients)]


# In[51]:


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


# In[52]:


# Load MNIST Data to clients
train_data, test_data = data.load_dataset("mnist")


# In[53]:


# For securing if the next cell execution is skipped
FLTrust = None
FLTrust_cosine_attack = None


# <h1>FLTrust: Skip section below for any other averaging than FLTrust.</h1>

# In[54]:


FLTrust = True
root_ratio = 0.01
train_data, root_data = torch.utils.data.random_split(train_data, [int(len(train_data) * (1-root_ratio)), 
                                                              int(len(train_data) * root_ratio)])
root_loader = torch.utils.data.DataLoader(root_data, batch_size=fedargs.client_batch_size, shuffle=True, **kwargs)


# <h2>Resume</h2>

# In[55]:


clients_data = data.split_data(train_data, clients)


# <h1>Poison: Skip section(s) below to run normal, modify if required.</h1>

# In[56]:


mal_clients = [c for c in range(24)]


# <h2>Label Flipping attack, Skip if not required</h2>

# In[12]:


#for client in mal_clients:
#    clients_data[clients[client]] = poison.label_flip(clients_data[clients[client]], 4, 9, poison_percent = -1)
    
#clients_data[clients[0]] = poison.label_flip(clients_data[clients[0]], 6, 2, poison_percent = 1)
#clients_data[clients[0]] = poison.label_flip(clients_data[clients[0]], 3, 8, poison_percent = 1)
#clients_data[clients[0]] = poison.label_flip(clients_data[clients[0]], 1, 5, poison_percent = 1)


# <h2>FLTrust: Sine Attack, Skip if not required</h2>

# In[57]:


FLTrust_cosine_attack = True


# <h2>Resume</h2>

# In[58]:


client_train_loaders, _ = data.load_client_data(clients_data, fedargs.client_batch_size, None, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=fedargs.test_batch_size, shuffle=True, **kwargs)

clients_info = {
        client: {"train_loader": client_train_loaders[client]}
        for client in clients
    }


# In[59]:


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def process(client, epoch, model, train_loader, fedargs, device):
    # Train
    model_update, model, loss = train_model(model, train_loader, fedargs, device)

    #Plot and Log
    for local_epoch, loss in enumerate(list(loss.values())):
        fedargs.tb.add_scalars("Training Loss/" + client, {str(epoch): loss}, str(local_epoch + 1))

    log.jsondebug(loss, "Epoch {} of {} : Federated Training loss, Client {}".format(epoch, fedargs.epochs, client))
    log.modeldebug(model_update, "Epoch {} of {} : Client {} Update".format(epoch, fedargs.epochs, client))
    
    return model_update


# In[60]:


import time
start_time = time.time()
    
# Federated Training
for _epoch in tqdm(range(fedargs.epochs)):

    epoch = _epoch + 1
    log.info("Federated Training Epoch {} of {}".format(epoch, fedargs.epochs))

    # Gloabal Model Update
    if epoch > 1:
        # For Tmean, not impacts others as of now
        avgargs = {"beta": 10}
        
        # For FLTrust, if FLTrust section is skipped, this piece of code will be ignored automatically
        if FLTrust:
            if FLTrust_cosine_attack is None:
                global_model_update, _, _ = train_model(global_model, root_loader, fedargs, device)
            avgargs["base_update"] = global_model_update
        
        # Average
        global_model = fl.federated_avg(client_model_updates, global_model, agg.Rule.FLTrust, **avgargs)
        log.modeldebug(global_model, "Epoch {} of {} : Server Update".format(epoch, fedargs.epochs))

        # Test
        global_test_output = fl.eval(global_model, test_loader, device)
        fedargs.tb.add_scalar("Gloabl Accuracy/", global_test_output["accuracy"], epoch)
        log.jsoninfo(global_test_output, "Global Test Outut after Epoch {} of {}".format(epoch, fedargs.epochs))
    
        # Update client models
        client_models = {client: copy.deepcopy(global_model) for client in clients}

    # Clients

    tasks = [process(client, epoch, client_models[client],
                     clients_info[client]['train_loader'],
                     fedargs, device) for client in clients]
    try:
        updates = fedargs.loop.run_until_complete(asyncio.gather(*tasks))
    except KeyboardInterrupt as e:
        print("Caught keyboard interrupt. Canceling tasks...")
        tasks.cancel()
        fedargs.loop.run_forever()
        tasks.exception()
    
    client_model_updates = {client: update for client, update in zip(clients, updates)}
    
    # For FLTrust cosine attack, Malicious Clients
    if FLTrust_cosine_attack:
        global_model_update, _, _ = train_model(global_model, root_loader, fedargs, device)
        for client in mal_clients:
            client_model_updates[clients[client]] = poison.model_poison_cosine_coord(global_model_update, client_model_updates[clients[client]], 1, client_model_updates[clients[client + 24]])
    
print(time.time() - start_time)


# In[ ]:





# In[ ]:



