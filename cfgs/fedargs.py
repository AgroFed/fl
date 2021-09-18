import asyncio, inspect, os, sys
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg, nn, poison, resnet

argsdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

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
        self.agg_rule = agg.Rule.FedAvg
        self.dataset = "mnist" #can run for fmnist, cifar-10
        self.labels = [label for label in range(10)]
        self.model = nn.ModelMNIST() # for mnist and f-mnist #resnet.ResNet18() for cifar - 10
        self.tb = SummaryWriter(argsdir + '/../out/runs/federated/FedAvg/mn-lfa-next-e-50', comment="Centralized Federated training")
        
fedargs = FedArgs()
        
# FLTrust
FLTrust = {"is": False if fedargs.agg_rule in [agg.Rule.FLTrust, agg.Rule.FLTC] else False,
           "ratio": 0.003,
           "data": None,
           "loader": None,
           "proxy": {"is": False,
                     "ratio": 0.5,
                     "data": None,
                     "loader": None}}

# No of malicious clients
mal_clients = [c for c in range(20)]

# Label Flip
label_flip_attack = {"is": True,
                     "func": poison.label_flip_next,
                     "labels": {},
                     "percent": -1}
label_flip_attack["labels"] = {4: 6} if label_flip_attack["is"] and label_flip_attack["func"] is poison.label_flip else None
label_flip_attack["labels"] = {label: fedargs.labels[(index + 1) % len(fedargs.labels)] for index, label in enumerate(fedargs.labels)} if label_flip_attack["is"] and label_flip_attack["func"] is poison.label_flip_next else label_flip_attack["labels"]

# Backdoor
backdoor_attack = {"is": False,
                   "trojan_func": poison.insert_trojan_pattern,
                   "target_label": 6,
                   "ratio": 0.006,
                   "data": None,
                   "loader": None}

# Layer replacement attack
layer_replacement_attack = {"is": False,
                            "layers": ["conv1.weight"]}

# Cosine attack
cosine_attack = {"is": False,
                 "args": {"poison_percent": 1,
                          "scale_dot": 5,
                          "scale_dot_factor": 1,
                          "scale_norm": 500,
                          "scale_norm_factor": 2,
                          "scale_epoch": 5}}

# Sybil attack, for sending same update as base
sybil_attack = {"is": False}