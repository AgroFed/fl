import copy, enum, torch
import numpy as np
from functools import reduce

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import sim, log

class Rule(enum.Enum):
    FedAvg = 0
    FLTrust = 1
    T_Mean = 2
    FLTC = 3
    FLTC_Layer = 4

def verify_model(base_model, model):
    params1 = base_model.state_dict().copy()
    params2 = model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 not in params2:
                return False
    return True

def sub_model(model1, model2):
    params1 = model1.state_dict().copy()
    params2 = model2.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] - params2[name1]
    model = copy.deepcopy(model1)
    model.load_state_dict(params1, strict=False)
    return model

def add_model(dst_model, src_model):
    params1 = dst_model.state_dict().copy()
    params2 = src_model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] + params2[name1]
    model = copy.deepcopy(dst_model)
    model.load_state_dict(params1, strict=False)
    return model

def scale_model(model, scale):
    params = model.state_dict().copy()
    scale = torch.tensor(scale)
    with torch.no_grad():
        for name in params:
            params[name] = params[name].type_as(scale) * scale
    scaled_model = copy.deepcopy(model)
    scaled_model.load_state_dict(params, strict=False)
    return scaled_model

def FedAvg(base_model, models):
    model_list = list(models.values())
    model = reduce(add_model, model_list)
    model = scale_model(model, 1.0 / len(models))
    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def FLTrust(base_model, models, **kwargs):
    base_model_update = kwargs["base_model_update"]
    base_norm = kwargs["base_norm"] if "base_norm" in kwargs else True

    if base_norm:
        # Base Model Norm
        base_model_update_norm = sim.grad_norm(base_model_update)
    
    model_list = list(models.values())
    ts_score_list=[]
    fl_score_list=[]
    updated_model_list = []
    for model in model_list:
        ts_score = sim.grad_cosine_similarity(base_model_update, model)

        # Relu
        if ts_score < 0:
            ts_score = 0
        ts_score_list.append(ts_score)

        if base_norm:
            # Model Norm    
            norm = sim.grad_norm(model)
            ndiv = base_model_update_norm/norm
            scale_norm = ts_score * ndiv
            model = scale_model(model, scale_norm)
            fl_score_list.append(scale_norm)
        else:
            model = scale_model(model, ts_score)

        updated_model_list.append(model)

    log.info("Cosine Score {}".format(ts_score_list))
    log.info("FLTrust Score {}".format(fl_score_list))
        
    model = reduce(add_model, updated_model_list)
    model = scale_model(model, 1.0 / sum(ts_score_list))

    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def FLTC(base_model, models, **kwargs):
    base_model_update = kwargs["base_model_update"]   
    base_model_arr, b_list = sim.get_net_arr(base_model_update)

    updated_model_list = []
    cosine_dist = []
    eucliden_dist = []
    for model in list(models.values()):
        model_arr, _ = sim.get_net_arr(model)
        updated_model_list.append(model_arr)
        cosine_dist.append(sim.cosine_similarity(base_model_arr, model_arr))
        eucliden_dist.append(sim.eucliden_dist(base_model_arr, model_arr))

    cosine_score = np.array(cosine_dist)
    cosine_score[np.where(cosine_score < 0)] = 0
    print("Cosine score", cosine_score)
    #cosine_score = sim.min_max_norm(cosine_score)
    eucliden_dist = np.array(eucliden_dist)

    updated_model_tensors = torch.tensor(updated_model_list)
    merged_updated_model_tensors = torch.stack([model for model in updated_model_tensors], 0)
    merged_updated_model_arrs = torch.transpose(merged_updated_model_tensors, 0, 1).numpy()

    client_scores = np.ones(len(models))
    model_arr = np.zeros(len(base_model_arr))
    
    for index, (b_arr, m_arr) in enumerate(zip(base_model_arr, merged_updated_model_arrs)):
        a_euc_score = (b_arr - m_arr) / eucliden_dist

        sign_p = np.where(np.sign(a_euc_score) == 1)
        sign_n = np.where(np.sign(a_euc_score) == -1)
        trusted_components = sign_p if len(sign_p[0]) > len(sign_n[0]) else sign_n

        euc_score  = a_euc_score[trusted_components]
        if len(euc_score) > 1:
            ts_score = sim.min_max_norm(euc_score)

            client_score = np.zeros(len(models))
            for _index, trusted_component in enumerate(trusted_components[0]):
                client_score[trusted_component] = ts_score[_index]
            client_scores = (client_scores + client_score) / 2

            if index == 10:
                print("m_arr", m_arr)                
                print("a_euc", a_euc_score)
                print("client_score", client_score)
                
            if sum(client_score) > 0:
                model_arr[index] = sum((m_arr * client_score) / sum(client_score))

    log.info("FLTC Score {}".format(client_scores))
    
    model = sim.get_arr_net(base_model_update, model_arr, b_list)
    
    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def FLTC_Layer(base_model, models, **kwargs):
    base_model_update = kwargs["base_model_update"] 
    base_model_arr, b_list = sim.get_net_arr(base_model_update)
    
    updated_model_list = []
    normalized_model_list = []
    for model in list(models.values()):
        model_arr, _ = sim.get_net_arr(model)
        updated_model_list.append(model_arr)
        
    updated_model_arrs = np.array(updated_model_list)    
    updated_model_tensors = torch.tensor(updated_model_list)
    merged_updated_model_tensors = torch.stack([model for model in updated_model_tensors], 0)
    merged_updated_model_arrs = torch.transpose(merged_updated_model_tensors, 0, 1).numpy()

    client_scores = np.ones(len(models))
    model_arr = np.zeros(len(base_model_arr))
    
    start_index = 0
    for shape in b_list:
        end_index = start_index + np.prod(list(shape))
        sub_base_arr = base_model_arr[start_index:end_index]
        eucliden_dist = np.array([sim.eucliden_dist(sub_base_arr, model[start_index:end_index]) for model in updated_model_arrs])
        
        for index in range(start_index, end_index):
            b_arr = base_model_arr[index]
            arr = merged_updated_model_arrs[index]

            euc_score = (b_arr - arr) / eucliden_dist

            sign_p = np.where(np.sign(euc_score) == 1)
            sign_n = np.where(np.sign(euc_score) == -1)
            trusted_components = sign_p if len(sign_p[0]) > len(sign_n[0]) else sign_n

            arr = arr[trusted_components]
            euc_score  = euc_score[trusted_components]
            if len(euc_score) > 1:
                ts_score = sim.min_max_norm(euc_score)

                client_score = np.zeros(len(models))
                for _index, trusted_component in enumerate(trusted_components[0]):
                    client_score[trusted_component] = ts_score[_index]
                client_scores = (client_scores + client_score) / 2

                if sum(ts_score) > 0:
                    model_arr[index] = sum((arr * ts_score) / sum(ts_score))
        
        start_index = end_index

    log.info("FLTC Score {}".format(client_scores))
    
    model = sim.get_arr_net(base_model_update, model_arr, b_list)
    
    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def _FLTC(base_model, models, **kwargs):
    base_model_update = kwargs["base_model_update"]
    base_model_update_norm = sim.grad_norm(base_model_update)
    base_model_arr, b_list = sim.get_net_arr(base_model_update)
    
    model_list = list(models.values())
    updated_model_list = []
    for model in model_list:
        norm = sim.grad_norm(model)
        ndiv = base_model_update_norm/norm
        model = scale_model(model, ndiv)
        model_arr, _ = sim.get_net_arr(model)
        updated_model_list.append(model_arr)
        
    updated_model_tensors = torch.tensor(updated_model_list)
    merged_updated_model_tensors = torch.sort(torch.stack([model for model in updated_model_tensors], 0), dim = 0)
    merged_updated_model_arrs = torch.transpose(merged_updated_model_tensors.values, 0, 1).numpy()
    merged_updated_model_indices = torch.transpose(merged_updated_model_tensors.indices, 0, 1).numpy()
    
    bs_score = base_model_arr / base_model_update_norm
    bs_theta = np.arccos(bs_score)
    client_scores = np.ones(len(models))

    model_arr = np.zeros(len(base_model_arr))
    
    for index, arr in enumerate(merged_updated_model_arrs):
        l_indices = np.where(arr >= base_model_arr[index])
        l_arr = arr[l_indices]
        
        s_indices = np.where(arr < base_model_arr[index])
        s_arr = arr[s_indices]
        
        arr = s_arr
        indices = s_indices
        if len(l_arr) > len(s_arr):
            arr = l_arr
            indices = l_indices

        ts_score = arr / base_model_update_norm
        ts_theta = np.arccos(ts_score)
        # wrt bs_theta
        ts_theta = bs_theta[index] - ts_theta
        ts_score = np.cos(ts_theta)

        # Relu
        #ts_score[np.where(ts_score < 0)] = 0
        
        # Sybils
        #ts_score = sim.max_min_norm(ts_score)
        
 
        # Clients scores
        client_score = np.zeros(len(models))
        for _ind, ind in enumerate(merged_updated_model_indices[index][indices]):
            client_score[ind] = ts_score[_ind]
        client_scores = (client_scores + client_score) / 2

        if sum(ts_score) > 0:
            model_arr[index] = sum((arr * ts_score) / sum(ts_score))


    print("Client_scores", client_scores)

    model = sim.get_arr_net(base_model_update, model_arr, b_list)
    
    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def T_Mean(base_model, models, **kwargs):
    model_list = list(models.values())
    dummy_model = copy.deepcopy(model_list[0])
    dummy_dict = dummy_model.state_dict()
    beta = kwargs["beta"]
    
    for k in dummy_dict.keys():
        merged_tensors = torch.sort(torch.stack([model.state_dict()[k].float() for model in model_list], 0), dim = 0)
        dummy_dict[k] = merged_tensors.values[beta : len(model_list) - beta].mean(0)

    dummy_model.load_state_dict(dummy_dict)
    if base_model is not None:
        base_model = sub_model(base_model, dummy_model)
    return base_model