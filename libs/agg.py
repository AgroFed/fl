import copy, enum, torch
import numpy as np
from functools import reduce

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import sim, log

class Rule(enum.Enum):
    FedAvg = 0
    FLTrust = 1
    TMean = 2
    FLTC = 3

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
    
    trimmed_updated_model_arrs = []
    for index, arr in enumerate(merged_updated_model_arrs):
        l_arr = arr[np.where(arr >= base_model_arr[index])]
        s_arr = arr[np.where(arr < base_model_arr[index])]        
        
        if len(l_arr) > len(s_arr):
            trimmed_updated_model_arrs.append(l_arr)
        else:
            trimmed_updated_model_arrs.append(s_arr)

    print("tarr_0", trimmed_updated_model_arrs[0])            
    ts_scores = [(arr / base_model_update_norm) for arr in trimmed_updated_model_arrs]
    print("ts", ts_scores[0], sum(ts_scores[0]))
    
    #ReLU
    sign_b = np.sign(base_model_arr)
    model_arr = np.zeros(len(base_model_arr))
    for index, (ts_score, arr) in enumerate(zip(ts_scores, trimmed_updated_model_arrs)):
        ts_score = sign_b[index] * ts_score
        #ts_score[np.where(ts_score < 1)] = 0
        if max(ts_score) - min(ts_score) > 0:
            ts_score = ((ts_score - min(ts_score)) / (max(ts_score) - min(ts_score)))
            model_arr[index] = sum((arr * ts_score) / sum(ts_score))
        ts_scores[index] = ts_score

        '''
        ts_scores[index] = sign_b[index] * ts_score
        sign_t = np.sign(ts_score)
        sign_p = len(ts_score[np.where(sign_bt < 1)]
        sign_n = len(ts_score[np.where(sign_bt < 1)]
        if 1 in sign_t and -1 in sign_t:
            sign_bt = sign_b[index] * sign_t
            ts_score[np.where(sign_bt < 1)] = 0
            ts_scores[index] = ts_score
        '''
    print("n_ts", sign_b[0], ts_scores[0], sum(ts_scores[0]))    
        
    #ts_scores = np.array([(ts_score / sum(ts_score)) for ts_score in ts_scores])
    #ts_scores = np.array([((ts_score - min(ts_score)) / (max(ts_score) - min(ts_score))) for ts_score in ts_scores if (max(ts_score) - min(ts_score)) > 0 else ts_score])

    #print("n_ts", ts_scores[0], sum(ts_scores[0]))
    #print("arr * ts", trimmed_updated_model_arrs[0] * ts_scores[0])
    #print("arr * ts / sts", (trimmed_updated_model_arrs[0] * ts_scores[0]) / sum(ts_scores[0]))

    #model_arr = np.array([sum((arr * ts) / sum(ts)) for arr, ts in zip(trimmed_updated_model_arrs, ts_scores)])
    print("arr * ts / sts", (model_arr[0]))
    
    model = sim.get_arr_net(base_model_update, model_arr, b_list)
    
    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def TMean(base_model, models, **kwargs):
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