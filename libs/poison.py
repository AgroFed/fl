import copy, heapq, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import sim

def label_flip(data, source_label, target_label, poison_percent = 0.5):
    data = list(data)
    total_occurences = len([1 for _, label in data if label == source_label])
    poison_count = poison_percent * total_occurences

    # Poison all and keep only poisoned samples
    if poison_percent == -1:
        data=[tuple([instance, target_label]) for instance, label in data if label == source_label]
        
    else:
        label_poisoned = 0
        for index, _ in enumerate(data):
            data[index] = list(data[index])
            if data[index][1] == source_label:
                data[index][1] = target_label
                label_poisoned += 1
            data[index] = tuple(data[index])
            if label_poisoned >= poison_count:
                break

    return tuple(data)

def layer_replacement_attack(model_to_attack, model_to_reference, layers):
    params1 = model_to_attack.state_dict().copy()
    params2 = model_to_reference.state_dict().copy()
    
    for layer in layers:
        params1[layer] = params2[layer]
        #params1['fc1.weight'] = params2['fc1.weight']
    
    model = copy.deepcopy(model_to_attack)
    model.load_state_dict(params1, strict=False)
    return model

def model_poison_cosine_coord(b_arr, cosargs, c_arr):
    poison_percent = cosargs["poison_percent"] if "poison_percent" in cosargs else 1
    scale_dot = cosargs["scale_dot"] if "scale_dot" in cosargs else 1
    
    #b_arr, b_list = sim.get_net_arr(base_model_update)
    #c_arr, c_list = sim.get_net_arr(client_model_update)

    npd = c_arr - b_arr
    p_arr = copy.deepcopy(c_arr)
    
    dot_mb = scale_dot * sim.dot(p_arr, b_arr)
    norm_m = sim.norm(p_arr)
    norm_c = sim.norm(c_arr)
    sim_mg = sim.cosine_similarity(p_arr, c_arr)
    
    kwargs = {"scale_norm": cosargs["scale_norm"]} if "scale_norm" in cosargs else {}
    
    for index in heapq.nlargest(int(len(npd) * poison_percent), range(len(npd)), npd.take):
        p_arr, dot_mb, norm_m, sim_mg, updated = sim.cosine_coord_vector_adapter(b_arr, p_arr, index, dot_mb, norm_m, sim_mg, c_arr, norm_c, **kwargs)
        
    params_changed = len(npd) - np.sum(p_arr == c_arr)

    return p_arr, params_changed#, c_list
    #client_model_update = sim.get_arr_net(client_model_update, p_arr, c_list)
    #return client_model_update

def model_poison_cosine_imp(base_model_update, client_model_update, poison_percent):
    b_arr, b_list = sim.get_net_arr(base_model_update)
    c_arr, c_list = sim.get_net_arr(client_model_update)
    
    npd = c_arr - b_arr
    p_arr = copy.deepcopy(c_arr)
    for index in heapq.nlargest(int(len(npd) * poison_percent), range(len(npd)), npd.take):
        p_arr[index] = p_arr[index] + (2* npd[index])

    client_model_update = sim.get_arr_net(client_model_update, p_arr, c_list)
    return client_model_update