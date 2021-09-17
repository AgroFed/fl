import copy, cv2, heapq, os, sys, torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import sim

def insert_trojan_plus(instance):
    instance = cv2.rectangle(instance, (13,26), (15,26), (2.8088), (1))
    instance = cv2.rectangle(instance, (14,25), (14,27), (2.8088), (1))
    return instance

def insert_trojan_pattern(instance):
    instance = cv2.rectangle(instance, (2,2), (2,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (3,3), (3,3), (2.8088), (1))
    instance = cv2.rectangle(instance, (4,2), (4,2), (2.8088), (1))
    return instance

def insert_trojan_gap(instance):
    instance = cv2.rectangle(instance, (0,2), (1,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (5,2), (6,2), (2.8088), (1))
    return instance

def insert_trojan_size(instance):
    instance = cv2.rectangle(instance, (0,2), (1,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (3,2), (4,2), (2.8088), (1))
    return instance

def insert_trojan_pos(instance):
    instance = cv2.rectangle(instance, (2,2), (3,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (2,4), (3,4), (2.8088), (1))
    instance = cv2.rectangle(instance, (5,2), (6,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (5,4), (6,4), (2.8088), (1))
    return instance

def insert_trojan(client_data, target, func, poison_percent):
    client_data = list(client_data)
    total_occurences = len([1 for _, label in client_data])
    poison_count = poison_percent * total_occurences

    for index, (instance, label) in enumerate(client_data):
        if index >= poison_count:
            break
        client_data[index] = list(client_data[index])
        instance = instance.squeeze().numpy()

        # insert trojan type
        instance = func(instance)

        client_data[index][0] = torch.Tensor(instance).unsqueeze(0)
        client_data[index][1] = target
        
        client_data[index] = tuple(client_data[index])

    return list(client_data)

def label_flip(data, flip_labels, poison_percent = 0.5):
    data = list(data)
    
    for source_label, target_label in flip_labels.items():
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

def label_flip_next(data, flip_labels, poison_percent = 0.5):
    data = list(data)

    poison_count = poison_percent * len(data)
    if poison_percent == -1:
        poison_count = len(data) 

    label_poisoned = 0
    for index, _ in enumerate(data):
        data[index] = list(data[index])
        if data[index][1] in flip_labels.keys():
            data[index][1] = flip_labels[data[index][1]]
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

    npd = c_arr - b_arr
    p_arr = copy.deepcopy(c_arr)
    
    dot_mb = scale_dot * sim.dot(p_arr, b_arr)
    norm_b = sim.norm(b_arr)
    norm_c = sim.norm(c_arr)
    norm_m = norm_c
    sim_mg = 1
    
    kwargs = {"scale_norm": cosargs["scale_norm"]} if "scale_norm" in cosargs else {}
    
    for index in heapq.nlargest(int(len(npd) * poison_percent), range(len(npd)), npd.take):
        p_arr, dot_mb, norm_m, sim_mg, updated = sim.cosine_coord_vector_adapter(b_arr, p_arr, index, dot_mb, norm_m, sim_mg, c_arr, norm_c, norm_b, **kwargs)
        
    params_changed = len(npd) - np.sum(p_arr == c_arr)

    return p_arr, params_changed

def model_poison_cosine_imp(base_model_update, client_model_update, poison_percent):
    b_arr, b_list = sim.get_net_arr(base_model_update)
    c_arr, c_list = sim.get_net_arr(client_model_update)
    
    npd = c_arr - b_arr
    p_arr = copy.deepcopy(c_arr)
    for index in heapq.nlargest(int(len(npd) * poison_percent), range(len(npd)), npd.take):
        p_arr[index] = p_arr[index] + (2* npd[index])

    client_model_update = sim.get_arr_net(client_model_update, p_arr, c_list)
    return client_model_update

def trim_attack(model):
    vi_shape = v[0].shape
    v_tran = nd.concat(*v, dim=1)
    maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
    minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
    direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
    # let the malicious clients (first f clients) perform the attack
    for i in range(f):
        random_12 = 1. + nd.random.uniform(shape=vi_shape)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v         