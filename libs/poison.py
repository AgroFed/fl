import heapq
import os, sys
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

def model_poison_cosine_coord(base_model_update, good_model_update, poison_percent, good_client_update):
    b_arr, blist = sim.get_net_arr(base_model_update)
    g_arr, glist = sim.get_net_arr(good_model_update)
    c_arr, clist = sim.get_net_arr(good_client_update)
    
    npd = c_arr - b_arr
    p_arr = c_arr
    for index in heapq.nlargest(int(len(npd) * poison_percent), range(len(npd)), npd.take):
        p_arr[index] = sim.cosine_coord_vector(b_arr, p_arr, index)

    poison_model_update = sim.get_arr_net(base_model_update, p_arr, blist)
    return poison_model_update

def model_poison_cosine_imp(base_model_update, good_model_update, poison_percent, good_client_update):
    b_arr, blist = sim.get_net_arr(base_model_update)
    g_arr, glist = sim.get_net_arr(good_model_update)
    c_arr, clist = sim.get_net_arr(good_client_update)
    
    npd = c_arr - b_arr
    p_arr = c_arr
    for index in heapq.nlargest(int(len(npd) * poison_percent), range(len(npd)), npd.take):
        p_arr[index] = p_arr[index] + (2* npd[index])

    poison_model_update = sim.get_arr_net(base_model_update, p_arr, blist)
    return poison_model_update

def model_poison_cosine_rand(base_model_update, good_model_update):
    b_arr, blist = sim.get_net_arr(base_model_update)
    g_arr, glist = sim.get_net_arr(good_model_update)
    
    npd = g_arr - b_arr
    np.random.shuffle(npd)
    
    p_arr = b_arr + npd
    poison_model_update = sim.get_arr_net(base_model_update, p_arr, blist)
    return poison_model_update