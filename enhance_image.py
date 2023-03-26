# =============================================================================
# Imports
# =============================================================================
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from models import MLP, normalized_pt, get_transform_to_params
from os.path import basename, join, exists
from safetensors import safe_open
from PIL import Image
import numpy as np
import argparse
import random
import torch
import clip
import time
import os 
from tqdm import tqdm 

#flag/folder/temp
device = "cuda" if torch.cuda.is_available() else "cpu"
runif = np.random.uniform


# =============================================================================
# Methods
# =============================================================================
#image => aesthetic score
def measure_aesthetic(pil_image, model_clip, model_evalaesthetic, flip=True):
    image = pil_image
    image = ToTensor()(image)
    image = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        images = [image]
        if flip:
            image_flip = torch.flip(image,(2,))
            images += [image_flip]
        aesthetic_values = []
        for img in images:
            image_features = model_clip.encode_image(img)
            im_emb_arr = normalized_pt(image_features)
            aesthetic_values += [model_evalaesthetic(im_emb_arr.float())]
        aesthetic_value = torch.stack(aesthetic_values).mean(0)
    return round(aesthetic_value[0].item(),4)

#get a random transform
def get_random_transform(transforms_list, transform_to_params):
    #get random transform & params
    t = random.choice(transforms_list)
    random_kwargs = transform_to_params[t]
    kwargs = {}
    for k,v in random_kwargs.items():
        if len(v) == 1:
            kwargs[k] = round(runif(*v[0]),3)
        else:
            vx = v[0](*v[1])
            if v[0] != np.random.randint:
                vx = round(v, 3)
            if k == 'kernel_size':
                if vx % 2 == 0:
                    vx += 1
            kwargs[k] = vx
    return t, kwargs

#get an optimal transform 
def get_optimal_transform(image, it, transforms_list, transform_to_params, model_clip, model_evalaesthetic, N_TRY):
    t = transforms_list[it%len(transforms_list)]
    best_score = 0
    best_kwargs = {}
    param_keys = list(transform_to_params[t].keys())
    v2s = {}
    for key in param_keys:
        min_v = transform_to_params[t][key][-1][0]
        max_v = transform_to_params[t][key][-1][1]
        values_to_try = np.linspace(min_v, max_v, N_TRY)
        if transform_to_params[t][key][0] == np.random.randint:
            values_to_try = np.linspace(min_v, max_v-1, N_TRY)
            values_to_try = set([int(np.floor(e)) for e in values_to_try])
        if key == 'kernel_size':
            values_to_try = [v for v in values_to_try if v % 2 != 0]
        for v in values_to_try:
            kwargs = {k:1 for k in param_keys}
            kwargs[key] = v
            post_image = t(image, **kwargs)
            score = measure_aesthetic(post_image, model_clip, model_evalaesthetic)
            v2s[v] = score
            if score > best_score:
                best_score = score
                best_kwargs = kwargs
    return t, best_kwargs


# =============================================================================
# Data
# =============================================================================
#process image
def start_process_image(base_pil_image, kwargs):
    print('|- process data...')
    #clean tmp
    id_run = kwargs['id_run']
    tmp_folder = kwargs['tmp_folder']
    if not exists(tmp_folder):
        os.mkdir(tmp_folder)
    else:
        for base in os.listdir(tmp_folder):
            pth_to_rm = join(tmp_folder, base)
            assert len(pth_to_rm) > 10, "safety check"
            os.remove(pth_to_rm)
    #crop/preproc image
    base_pil_image.save(join(tmp_folder,f'{id_run}_step_0.jpg'))
    base_pil_image_c = CenterCrop(224)(Resize(224)(base_pil_image))
    total_light_base = ToTensor()(base_pil_image).mean()
    return base_pil_image_c, total_light_base

# base_pil_image_c, total_light_base = start_process_image(base_pil_image)

# =============================================================================
# Models
# =============================================================================
def get_models(kwargs):
    print('|- loading models...')
    weights = kwargs['weights']
    model_evalaesthetic = MLP(768)  
    model_evalaesthetic.load_state_dict(weights)
    model_evalaesthetic.to(device)
    model_evalaesthetic.eval()
    model_clip, _ = clip.load("ViT-L/14", device=device)  
    return model_clip, model_evalaesthetic
     
# model_clip, model_evalaesthetic = get_models()

# =============================================================================
# LOOP
# =============================================================================

#reproduce the transforms of a node of the tree (recursive)
def apply_node(image, node, dtree):
    if node == 0:
        return image
    elif isinstance(node, tuple):
        return node[0](image, **node[1])
    elif isinstance(node, int):
        nodes = dtree[node]
        for node in nodes:
            image = apply_node(image, node, dtree)
    return image

    
def enhance_image(base_pil_image, model_clip, model_evalaesthetic, eikwargs):
    #init data
    base_pil_image_c, total_light_base = start_process_image(base_pil_image, eikwargs)
    tmp_folder = eikwargs['tmp_folder']
    
    #evaluate base score
    base_score = measure_aesthetic(base_pil_image_c, model_clip, model_evalaesthetic)
    print('|- base_score:', base_score)
    #init decision tree
    node_to_score = {}
    dtree = [[0]]
    node_to_score[0] = base_score
    #node parameters
    thres_bestscorelow = 0.2
    thres_basescorelow = 0.1
    max_it_try_random = 300
    p_non_optimal = 0.8
    ratio_light_thres = eikwargs['min_light_ratio']
    
    #init transforms
    transform_to_params = get_transform_to_params(eikwargs['mode'])
    transforms_list = list(transform_to_params.keys())
    
    
    #init
    id_run = eikwargs['id_run']
    it = 1
    n_iter = eikwargs['n_iter']
    best_score = base_score
    best_node = 0
    start_loop_time = time.time()
    last_save_time = time.time() #-9999
    max_delay = eikwargs['max_delay']
    last_improv_it = 0
    print('|- starting loop')
    #for loop
    for it in tqdm(range(it, n_iter)):
        #decide of previous node 
        do_optimal = it < last_improv_it + len(transform_to_params)
        if it < max_it_try_random:
            prev_node = int(runif(0, max(1, len(dtree))))
        else:
            #start from random node 
            if runif(0,1) < p_non_optimal and not do_optimal:
                prev_node = int(runif(0, len(dtree)))
                #filter good
                worst_than_best = node_to_score[prev_node] < best_score - thres_bestscorelow
                worst_than_base = node_to_score[prev_node] < base_score - thres_basescorelow
                if worst_than_best or worst_than_base: continue
            #use best node
            else:
                prev_node = best_node
        
        #apply previous node
        image = base_pil_image_c
        image = apply_node(image, prev_node, dtree)
        #apply new node
        if do_optimal:
            t, kwargs = get_optimal_transform(image, it, transforms_list, transform_to_params, model_clip, model_evalaesthetic, eikwargs['N_TRY'])
        else:
            t, kwargs = get_random_transform(transforms_list, transform_to_params)
        image = t(image, **kwargs)
        
        #process
        score = measure_aesthetic(image, model_clip, model_evalaesthetic)
        
        #post process score (fix "too dark = great" as evaluated by the unperfect model)
        total_light_post = ToTensor()(image).mean()
        ratio_light = total_light_post / total_light_base
        if ratio_light < ratio_light_thres:
            score -= (1-ratio_light) * 6
        
        #add to tree
        current_node = len(dtree)
        dtree += [[prev_node, (t, kwargs)]]
        #log
        log_it = f'|- it: {it}/{n_iter} | score: {score:.2f} | best_score: {best_score:.2f} / {base_score:.2f}'
        logged = False
        if len(node_to_score) == 0 or score > best_score:
            best_node = current_node
            # if manual_args is not None:
            #     plt.imshow(image)
            #     plt.show()
            if time.time() - last_save_time > 5 or it < 100:
                tmp_file = join(tmp_folder,f'{id_run}_step_{it}.jpg')
                tmp_img = apply_node(base_pil_image, best_node, dtree)
                tmp_img.save(tmp_file)
                yield tmp_img
                # print(f'- intermediate result saved in {tmp_file}')
                last_save_time = time.time()
            best_score = score
            last_improv_it = it
            #log
            log_it = f'|- it: {it}/{n_iter} | score: {score:.2f} | best_score: {best_score:.2f} / {base_score:.2f}'
            logged = True
            # print(log_it)
        node_to_score[current_node] = score
        # if (it % 100 == 0 or it < 10) and logged == False: #may be skipped
            # print(log_it)
        
        #if too long, stop
        if time.time() - start_loop_time > max_delay:
            print(f'|- process stopped, over {max_delay}s')
            break
        
        #if stop signal, stop
        if exists(eikwargs['stop_signal']):
            print('|- stopped!')
            os.remove(eikwargs['stop_signal'])
            break
    
    best_score = max(node_to_score.values())
    print('|- base_score:', base_score)
    print('|- best_score:', best_score, f'(+{best_score-base_score:.2f})')

    yield dtree, best_node

# #display best image
# best_node = [k for k,v in node_to_score.items() if v == max(node_to_score.values())][0]
# image = apply_node(base_pil_image_c, best_node)
# score = measure_aesthetic(image)
# assert score == max(node_to_score.values())

#save out
def log_output(base_pil_image, dtree, best_node, id_run, base_pth, kwargs):
    out_folder = kwargs['out_folder']
    if not exists(out_folder):
        os.mkdir(out_folder)
    out_pth = join(out_folder, f'{id_run}_best_{base_pth}')
    out_image = apply_node(base_pil_image, best_node, dtree)
    out_image.save(out_pth)
    
    print('|- out_pth:', out_pth)
    return out_image

















