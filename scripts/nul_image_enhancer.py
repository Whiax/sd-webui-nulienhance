from modules import script_callbacks, scripts
from os.path import basename, join, exists
from safetensors import safe_open
import enhance_image as nulie
from PIL import Image
import gradio as gr
import shutil
import time
import os

__version__ = '0.1.0'


#prepare dir
base_dir = scripts.basedir()
base_dir_nulie = join(base_dir, 'nulie_out')
stop_signal = join(base_dir_nulie, 'stop')
tmp_folder = join(base_dir_nulie, '_tmp')
res_folder = join(base_dir_nulie, 'results')
for d in [base_dir_nulie, tmp_folder, res_folder]:
    if not exists(d):
        os.mkdir(d)


#enhance image
enhancing = False
model_clip, model_evalaesthetic = None, None
def enhance_image(base_pil_image, max_delay, n_iter, min_light_ratio):
    global model_clip, model_evalaesthetic, base_dir, base_dir_nulie, stop_signal, enhancing
    enhancing = True
    #Flags / init / constant
    print('Starting NUlie - image enhancer...')
    start_time = time.time()
    id_run = int(start_time*100)
    mode = 'soft'
    N_TRY = 20 if mode == 'hard' else 10
    kwargs = {'id_run':id_run, 'n_iter':n_iter, 'max_delay':max_delay, 'mode':'soft', 'N_TRY':N_TRY}
    kwargs['base_dir'] = base_dir
    kwargs['tmp_folder'] = tmp_folder
    kwargs['out_folder'] = res_folder
    pth = 'sac+logos+ava1-l14-linearMSE.safetensors'
    if not exists(pth):
        pth = './extensions/sd-webui-nulienhance/' + pth
    kwargs['min_light_ratio'] = min_light_ratio
    kwargs['stop_signal'] = stop_signal
    
    #load models
    if model_clip is None:
        weights = {}
        with safe_open(pth, framework="pt", device="cpu") as f:
           for key in f.keys():
               weights[key] = f.get_tensor(key)
        kwargs['weights'] = weights
        model_clip, model_evalaesthetic = nulie.get_models(kwargs)
    
    #enhance
    generator = nulie.enhance_image(base_pil_image, model_clip, model_evalaesthetic, kwargs)
    out = True
    out_image = None
    while type(out) != tuple:
        out = next(generator)
        if type(out) != tuple:
            yield out
        else:
            dtree, best_node = out
            
            #save
            base_pth = f'{id_run}_nulie_out.jpg'
            out_image = nulie.log_output(base_pil_image, dtree, best_node, id_run, base_pth, kwargs)
            
            #runtime
            print('|- total runtime:', int(time.time() - start_time), 's')
    print('Ending NUlie')
    enhancing = False
    return out_image
 
def stop():
    global stop_signal
    if enhancing:
        open(stop_signal, 'w+').close()
 

def about_tab():
    gr.Markdown("## NUlIE: Non-Ultimate Image Enhancer 🐌 - About")
    s = ''
    s += "An algorithm based on decision trees and deep learning models to enhance images.  \n"
    s += "The algorithm is simple, it does not always provide good results on already great images.  \n"
    s += "The default maximum runtime is 60 seconds / image. The algorithm can provide good results in 10 seconds on some images.  \n"
    s += "The algorithm tries multiple transformations on contrast / saturation / lighting etc., it evaluates the changes and keeps the best ones.  \n"
    gr.Markdown(s)
    gr.Markdown("## Github")
    gr.Markdown("More on [NUl ImageEnhancer on Github](https://github.com/Whiax/NUl-Image-Enhancer) ⭐ if you like it!")

def help_tab():
    gr.Markdown("## NUlIE: Non-Ultimate Image Enhancer 🐌 - Help")
    gr.Markdown("### Quick Guide")
    s = ''
    s += "1. Load an image.  \n"
    s += "2. Click Enhance.  \n"
    s += "3. See the progress.  \n"
    s += "4. Access results in 'Open results'.  \n"
    s += "  \n"
    s += "### Tips  \n"
    s += "  \n"
    s += "- You can stop the algorithm at any time  \n"
    s += "- You can change the maximum execution time, by default it's 30seconds.   \n"
    s += "- By default, the algorithm tends to like dark images too much, if you think the output is too dark or not dark enough, you can adjust this ratio. 1 = 'Do not darken at all', 0 = 'A totally black image is ok', default = 0.9.   \n"
    s += "- The algorithm is not able to enhance all images.   \n"
    gr.Markdown(s)

from modules import shared
import sys, platform
import subprocess as sp
def open_folder(f):
    if not os.path.exists(f):
        print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
        return
    elif not os.path.isdir(f):
        print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
        return

    if not shared.cmd_opts.hide_ui_dir_config:
        path = os.path.normpath(f)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            sp.Popen(["open", path])
        elif "microsoft-standard-WSL2" in platform.uname().release:
            sp.Popen(["wsl-open", path])
        else:
            sp.Popen(["xdg-open", path])
            


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tab("Enhance"):
            with gr.Row().style(equal_height=False): 
                image = gr.Image(type='pil', show_label=False, source="upload", interactive=True, label="Image", elem_id="myimgin").style(height=480)
                out_grimage = gr.Image(source='canvas', label='ImageOut').style(height=480)
            with gr.Row(): 
                button_enhance = gr.Button("Enhance", variant='primary')
                button_stop = gr.Button("Stop", variant='primary')
                button_open_results = gr.Button("Open results", variant='secondary')
                button_open_intermediate = gr.Button("Open intermediate results", variant='secondary')
            with gr.Row(): 
                max_delay = gr.Slider(minimum=10, maximum=120, label='Maximum execution time (seconds)', value=30)
                n_iter = gr.Slider(minimum=10, maximum=10000, step=50, label='Number of iterations', value=1000)
                min_light_ratio = gr.Slider(minimum=0.2, maximum=1, step=0.05, label='Minimum lighting ratio', value=0.9)
                
            button_open_results.click( fn=lambda: open_folder(res_folder), inputs=[], outputs=[])
            button_open_intermediate.click( fn=lambda: open_folder(tmp_folder), inputs=[], outputs=[])
            
        with gr.Tab("Help"):
            help_tab()
        with gr.Tab("About"):
            about_tab()
        button_enhance.click(enhance_image, inputs=[image, max_delay, n_iter, min_light_ratio], outputs=[out_grimage])
        button_stop.click(stop)
    return [(ui, "NUl Image Enhancer", "NUlie")]


script_callbacks.on_ui_tabs(add_tab)





