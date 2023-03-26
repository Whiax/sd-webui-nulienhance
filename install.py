import importlib
import subprocess

def is_installed(package):
    try:
        spec=importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False
    return spec is not None

if not is_installed('torch'):
    print('installing pytorch')
    subprocess.run('conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0', shell=True)
    
requirements2v = {'ftfy':'','regex':'','tqdm':'', 'git+https://github.com/openai/CLIP.git':'', 'pillow':'', 'safetensors':''} #, 'torch':'', 'torchvision':''
url2name = {'git+https://github.com/openai/CLIP.git':'clip', 'pillow':'PIL'}
for requirementlink, version in requirements2v.items():
    requirement = url2name.get(requirementlink, requirementlink)
    if not is_installed(requirement):
        if version != '':
            version = f'=={version}'
        subprocess.run(f"pip3 install {requirementlink}{version}", shell=True)

print('Installed.')
