# SD-WEBUI Extension for NUlIE 🐌
An algorithm based on decision trees and deep learning models to enhance images.  
The algorithm is simple, it does not always provide good results on already great images.  
The default maximum runtime is 60 seconds / image. The algorithm can provide good results in 10 seconds on some images.  
The algorithm tries multiple transformations on contrast / saturation / lighting etc., it evaluates the changes and keeps the best ones. 

## Results / Original repository
- [NUlIE](https://github.com/Whiax/NUl-Image-Enhancer)

## Usage

The SD-webui needs to be started with the following command:
```bash
./webui.sh --gradio-queue
```


## Requirements

- A GPU with > 2GB VRAM (very slow on CPU)
- [CLIP](https://github.com/openai/CLIP)
- [sd-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/)
- [safetensors](https://github.com/huggingface/safetensors)

## Models

- [Aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor): [github.com/christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [CLIP](https://github.com/openai/CLIP): [github.com/openai/CLIP](https://github.com/openai/CLIP)
