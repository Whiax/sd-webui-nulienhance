# SD-WEBUI Extension for NUlIE 🐌 
An algorithm based on decision trees and deep learning models to enhance images.  
The algorithm is simple, **it does not always provide good results on already great images**.  
The default maximum runtime is 30 seconds / image. The algorithm can provide good results in 10 seconds on some images.  
The algorithm tries multiple transformations on contrast / saturation / lighting etc., it evaluates the changes and keeps the best ones. 

Changelog:
- 2023/03/29: Added support for manual image editing (brightness, saturation, contrast..)
- 2023/03/28: Fixed issue on loading nn weights

## Results / Original repository
- Standalone: [NUlIE](https://github.com/Whiax/NUl-Image-Enhancer) (with other results)




https://user-images.githubusercontent.com/12411288/227802954-bd229390-9c73-40d1-8ed2-80f8cd06dc59.mp4






## Usage

#### After 2023/03/25
```bash
./webui.sh
```
- Navigate to the NUl Image Enhancer Tab

#### Before 2023/03/25  
See: https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/af2db25c84c9a226ab34959e868fc18740418b4b  
Before 2023/03/25, the SD-webui needed to be started with the following command:
```bash
./webui.sh --gradio-queue
```
/!\ Warning: This command was experimental and breaks the UI restart button.

## Requirements

- [sd-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/)
- An NVIDIA GPU with > 2GB VRAM (very slow on CPU)
- [CLIP](https://github.com/openai/CLIP)
- [safetensors](https://github.com/huggingface/safetensors)

## Models

- [Aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor): [github.com/christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [CLIP](https://github.com/openai/CLIP): [github.com/openai/CLIP](https://github.com/openai/CLIP)

## 
➡️ More details on the algorithm: https://medium.com/p/how-i-built-an-ai-image-enhancer-f7d77186678e

⭐ If you like it! And I'll also try to improve this extension!
