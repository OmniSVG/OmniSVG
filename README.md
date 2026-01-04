<!-- <div align= "center">
    <h1> Official repo for OmniSVG</h1>

</div> -->

<h3 align="center"><strong>OmniSVG: A Unified Scalable Vector Graphics Generation Model</strong></h3>


<div align="center">
<a href='https://arxiv.org/abs/2504.06263'><img src='https://img.shields.io/badge/arXiv-2504.06263-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://omnisvg.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/OmniSVG/OmniSVG1.1_8B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/OmniSVG"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset%20-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/OmniSVG/MMSVGBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Bench-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/spaces/OmniSVG/OmniSVG-3B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo%20-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://github.com/OpenVGLab/OmniSVG-train'><img src='https://img.shields.io/badge/Training-Code-blue?logo=github'></a>
</div>

## 🔥🔥🔥 News !!
- [2025/12/31] 👋 We have released the training code of OmniSVG， Check out [OpenVGLab/OmniSVG-Train](https://github.com/OpenVGLab/OmniSVG-train)
- [2025/12/22] We have updated **MMSVG-Icon** (264K→904K) and **MMSVG-Illustration** (66K→255K) datasets with enhanced captions and PNG previews! Check out [MMSVG-Icon](https://huggingface.co/datasets/OmniSVG/MMSVG-Icon) and [MMSVG-Illustration](https://huggingface.co/datasets/OmniSVG/MMSVG-Illustration).
- [2025/12/02] We have released the **OmniSVG1.1_8B** weights and updated **OmniSVG1.1_4B** model weights! Check out [OmniSVG1.1_8B](https://huggingface.co/OmniSVG/OmniSVG1.1_8B) and [OmniSVG1.1_4B](https://huggingface.co/OmniSVG/OmniSVG1.1_4B).
- [2025/12/02] We have released **MMSVGBench** benchmark dataset and evaluation code! Check out [MMSVGBench](https://huggingface.co/datasets/OmniSVG/MMSVGBench) and [Evaluation](https://github.com/OmniSVG/OmniSVG?tab=readme-ov-file#5-evaluation).
- [2025/09/18] OmniSVG is accepted to **NeurIPS 2025**🔥! See you in San Diego!
- [2025/07/22] 👋 We have released the Huggingface Demo. 🤗[Demo](https://huggingface.co/spaces/OmniSVG/OmniSVG-3B).
- [2025/07/22] 👋 We have released the inference code and model weight of MMSVG-Icon and MMSVG-Illustration dataset. 🤗[Weight](https://huggingface.co/OmniSVG/OmniSVG).
- [2025/04/09] 👋 Release MMSVG-Icon and MMSVG-Illustration 🤗[Dataset](https://huggingface.co/OmniSVG).
- [2025/04/09] 👋 Upload paper and init project. [Read](https://arxiv.org/abs/2504.06263)


<p align="center">
    <img src="assets/OmniSVG-demo-gen-proc-anime-1080.gif" alt="Demo GIF" width="720px" />
</p>

## 🧩 Community Contributions
If you are developing / using OmniSVG in your projects, or you want to contribute to OmniSVG, please let us know 🎉.

- If you find data issues when using MMSVG dataset, please drop an issue in this [form](https://npqawhh9ht.feishu.cn/wiki/KHv2wDqAxiSV8skpkANcbmlwnqc?from=from_copylink).
- 👋 OmniSVG ComfyUI Plugin by [@smthemex](https://github.com/smthemex) [ComfyUI_OmniSVG](https://github.com/smthemex/ComfyUI_OmniSVG).

## 📑 Open-source Plan
- [x] Project Page & Technical Report
- [x] MMSVG-Icon and MMSVG-Illustration Dataset Release
- [x] Inference Code & Model Weight of MMSVG-Icon and MMSVG-Illustration Dataset
- [x] Online Demo (Gradio deployed on Huggingface)
- [x] Model Weight of OmniSVG1.1_8B Release
- [x] Model Weight of OmniSVG1.1_4B Release
- [x] MMSVGBench Benchmark & Evaluation Code Release



## 1. Introduction

**OmniSVG** is the first family of end-to-end multimodal SVG generators that leverage pre-trained Vision-Language Models (VLMs), capable of generating complex and detailed SVGs, from simple icons to intricate anime characters. We also introduce MMSVG-2M, a multimodal dataset with two million richly annotated SVG assets, along with a standardized evaluation protocol for conditional SVG generation tasks. 


## 2. Models Downloading
| Model                       | Download link                   | Size       | Update date |                                                                                     
|-----------------------------|-------------------------------|------------|------|
| OmniSVG1.1_8B | [Huggingface](https://huggingface.co/OmniSVG/OmniSVG1.1_8B)    | 17.2 GB | 2025-12-02  |
| OmniSVG1.1_4B | [Huggingface](https://huggingface.co/OmniSVG/OmniSVG1.1_4B)    | 7.69 GB | 2025-12-02  |
| OmniSVG-3B | [Huggingface](https://huggingface.co/OmniSVG/OmniSVG)    | 8.49 GB | 2025-07-22  | 



##  3. Dependencies and Installation
The dependencies configured according to the following instructions provide an environment equipped for inference

### 3.1 Clone the Repository
```bash
git clone https://github.com/OmniSVG/OmniSVG.git
cd OmniSVG
```

### 3.2 Create Conda Environment
Create and activate a new conda environment with Python 3.10:
```bash
conda create -n omnisvg python=3.10
conda activate omnisvg
```

### 3.3 Install Dependencies

#### System Dependencies
Before installing Python packages, you need to install Cairo library which is required by `CairoSVG` in our dependencies:

**macOS:**
```bash
brew install cairo
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install libcairo2 libcairo2-dev
```

> **Note:** Installing Cairo system library beforehand helps prevent potential build errors when installing `CairoSVG` via pip.

#### Python Dependencies
We have tested our environment with CUDA 12.1. You can install CUDA 12.1 by following the [CUDA Toolkit installation guide](https://developer.nvidia.com/cuda-12-1-0-download-archive).

Install PyTorch with CUDA 12.1 support:
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## 4. Inference Script

|                                                  | GPU Memory Usage | Time per 256/512/1024/2048/4096 tokens |
| ------------------------------------------------ | ---------------- | ----------------- |
| OmniSVG1.1_8B     | 26G              | 5.38/9.02/20.11/40.34/98.11 seconds       |
| OmniSVG1.1_4B     | 17G              | 4.08/8.68/18.07/37.51/82.70 seconds       |
| OmniSVG-3B     | 17G              | 4.08/8.68/18.07/37.51/82.70 seconds       |


<font color="red">**Note: The inference time shown here is measured per OmniSVG SVG tokens, while the inference time reported in our paper is measured per XML code tokens for fair comparison with baseline methods.**</font> 

### Quick Start

**Download Model Weights**

First, install the Hugging Face CLI tool:
```bash
pip install huggingface-hub
```

**Download the model from Hugging Face:**
```bash
# Download OmniSVG1.1-8B
huggingface-cli download OmniSVG/OmniSVG1.1_8B --local-dir /PATH/TO/OmniSVG1.1_8B

# Download OmniSVG1.1-4B
huggingface-cli download OmniSVG/OmniSVG1.1_4B --local-dir /PATH/TO/OmniSVG1.1_4B

# Download OmniSVG-3B (legacy)
huggingface-cli download OmniSVG/OmniSVG --local-dir /PATH/TO/OmniSVG-3B
```

### Text-to-SVG Generation

**Basic usage - Generate SVG from txt file:**
```bash
python inference.py --task text-to-svg --input prompts.txt --output ./output_text --save-all-candidates
```

**Use 4B model:**
```bash
python inference.py --task text-to-svg --input prompts.txt --output ./output_text --model-size 4B --save-all-candidates
```

**Generate more candidates and save PNG:**
```bash
python inference.py --task text-to-svg --input prompts.txt --output ./output_text \
    --num-candidates 8 --save-png --save-all-candidates
```

**Custom generation parameters:**
```bash
python inference.py --task text-to-svg --input prompts.txt --output ./output_text \
    --temperature 0.5 --top-p 0.9 --top-k 50 --repetition-penalty 1.05
```

**Use local model:**
```bash
python inference.py --task text-to-svg --input prompts.txt --output ./output_text \
    --model-path /path/to/qwen --weight-path /path/to/omnisvg
```

### Image-to-SVG Generation

```bash
python inference.py --task image-to-svg --input ./examples --output ./output_image --save-all-candidates
```

### Interactive Demo

We provide an interactive generation interface using Gradio:

- **Local Deployment**
  ```bash
  python app.py
  ```

- **Online Demo**
  
  Try our live demo on [Hugging Face Spaces](https://huggingface.co/spaces/OmniSVG/OmniSVG-3B)

<div align="center">
  <img src="assets/commands.png" alt="cmd" height="256px" />
  <img src="assets/omnisvg-teaser.gif" alt="Demo GIF" height="256px" style="margin-right: 10px;" />
</div>



## 5. Evaluation

We provide **MMSVGBench** for standardized evaluation of SVG generation models.

**Download MMSVGBench:**
```bash
huggingface-cli download OmniSVG/MMSVGBench --repo-type dataset --local-dir /PATH/TO/MMSVGBench
```

### Benchmark Overview

MMSVGBench is a **purely synthetic benchmark** where all prompts and images are generated using GPT models, ensuring the data is **unseen** during model training for fair generalization evaluation. The generation procedure MMSVGBench's prompt is logged, for example the [text2svg prompt log](https://chatgpt.com/share/68f773e9-2814-8002-99ed-5e2980e9b9bf). 

| Task | Complexity Level | Samples | Description |
|------|------------------|---------|-------------|
| Text-to-SVG | Icon | 150 | Simple icons (1-2 elements) |
| Text-to-SVG | Illustration | 150 | Complex illustrations (1-3 interacting elements) |
| Image-to-SVG | Icon | 150 | GPT-4o generated icon images |
| Image-to-SVG | Illustration | 150 | GPT-4o generated illustration images |

**Key Advantages of Synthetic Design:**
- ✅ **True generalization test** — models cannot have seen these samples during training
- ✅ **Controlled diversity** — systematic coverage of styles and semantic categories  
- ✅ **Fairness** — no model has unfair advantage from training data overlap

The evaluation code is available in the `metrics` directory. For more details about MMSVGBench construction and evaluation metrics, please check [MMSVGBench](https://huggingface.co/datasets/OmniSVG/MMSVGBench/blob/main/README.md).



## 6. License
OmniSVG is licensed under the [**Apache License 2.0**](https://www.apache.org/licenses/LICENSE-2.0), while MMSVG dataset is under [**Creative Commons Attribution Non Commercial Share Alike 4.0 License**](https://spdx.org/licenses/CC-BY-NC-SA-4.0). You can find the license files in the respective github and HuggingFace repositories.



## Citation

```bibtex
@article{yang2025omnisvg,
  title={OmniSVG: A Unified Scalable Vector Graphics Generation Model}, 
  author={Yiying Yang and Wei Cheng and Sijin Chen and Xianfang Zeng and Jiaxu Zhang and Liao Wang and Gang Yu and Xinjun Ma and Yu-Gang Jiang},
  journal={arXiv preprint arxiv:2504.06263},
  year={2025}
}
```

## Acknowledgments
We thank the following excellent open-source works:

[IconShop](https://icon-shop.github.io/): is the first advanced work that leverages LLMs to generate monochrome, icon-level SVGs. We referred to its parametric implementation.

Here is the list of highly related concurrent works:

[LLM4SVG](https://arxiv.org/abs/2412.11102): treats SVG coordinates as number strings and predicts decimal part for higher spatial accuracy.

[StarVector](https://starvector.github.io/): equips LLM with an image encoder for Image-to-SVG generation.

## Star History


[![Star History Chart](https://api.star-history.com/svg?repos=OmniSVG/OmniSVG&type=Date)](https://www.star-history.com/#OmniSVG/OmniSVG&Date)

