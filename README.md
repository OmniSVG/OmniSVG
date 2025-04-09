<div align= "center">
    <h1> Official repo for OmniSVG</h1>

</div>

<p align="center">
  <h3 align="center"><strong>OmniSVG: A Unified Scalable Vector Graphics Generation Model</strong></h3>


<div align="center">

<a href='https://arxiv.org/abs/2504.06263'><img src='https://img.shields.io/badge/arXiv-2504.06263-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://omnisvg.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/OmniSVG"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/OmniSVG"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset%20-HF-orange"></a>

</div>

## 🏃 Intro OmniSVG

**OmniSVG** is the first family of end-to-end multimodal SVG generators that leverage pre-trained Vision-Language Models (VLMs), capable of generating complex and detailed SVGs, from simple icons to intricate anime characters.


<p align="center">
    <img src="assets/omnisvg-teaser.gif" alt="Demo GIF" width="512px" />
</p>
<p align="center">
    <img src="assets/OmniSVG-demo-gen-proc-anime-1080.gif" alt="Demo GIF" width="512px" />
</p>

## 🚩 News

- [2025/04/09] Release MMSVG-Icon and MMSVG-Illustration 🤗[Dataset](https://huggingface.co/OmniSVG).
- [2025/04/09] Upload paper and init project.



## To Do List
- [ ] Code and Pretrained Models Release
- [ ] MMSVG-Character Release
- [x] MMSVG-Icon and MMSVG-Illustration Dataset Release
- [x] Project Page & Technical Report

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

