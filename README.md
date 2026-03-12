# 👁️ Giving Eyes to a Blind Medical LLM

### VoRA + BioMistral-7B for Radiology Visual Question Answering

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

*Can a text-only medical LLM learn to interpret chest X-rays through a small LoRA adapter alone?*

</div>

---

## What is this?

This project explores a simple but powerful idea: instead of building a dedicated multimodal medical model from scratch, what if we could **inject vision capabilities into an existing text-only medical LLM** using nothing more than a lightweight adapter?

[VoRA (Vision as LoRA)](https://arxiv.org/abs/2503.20680) by Wang et al. (2025) showed that this is possible for general-domain models. We take it a step further and apply VoRA to **BioMistral-7B**, a medical LLM pre-trained on PubMed Central, to make it understand radiology images and answer clinical questions about them.

To our knowledge, this is the **first attempt to apply VoRA in a biomedical context**.

## The Pipeline
```
                    ┌─────────────────────────────┐
                    │   BioMistral-7B (frozen)     │
                    │   Text-only medical LLM      │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     ┌────────▼────────┐  ┌───────▼────────┐  ┌────────▼────────┐
     │  LoRA Adapters   │  │ Vision Embed   │  │  ViT Teacher    │
     │  (trainable)     │  │ (trainable)    │  │  (frozen)       │
     │  ~336M params    │  │ ~4.4M params   │  │  AIMv2-Huge     │
     └────────┬────────┘  └───────┬────────┘  └────────┬────────┘
              │                    │                     │
              │         Block-wise Distillation          │
              │         (cosine similarity loss)         │
              └────────────────────┼────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Stage 1: Pre-training      │
                    │   ROCO dataset (~65K images)  │
                    │   Image-caption pairs         │
                    └──────────────┬──────────────┘
                                   │
                         Merge LoRA → base
                                   │
                    ┌──────────────▼──────────────┐
                    │   Stage 2: Fine-tuning       │
                    │   VQA-RAD (1,793 QA pairs)   │
                    │   Radiology VQA task          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   BioMistral-7B + Vision     │
                    │   "Can see chest X-rays"     │
                    └─────────────────────────────┘
```

**Stage 1** teaches BioMistral to process images by distilling knowledge from a pre-trained ViT into LoRA layers, using radiology image-caption pairs from ROCO.

**Stage 2** merges the LoRA weights back into the base model and fine-tunes everything on VQA-RAD for radiology question answering.

## Three Key Ideas from VoRA

| Component | What it does | Why it matters |
|---|---|---|
| **Vision as LoRA** | Encodes vision through small adapter weights attached to the LLM | Keeps the base LLM frozen, preventing catastrophic forgetting of medical knowledge |
| **Block-wise distillation** | Aligns LLM hidden states with ViT features layer by layer | Transfers visual knowledge efficiently, reducing data needs by ~35% |
| **Bidirectional attention for vision** | Image patches attend to all other patches (not just left-to-right) | Captures global spatial context in medical images |

## Datasets

| Dataset | Size | Role |
|---|---|---|
| [ROCO](https://github.com/razorx89/roco-dataset) | ~65K radiology images + captions | Pre-training: inject visual knowledge |
| [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | 1,793 train / 451 test QA pairs | Fine-tuning & evaluation: radiology VQA |

## Repository Structure
```
.
├── VoRA_BioMistral_MedicalVQA.ipynb   # Main notebook (runs on Colab)
├── README.md
├── figures/                            # Generated plots and visualizations
└── references/                         # VoRA paper and related materials
```

## Running the Notebook

**Requirements:** Google Colab with A100 GPU (recommended) or a dedicated GPU cluster with 40GB+ VRAM.

1. Open `VoRA_BioMistral_MedicalVQA.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → A100)
3. Run all cells sequentially

The notebook handles all dependency installation, data downloading, and model setup automatically.

## Built With

- [BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B) (Labrak et al., ACL 2024) as the base medical LLM
- [VoRA](https://github.com/Hon-Wong/VoRA) (Wang et al., ICML 2025) for vision injection via LoRA
- [AIMv2-Huge](https://huggingface.co/apple/aimv2-huge-patch14-448) as the ViT teacher for distillation
- [ROCO](https://github.com/razorx89/roco-dataset) (Pelka et al., 2018) for pre-training data
- [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) (Lau et al., 2018) for evaluation

## References
```bibtex
@article{wang2025vora,
  title={Vision as LoRA},
  author={Wang, Han and Ye, Yongjie and Li, Bingru and Nie, Yuxiang and 
          Lu, Jinghui and Tang, Jingqun and Wang, Yanjie and Huang, Can},
  journal={arXiv preprint arXiv:2503.20680},
  year={2025}
}

@inproceedings{labrak2024biomistral,
  title={BioMistral: A Collection of Open-Source Pretrained Large Language 
         Models for Medical Domains},
  author={Labrak, Yanis and Bazoge, Adrien and Mothe, Josiane and others},
  booktitle={Proceedings of ACL},
  year={2024}
}

@article{pelka2018roco,
  title={Radiology Objects in COntext (ROCO): A Multimodal Image Dataset},
  author={Pelka, Obioma and Koitka, Sven and R{\"u}ckert, Johannes and others},
  journal={arXiv preprint arXiv:1809.04420},
  year={2018}
}

@article{lau2018vqarad,
  title={A Dataset of Clinically Generated Visual Questions and Answers 
         about Radiology Images},
  author={Lau, Jason J and Gayen, Soumya and Ben Abacha, Asma and Demner-Fushman, Dina},
  journal={Scientific Data},
  year={2018}
}
```

## Acknowledgments

This project is developed as part of coursework og Big Data and Text Mining at the **University of Bologna**, supervised by **Dr. Giacomo Frisoni** and **Prof. Gianluca Moro**.

---

<div align="center">
<i>Luwam Major Kefali · University of Bologna · 2025</i>
</div>
