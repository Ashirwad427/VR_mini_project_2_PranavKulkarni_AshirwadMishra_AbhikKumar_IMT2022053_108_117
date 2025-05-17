# VR_mini_project_2_PranavKulkarni_AshirwadMishra_AbhikKumar_IMT2022053_108_117
This repository includes all the notebooks, datasets and scripts used in AIM825 - Visual Recognition Mini Project 2 on Multimodal Visual Question Answering (VQA).

## Team Members
- Pranav Anand Kulkarni (IMT2022053)
- Ashirwad Mishra (IMT2022108)
- Abhik Kumar (IMT2022117)

## Curated Dataset
- `Curated_VQA_Dataset_1.csv`: VQA curated dataset from ABO dataset.
- `blip_vqa_train.csv`: Used for training the VQA models
- `blip_vqa_test.csv` & `blip_vqa_val.csv`: Used for testing the VQA models (before and after fine-tuning)

## Inference Script
-  `inference.py`: Script to use the final fine-tuned model to make predictions over unseen test data
-  `requirements.txt`: Python dependencies required to run the inference script.

## Notebooks

### └── Baseline Evaluation
- `llava-baseline-evaluation.ipynb`: Baseline evaluation on non fine-tuned LLaVA model.

### └── Dataset Curation
- `data-curation-2.ipynb`:
- `df_manipulation_split.ipynb`:
- `gemini-structured-output.ipynb`:
- `vr-json2df.ipynb`:
 
### └── Fine Tuning with LoRA
- `vrmp2-blip-fine-tuning.ipynb`: Fine-Tuning BLIP model using LoRA
- `vrmp2-paligemma-fine-tuning.ipynb`: Fine-Tuning PaliGemma model using LoRA

### └── Evaluation on Fine-Tuned Models
- `vrmp2-blip-finetuned-prediction.ipynb`: Prediction from fine-tuned BLIP model
- `vrmp2-paligemma-finetuned-prediction.ipynb`: Prediction from fine-tuned PaliGemma model

## Report

