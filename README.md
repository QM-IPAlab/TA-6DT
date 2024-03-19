# TA-6DT  
Test-time Adaptation for 6D Pose Tracking.

[Project Page](https://bartektian.github.io/TA-6DT.html) 


## Overview
This repository is the implementation code of the paper "Test-time adaptation for 6D pose Tracking".
In this repo, we provide our full implementation code of training/evaluation/inference of the TA-6DT model.

## Requirements

* Python 2.7/3.5/3.6
* [PyTorch 0.4.1](https://pytorch.org/) 
* PIL
* scipy
* numpy
* logging
* CUDA 7.5/8.0/9.0

## Dataset

This work is evaluated on two datasets:
* [NOCS-REAL275](https://github.com/hughw19/NOCS_CVPR2019)
* [Hand-eye Camera dataset](https://zenodo.org/record/8172205)



## Training
For NOCS-REAL275 dataset. Please go to /NOCS_REAL/ folder and run:
```bash
python train.py --category NUM --dataset_root YOUR_NOCS_DATASET_PATH
```
where `NUM` is the category number you want to train.


Hand-eye Camera dataset. Please go to /Hand_eye/ folder and run:
```bash
python real_train.py --object NAME --dataset_root YOUR_HAND-EYE_DATASET_PATH --chechpoint YOUR_SAVED_MODEL_PATH
```
where `NAME` is the object name that you want to adapt to.

**Checkpoints and Resuming**: 
After the training of each epoch, a `model_current_(category_name).pth` checkpoint will be saved. 
You can use it to resume the training. 
We test the model after each epoch and save the model has the best performance so far, as `model_(epoch)_(best_score)_(category_name).pth`, which can be used for the evaluation.

## Evaluation
For NOCS-REAL275. Please go to /NOCS_REAL/ folder and run:
```bash
python eval.py --checkpoint YOUR_SAVED_MODEL_PATH
```
This code is used to visualize the predicted 6D pose information, and save the predicted poses in a .txt file for compute accuracy score. 


Hand-eye Camera. Please go to /Hand_eye/ folder and run:
```bash
python test.py --checkpoint YOUR_SAVED_MODEL_PATH
```

## Accuracy calculation
```bash
python benchmark.py
```
This code is used to compute the accuracy score, including 5cm 5degree, IoU25, Rotation error, and Translation error


## Trained Checkpoints
You can download our trained checkpoints used for benchmarking and real robot experiments 
from [Link](https://xxxx).
