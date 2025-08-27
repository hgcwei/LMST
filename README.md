# Language Model-based Self-Training for Biological Sequence Classification


corresponding author: Chao Wei
e-mail:weichao.2022@hbut.edu.cn

Overview
This repository contains the implementation of our paper ​​"Language Model-based Self-Training Reduces Labeled Data Requirements by 99% for Biological Sequence Classification"​. The proposed method combines pre-trained language models (PLMs) with semi-supervised learning (SSL) to achieve state-of-the-art performance in DNA-binding protein (DBP) and non-coding RNA (ncRNA) classification tasks with minimal labeled data requirements.

2. .fasta input
   
    Please input a fa file.

3. Feature Generation

    Generate tfrecord (features) by running main.py in NeuroTIS2.0/NeuroTIS2.0-adaptive grouping (.tfrecord)

4. TIS Prediction
   
    Predict TIS by running tisTest.py in NeuroTIS2.0/NeuroTIS2.0 frame-specific CNN
