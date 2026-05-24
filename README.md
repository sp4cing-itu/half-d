# HalF&D: A Parameter Efficient Small Object Detection Approach

[cite_start]This repository contains the official implementation for the paper **"HalF&D: A Parameter Efficient Small Object Detection Approach"**[cite: 1, 2].

## Abstract
[cite_start]Small-object detection in aerial imagery is a challenging problem due to large variations in object scale, high density, and limited spatial resolution[cite: 13]. [cite_start]The Focus-and-Detect (F-D) framework addresses this problem by dividing the process into two stages: a Focus stage that predicts object-dense regions, and a Detect stage that performs fine-grained detection within these focal crops[cite: 14]. [cite_start]In this study, we present HalF&D, a parameter-efficient variant of the F-D framework that integrates the HaLVIT (Half of the Weights are Enough) mechanism into both stages[cite: 15]. [cite_start]Applied to ResNet-50 in the Focus stage and ResNeXt-101-32x8d in the Detect stage, HalF&D achieves significant compression while maintaining usable accuracy on the VisDrone dataset[cite: 17]. [cite_start]Specifically, the Focus network achieves 47.9% mAP@50 with a 47% parameter reduction, and the Detect network reaches 32.2% mAP with more than 70% fewer parameters[cite: 18].

## Introduction & Method

**What the method does:**
[cite_start]HalF&D improves the parameter efficiency of the two-stage Focus-and-Detect (F-D) pipeline without sacrificing accuracy[cite: 24]. [cite_start]It achieves this by integrating the HaLVIT CNN pathway into both the Focus and Detect stages[cite: 36].

* [cite_start]**Focus Stage:** We apply within-stage sharing of the 1x1 bottleneck weights W and W^T to the ResNet-50 backbone[cite: 37]. [cite_start]HaLVIT is applied only in the third and fourth residual stages, collapsing the heavy 1x1 projections into a single shared matrix per stage[cite: 41, 76]. [cite_start]Meanwhile, the 3x3 convolutions remain unique to their specific block[cite: 74].
* [cite_start]**Detect Stage:** We apply the same weight-sharing strategy to the ResNeXt-101 (32x8d) backbone running on cropped, high-resolution focal regions[cite: 38, 79]. [cite_start]A single stage-level matrix W is used at block entry, and features are projected back with W^T[cite: 85]. [cite_start]The grouped 3x3 convolutions remain block-specific to preserve the cardinality of the ResNeXt architecture[cite: 86, 111].

## Key Results

* [cite_start]**Focus Stage (ResNet-50):** Reduces backbone parameters by 47.3% (from 25.6 M to 13.4 M)[cite: 167].
* [cite_start]**Detect Stage (ResNeXt-101-32x8d):** Reduces backbone parameters by 70.9% (from 88.8 M to 25.9 M)[cite: 167].
* [cite_start]**Overall:** Achieves over 60% total model-size reduction while maintaining mAP values comparable to other specialized region-based detectors[cite: 156].

## Installation

*(Note for the authors: You can add your specific environment requirements here)*

```bash
# Clone the repository
git clone [https://github.com/your-username/halfd.git](https://github.com/your-username/halfd.git)
cd halfd

# Install dependencies
pip install -r requirements.txt
