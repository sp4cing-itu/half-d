# HalF&D: A Parameter Efficient Small Object Detection Approach

This repository contains the introduction and implementation details for the paper **"HalF&D: A Parameter Efficient Small Object Detection Approach"**.

## Abstract

Small-object detection in aerial imagery is a challenging problem due to large variations in object scale, high density, and limited spatial resolution. The Focus-and-Detect (F-D) framework addresses this problem by dividing the process into two stages: a Focus stage that predicts object-dense regions, and a Detect stage that performs fine-grained detection within these focal crops. In this study, we present HalF&D, a parameter-efficient variant of the F-D framework that integrates the HaLVIT (Half of the Weights are Enough) mechanism into both stages. HaLVIT reduces parameters by sharing a single projection matrix and its transpose across bottleneck blocks within each residual stage, thereby removing redundant weight sets without altering network depth or structure. Applied to ResNet-50 in the Focus stage and ResNeXt-101-32x8d in the Detect stage, HalF&D achieves significant compression while maintaining usable accuracy on the VisDrone dataset.

## Web

* epapers2.org/iscas2026/ESR/paper_details.php?paper_id=1770

## Introduction

**What the method does:**
Instead of using separate weight matrices for consecutive linear transformations in every bottleneck block, the method uses a **single weight matrix ($W$)** per stage. It uses the matrix $W$ for the first transformation (reduction) and its **transpose ($W^T$)** for the second transformation (expansion).

## Method

The method applies this stagewise weight-sharing strategy to both the **Focus stage (ResNet-50)** and the **Detect stage (ResNeXt-101-32x8d)** of the Focus-and-Detect pipeline.


########
Signal Processing for Computational Intelligence Research Group (SP4CING).
