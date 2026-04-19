# Mask Architecture Anomaly Segmentation for Road Scenes

Existing deep networks, when deployed in open-world settings, perform poorly on **Unknown/Anomaly/Out-of-Distribution (OoD)** objects that were not present during training. Detecting OoD objects becomes critical for autonomous driving applications and branches of computer vision problems such as continual learning and open-world problems.

The goal of this project is to build, train, and test an **anomaly segmentation model** to segment anomalies on road scenes.

---

## Project Steps

### 1. Study ERFNet for Semantic Segmentation
Study a simple and efficient model for real-time semantic segmentation: **ERFNet** — Efficient Residual Factorized ConvNet.

### 2. Study Mask Architecture Literature
Understand the core concepts behind mask architectures, starting from **MaskFormer**, its evolution **Mask2Former**, and the use of **DINOv2** in the **EoMT** model (*Your ViT is Secretly an Image Segmentation Model*, CVPR 2025).

### 3. Understand Anomaly Segmentation and Post-Hoc Methods
Get familiar with the anomaly segmentation task, the relevant benchmarks (SMIYC, Fishyscapes, Road Anomaly), and the most important post-hoc anomaly scoring methods.

### 4. Pixel-based Baselines (ERFNet)
Apply post-hoc anomaly scoring methods on top of a pretrained ERFNet model:
- **MSP** — Maximum Softmax Probability
- **Max Logit**
- **Max Entropy**

### 5. Mask-based Baselines (EoMT)
Evaluate the EoMT model on the same benchmarks. Since EoMT is a mask architecture, its output differs from pixel-based models, enabling an additional method:
- **MSP**, **Max Logit**, **Max Entropy**
- **RbA** — Rejected by All (mask-architecture specific)
- **Temperature Scaling** — confidence calibration to improve anomaly detection

### 6. Extensions
Propose and implement additional analyses or improvements, such as:
- Fine-tuning with anomaly-specific losses (Enhanced Isotropy Maximization, Logit Normalization)
- Outlier Exposure (cut-paste objects from COCO onto Cityscapes)
- RbA with Outlier Exposure
- LoRA fine-tuning for resource-constrained settings

---

## References

1. SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation
2. The Fishyscapes Benchmark: Anomaly Detection for Semantic Segmentation
3. Per-Pixel Classification is Not All You Need for Semantic Segmentation — MaskFormer
4. Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)
5. DINOv2: Learning Robust Visual Features without Supervision
6. Your ViT is Secretly an Image Segmentation Model — EoMT (CVPR 2025)
7. RbA: Segmenting Unknown Regions Rejected by All
8. Scaling Out-of-Distribution Detection for Real-World Settings
9. LoRA: Low-Rank Adaptation of Large Language Models
10. ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation
