# Multi-Scale-Edge-Guided-Context-Aggregation-Network-for-Single-Image-Deraining
Multi-Scale Edge-Guided Context Aggregation Network for Single Image Deraining

# Authors: Jun Wang, Huiyuan Zuo, Zaiyu Pan, Shuyu Han

# Abstract
Deep Convolutional Neural Networks (DCNNs) have shown remarkable results in the task of image deraining. However, many existing methods for single image deraining do not take into account the restoration of edge textures in the image. Some approaches use a backbone network to handle both image deraining and detail restoration. To overcome these limitations, we have developed a dual-branch structure called Multi-Scale Edge-Guided Context Aggregation Network (MSEGCA-Net), which employs a coarse-to-fine approach. This enables us to achieve superior deraining performance while preserving image resolution. Firstly, we present a main image deraining branch to obtain coarse deraining information from the rainy image, as well as an auxiliary edge texture detection branch to obtain edge texture information. Then, different from previous methods relying on direct guidancefrom the edge textures, we design a novel connection block, named Edge-Guided Context Aggregation Block (EGCAB), which aggregates the coarse deraining information and the edge texture information. Finally, the aggregated information guides the main image deraining branch to obtain the final derained image. We have validated our approach on three datasets (Rain200L, Rain1200, and Rain1400). Both quantitative and qualitative comparisons show that our approach outperforms the state-of-the-art deraining methods in terms of the deraining robustness and detail accuracy.

![image](https://github.com/B201cv/Multi-Scale-Edge-Guided-Context-Aggregation-Network-for-Single-Image-Deraining/assets/150791781/3dd9fa5f-b6c3-4623-bbce-1b235bd6bb67)


# Results
![image](https://github.com/B201cv/Multi-Scale-Edge-Guided-Context-Aggregation-Network-for-Single-Image-Deraining/assets/150791781/4612dead-13d2-4904-9d45-060de5d0cc6e)




