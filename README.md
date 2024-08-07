# IRNet (TOMM-2024)

ACM TOMM 2024: [Rethinking Feature Mining for Light Field Salient Object Detection](https://dl.acm.org/doi/10.1145/3676967)

## Abstract
Light field salient object detection (LF SOD) has recently received increasing attention. However, most current works typically rely on an individual focal stack backbone for feature extraction. This manner ignores the characteristic of blurred saliency-related regions and contour within focal slices, resulting in insufficient or even inaccurate saliency responses. Aiming at addressing this issue, we rethink the feature mining (i.e., exploration) within focal slices, and focus on exploiting informative focal slice features and fully leveraging contour information for accurate LF SOD. First, we observe that the geometric relation between different regions within the focal slices is conducive to useful saliency feature mining if utilized properly. In light of this, we propose an implicit graph learning (IGL) approach. The IGL constructs graph structures to propagate informative geometric relations within the focal slices and all-focus features, and promotes crucial and discriminative focal stack feature mining via graph feature distillation. Second, unlike previous works that rarely utilize contour information, we propose a reciprocal refinement fusion (RRF) strategy. This strategy encourages saliency features and object contour cues to effectively complement each other. Furthermore, a contour hint injection mechanism is introduced to refine the feature expressions. Extensive experiments showcase the superiority of our approach over previous state-of-the-art models with an efficient real-time inference speed. 


## Requirements
* Pytorch 1.3.0+   

* Torchvision   

* PIL   

* Numpy


## Results
* Light field saliency maps of HFUT, DUTLF, and LFSD datasets can be downloaded from [here](https://pan.baidu.com/s/1QvbKM_t2SMaQKL6sh5HqXw) [code: wtz3]  

* Pre-trained model weights from [here](https://pan.baidu.com/s/1Lk-rxJo6swf3sjW2t6nf5Q) [code: zo14]


## Citation
Please consider citing our work if you use this repository in your research.
```
@ARTICLE{IRNet_TOMM24,
  author={Liao, Guibiao and Gao, Wei},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  title={Rethinking Feature Mining for Light Field Salient Object Detection}, 
  year={2024},
  publisher={ACM New York, NY}
}
```
