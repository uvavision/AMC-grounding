## [Improving Visual Grounding by Encouraging Consistent Gradient-based Explanations](https://arxiv.org/abs/2206.15462)
[Ziyan Yang](https://ziyanyang.github.io/), [Kushal Kafle](https://kushalkafle.com/), [Franck Dernoncourt](https://research.adobe.com/person/franck-dernoncourt/), [Vicente Ordonez](https://www.cs.rice.edu/~vo9/), CVPR 2023

If you have any questions, please email ziyan.yang@rice.edu

:sparkles:  We make a [demo](https://vislang.ai/amc) for this work! Feel free to try it!  

### Abstract
We propose a margin-based loss for vision-language model pretraining that encourages gradient-based explanations that are consistent with region-level annotations. We refer to this objective as Attention Mask Consistency (AMC) and demonstrate that it produces superior visual grounding performance compared to models that rely instead on region-level annotations for explicitly training an object detector such as Faster R-CNN. AMC works by encouraging gradient-based explanation masks that focus their attention scores mostly within annotated regions of interest for images that contain such annotations. Particularly, a model trained with AMC on top of standard vision-language modeling objectives obtains a state-of-the-art accuracy of 86.59% in the Flickr30k visual grounding benchmark, an absolute improvement of 5.48% when compared to the best previous model. Our approach also performs exceedingly well on established benchmarks for referring expression comprehension and offers the added benefit by design of gradient-based explanations that better align with human annotations.
### Requirements
- Python 3.8
- PyTorch 1.8.0+cu111
- transformers==4.8.1
- Numpy, scikit-image, opencv-python, pillow, matplotlib, timm

### Data
-  Visual Genome (VG) images: Please download [VG](https://visualgenome.org/) images first. 
-  Annotations: Please download our pre-processed [text annotations](https://drive.google.com/drive/folders/1XhFVjJ2cm2HNeNVOZrUrPG_MpprHLWgv?usp=share_link) for VG images. You may need to modify the image path in each sample to load images.

### Train
After downloading the pre-trained [ALBEF-14M](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth) model, You can run the following command to train the model:
```Shell
# Train the model using bounding box annotations from VG
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain.py --config configs/Pretrain.yaml --output_dir ALBEF_Grounding --checkpoint ALBEF.pth 
```

### Evaluation
To evaluate Flickr30k, please follow [info-ground](https://github.com/BigRedT/info-ground) to process the data.

You can run the following command to evaluate the RefCOCO+, RefCLEF and Flickr30k datasets using all the checkpoints in your ALBEF_Grounding folder:
```Shell
CUDA_VISIBLE_DEVICES=1 python grounding_eval_singlegpu.py --checkpoint ALBEF_Grounding --output_dir ALBEF_Grounding/refcoco_results --config configs/Grounding_refcoco.yaml

CUDA_VISIBLE_DEVICES=1 python grounding_eval_singlegpu_refclef.py --checkpoint ALBEF_Grounding --output_dir ALBEF_Grounding/refclef_results --config configs/Grounding_refclef.yaml

CUDA_VISIBLE_DEVICES=1 python grounding_eval_singlegpu_flickr.py --checkpoint ALBEF_Grounding --output_dir ALBEF_Grounding/flickr_results --config configs/Grounding_flickr.yaml
```

You can also download these [checkpoints](https://drive.google.com/drive/folders/1syngIWXbySzbcb7lZnmoL7_-H6RAeb3H?usp=sharing) and put them into the corresponding folder to reproduce our results:
```Shell
CUDA_VISIBLE_DEVICES=1 python grounding_eval_singlegpu.py --checkpoint best_refcoco.pth --output_dir best_refcoco_results --config configs/Grounding_refcoco.yaml

CUDA_VISIBLE_DEVICES=1 python grounding_eval_singlegpu_refclef.py --checkpoint best_refclef.pth --output_dir best_refclef_results --config configs/Grounding_refclef.yaml

CUDA_VISIBLE_DEVICES=1 python grounding_eval_singlegpu_flickr.py --checkpoint best_flickr.pth --output_dir best_flickr_results --config configs/Grounding_flickr.yaml
```

### Citing
If you find our paper/code useful, please consider citing:

```
@inproceedings{yang2023improving,
  title={Improving Visual Grounding by Encouraging Consistent Gradient-based Explanations},
  author={Yang, Ziyan and Kafle, Kushal and Dernoncourt, Franck and Ordonez, Vicente},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19165--19174},
  year={2023}
}
```

### Acknowledgement
The implementation of AMC relies on the code from [ALBEF](https://github.com/salesforce/ALBEF/tree/main). We would like to thank the authors who have open-sourced their work and made it available to the community. 
