
# Faster R-CNN

Faster R-CNN is a two-stage object detection algorithm.
It adopts Region Proposal Network (RPN) and shares the image features across ROIs.
A Faster R-CNN contains the following modules:
- A backbone network (typically a [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) equipped with a [FPN](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)) to extract the image-level features;
- An RPN to generate boxes proposals;
- An ROI head for classification and box coordinate regression.

Please refer to the following paper for more details:

[Ren S, He K, Girshick R, et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. TPAMI 2015.](https://arxiv.org/abs/1506.01497): 

Note that FPN and ROI Align are not included in the original implementation.

# Files Description

Following is the structure of the code framework:

```shell
.
└─project1_fasterrcnn
  ├─README.md                           // description of the framework
  ├─scripts
    └─run_eval_ascend.sh                // shell script for evaluation (Ascend)
    └─run_eval_gpu.sh                   // shell script for evaluation (GPU)
    └─run_eval_cpu.sh                   // shell script for evaluation (CPU)
  ├─src
    ├─FasterRcnn
      ├─__init__.py                     // init file
      ├─anchor_generator.py             // to generate anchors
      ├─bbox_assign_sample.py           // sampler for the first stage (RPN)
      ├─bbox_assign_sample_stage2.py    // sampler for the second stage (ROI Head)
      ├─faster_rcnn_r50.py              // Faster R-CNN
      ├─fpn_neck.py                     // Feature Pyramid Network
      ├─proposal_generator.py           // to generate proposal from RPN\'s outputs
      ├─rcnn.py                         // ROI Head
      ├─resnet50.py                     // ResNet-50
      ├─roi_align.py                    // ROI Align module
      └─rpn.py                          // Region Proposal Network
    ├─config.py               // configuration file 
    ├─dataset.py              // to create and process the dataset
    ├─lr_schedule.py          // to change the learning rate
    ├─network_define.py       // FasterRcnn training network wrapper
    └─util.py                 // other useful functions
  ├─coco
    ├─annotations             // annotations of the images
    ├─demo_val2017            // the images (sampled from COCO dataset)
  ├─cocoapi                   // evaluation API file, you need to install it first
  ├─pretrained_faster_rcnn.ckpt         // pre-trained model parameters
  ├─eval.py                   // evaluation script
  └─train.py                  // training script
```

# Environmental preparation
```shell
pip install -r requirements.txt

# install COCO evaluation API
cd cocoapi/PythonAPI
python setup.py install
```

# How to evaluate
Execute the following command:
```shell
# evaluate (on Ascend/GPU/CPU. Choose one according to your device.)
sh ./scripts/run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
sh ./scripts/run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
sh ./scripts/run_eval_cpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

# Example of evaluation results

The evaluation result will be saved in `eval/log`. It look like this:

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.229
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
```
