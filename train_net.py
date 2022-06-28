#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
import numpy as np
import json, cv2, random
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    inference_on_dataset,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.solver import get_default_optimizer_params
from detectron2.data import detection_utils as utils
import copy

def get_label_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        #if int(v["filename"][5:9]) >= 117 and int(v["filename"][5:9]) <= 145:
            
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        pxs = []
        for anno in annos:
            #assert not anno["region_attributes"]
            anno_shape = anno["shape_attributes"]
            if anno_shape['name'] == 'polyline':
                px = anno_shape["all_points_x"]
                py = anno_shape["all_points_y"]
                if px in pxs:
                    continue
                else:
                    pxs.append(px)
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                if anno["region_attributes"] != {} and len(px)>=6:
                    category_id = int(anno["region_attributes"]["type"])
                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": int(category_id),
                    }
                    objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        ),
        T.RandomFlip(prob=0.5),
        # T.RandomBrightness(0.9, 1.1),
        # T.RandomContrast(0.8, 1.2),
        # T.RandomSaturation(0.9, 1.1),
        # T.RandomLighting(0.1),
        T.RandomRotation([-359, 359], expand=True, center = [[0.4, 0.4], [0.6, 0.6]]),
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
            )
        )
    
    return augs
 # Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # See "Data Augmentation" tutorial for details usage
    auginput = T.AugInput(image)
    transform = T.Resize((800, 800))(auginput)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
       "image": image,
       "instances": utils.annotations_to_instances(annos, image.shape[1:])
    }
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = "coco"
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator("seeds_val", ("segm",), True, output_dir=output_folder))
        return evaluator_list[0]

    # @classmethod
    # def build_optimizer(cls, cfg, model):
    #     params = get_default_optimizer_params(
    #         model,
    #         base_lr=cfg.SOLVER.BASE_LR,
    #         weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    #         weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
    #         bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
    #         weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    #     ) 
    #     optimizer = torch.optim.Adam(
    #         params, lr=cfg.SOLVER.BASE_LR
    #     )
    #     return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    for d in ["train_original", "val_original"]:
        # You could change the address of image directory here
        DatasetCatalog.register("seeds_" + d, lambda d=d: get_label_dicts("/mnt/3bcb7712-931d-490e-937d-b920d10759bc/Junz_data/detectron2/Seed/" + d))
        MetadataCatalog.get("seeds_" + d).set(thing_classes=["Healthy", "Diseased"])
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("seeds_train_original",)
    cfg.DATASETS.TEST = ("seeds_val_original",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml")  # Let training initialize from model zoo
    #### Hyperparameters ####
    # 1. Optimizer
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.MAX_ITER = 50000   
    cfg.SOLVER.STEPS = [7000, 20000, 30000, 40000] # define the steps to adjust the LR
    cfg.SOLVER.WEIGHT_DECAY = 0.001
    # 2. Data augmentation
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.INPUT.CROP.ENABLED = False
    # Cropping type:
    # - "relative" crop (H * CROP.SIZE[0], W * CROP.SIZE[1]) part of an input of size (H, W)
    # - "relative_range" uniformly sample relative crop size from between [CROP.SIZE[0], [CROP.SIZE[1]].
    #   and  [1, 1] and use it as in "relative" scenario.
    # - "absolute" crop part of an input with absolute size: (CROP.SIZE[0], CROP.SIZE[1]).
    # - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
    #   [CROP.SIZE[0], min(H, CROP.SIZE[1])] and W_crop in [CROP.SIZE[0], min(W, CROP.SIZE[1])]
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    cfg.INPUT.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    # 3. Transfer learning
    # Freeze the first several stages so they are not trained.
    # There are 5 stages in ResNet. The first is a convolution, and the following
    # stages are each group of residual blocks.
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    #### end ####       
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.OUTPUT_DIR = '/mnt/3bcb7712-931d-490e-937d-b920d10759bc/Junz_data/detectron2/seeds_output'
    #cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    # if cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
    #     )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
