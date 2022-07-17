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

import argparse
import sys
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
            if anno_shape['name'] == 'polygon':
                px = anno_shape["all_points_x"]
                py = anno_shape["all_points_y"]
                if px in pxs:
                    continue
                else:
                    pxs.append(px)
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                if anno["region_attributes"] != {} and len(px)>=6:
                    category_id = int(list(anno["region_attributes"].values())[0])
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
    
    return augs

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
            evaluator_list.append(COCOEvaluator("set_val", ("segm",), True, output_dir=output_folder))
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
    # Dataset Information
    for d in ["train", "val"]:
        # You could change the address of image directory here
        DatasetCatalog.register("set_" + d, lambda d=d: get_label_dicts(args.data + d))
        MetadataCatalog.get("set_" + d).set(thing_classes=args.name_classes)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
    cfg.DATASETS.TRAIN = ("set_train",)
    cfg.DATASETS.TEST = ("set_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config_file)  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = args.output_dir
    #### Hyperparameters ####
    # 1. Optimizer
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MOMENTUM = args.momentum
    cfg.SOLVER.MAX_ITER = args.num_iter   
    cfg.SOLVER.STEPS = args.step
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    # 2. Data augmentation
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    # 3. Transfer learning
    # Freeze the first several stages so they are not trained.
    # There are 5 stages in ResNet. The first is a convolution, and the following
    # stages are each group of residual blocks.
    cfg.MODEL.BACKBONE.FREEZE_AT = args.freeze_level
    #### end ####
    cfg.SOLVER.CHECKPOINT_PERIOD = args.ckpt_period
    cfg.TEST.EVAL_PERIOD = args.eval_period
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
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config-file", default="LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    # Dataset Information
    parser.add_argument("--data", type=str, default="/mnt/3bcb7712-931d-490e-937d-b920d10759bc/Junz_data/detectron2/Seed/", help="path to dataset")
    parser.add_argument("--num-classes", type=int, default=2, help="total number of dataset classes")
    parser.add_argument("--name-classes", type=list, default=["Healthy", "Diseased"], help="names of dataset classes")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--output-dir", type=str, default='/mnt/3bcb7712-931d-490e-937d-b920d10759bc/Junz_data/detectron2/seeds_output', help="path to output")
    
    #### Hyperparameters ####
    # 1. Optimizer
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-iter", type=int, default=50000, help="total number of iterations")
    parser.add_argument("--step", type=list, default=[10000, 20000, 30000, 40000], help="the steps to adjust the LR (multiply LR with 0.1)")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    # 3. Transfer learning
    # Freeze the first several stages so they are not trained.
    # There are 5 stages in ResNet. The first is a convolution, and the following stages are each group of residual blocks.
    parser.add_argument("--freeze-level", type=int, default=2)
    #### end ####
    parser.add_argument("--ckpt-period", type=int, default=1000, help="period of saving checkpoints")
    parser.add_argument("--eval-period", type=int, default=1000, help="period of evaluation")

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
