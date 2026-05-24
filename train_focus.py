# -*- coding: utf-8 -*-
"""
================================================================================
Focus-and-Detect Framework: Focus Stage Automated Training Pipeline
================================================================================

Description:
This script automates the complete training pipeline for the "Focus" stage 
of the Focus-and-Detect framework. It trains a Generalized Focal Loss (GFL) 
model using a custom ResNet-50 HalVit backbone to identify and crop focal 
regions from high-resolution aerial images.

Pipeline Overview:
------------------
* STEP 1: Environment Setup
  Sets up the MMDetection framework with PyTorch and MMCV, ensuring strict 
  dependency compatibility (e.g., pinning NumPy < 2.0).

* STEP 2: Dataset Preparation
  Extracts raw VisDrone images and custom bounding box annotations, converts 
  them into a single-class ("focal_region") COCO format, and deterministically 
  splits them into a 90% training and 10% validation set with zero overlap 
  to prevent data leakage.

* STEP 3: Model Architecture & Configuration
  Integrates the custom ResNet-50 HalVit backbone into MMDetection's registry. 
  Generates Python configuration files that implement multi-scale training 
  (dynamically resizing between 400x1400 and 1200x1400) to improve scale invariance.

* STEP 4: Training Execution
  Initiates the MMDetection training loop, saving the trained model checkpoints 
  and metrics directly to Google Drive.
================================================================================
"""

import os
import subprocess
import sys
import time
import shutil

def run_command(command, **kwargs):
    """
    Executes a shell command and streams the standard output in real-time.
    Raises a CalledProcessError if the command execution fails.
    """
    print(f"\n> Executing: {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True, bufsize=1, **kwargs)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

try:
    # --- STEP 1: ENVIRONMENT SETUP ---
    print("="*80)
    print("STEP 1: ENVIRONMENT SETUP STARTING...")
    print("="*80)

    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    # Prevent numpy 2.x compatibility issues with pre-compiled ML libraries
    run_command("pip install 'numpy<2.0'")
    # Install specific PyTorch version mapping to CUDA 12.1
    run_command("pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121")
    # Install compatible MMCV version via OpenMMLab
    run_command("pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html")
    run_command("pip install mmengine")

    # Clone the MMDetection repository if it does not already exist
    if not os.path.exists('/content/mmdetection'):
        run_command("git clone https://github.com/open-mmlab/mmdetection.git /content/mmdetection")

    os.chdir('/content/mmdetection')
    
    # Strip numpy from requirements.txt to prevent it from overriding our specific <2.0 installation
    with open('requirements.txt', 'r') as f: lines = f.readlines()
    with open('requirements.txt', 'w') as f:
        for line in lines:
            if 'numpy' not in line: f.write(line)
            
    # Install MMDetection in editable mode
    run_command("pip install -e .")

    print("\nSTEP 1 COMPLETED: Environment setup successfully.")
    time.sleep(1)

    # --- STEP 2: DATASET PREPARATION AND CONVERSION ---
    print("\n" + "="*80)
    print("STEP 2: DATASET PREPARATION AND CONVERSION STARTING...")
    print("="*80)

    import json
    import cv2
    from tqdm import tqdm
    import datetime

    # Define standard directories for the dataset
    DATA_DIR = '/content/mmdetection/data/visdrone_gmm/'
    IMAGE_DIR = os.path.join(DATA_DIR, 'images')
    ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)

    # Extract images and annotations directly from Google Drive
    run_command("unzip -q -o /content/drive/MyDrive/visdrone_gmm/images.zip -d " + IMAGE_DIR)
    run_command("unzip -q -o /content/drive/MyDrive/visdrone_gmm/annotations.zip -d " + ANNOTATION_DIR)

    def flatten_directory(base_path, extension):
        """
        Recursively searches for files with a specific extension and moves them 
        to the root base_path. Resolves issues caused by nested zip extractions.
        """
        src_path = base_path
        for root, dirs, files in os.walk(base_path):
            if any(f.endswith(extension) for f in files):
                src_path = root
                break

        if src_path != base_path and os.path.exists(src_path):
            print(f"Moving files from '{src_path}' to '{base_path}' directory...")
            for item in os.listdir(src_path):
                s, d = os.path.join(src_path, item), os.path.join(base_path, item)
                shutil.move(s, d)
            try: os.rmdir(src_path)
            except OSError: pass

    flatten_directory(IMAGE_DIR, '.jpg')
    flatten_directory(ANNOTATION_DIR, '.txt')

    def convert_visdrone_to_coco(img_dir, ann_dir, output_json_path):
        """
        Parses custom .txt bounding box annotations and structures them into 
        a standard COCO JSON format required by MMDetection pipelines.
        """
        coco_output = {
            "info": {
                "description": "VisDrone Focus Stage Dataset",
                "url": "", "version": "1.0", "year": datetime.date.today().year,
                "contributor": "Focus-and-Detect Framework", "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            # The focus stage operates as a single-class detector: identifying the "focal_region"
            "categories": [{"id": 0, "name": "focal_region", "supercategory": "none"}]
        }
        image_id_counter, annotation_id_counter = 0, 0
        txt_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.txt')])
        if not txt_files: raise FileNotFoundError(f"No .txt files found in the specified directory: {ann_dir}")

        for txt_file in tqdm(txt_files, desc="Processing files"):
            image_name = txt_file.replace('.txt', '.jpg')
            image_path = os.path.join(img_dir, image_name)
            if not os.path.exists(image_path): continue

            # Extract image dimensions required for COCO
            try: image_cv = cv2.imread(image_path); height, width, _ = image_cv.shape
            except Exception: continue

            coco_output["images"].append({"id": image_id_counter, "file_name": image_name, "height": height, "width": width})

            # Parse bounding box coordinates
            with open(os.path.join(ann_dir, txt_file), 'r') as f:
                for line in f:
                    parts = [int(p) for p in line.strip().split(',')]
                    bbox_left, bbox_top, bbox_width, bbox_height = parts[0], parts[1], parts[2], parts[3]
                    coco_output["annotations"].append({
                        "id": annotation_id_counter, 
                        "image_id": image_id_counter, 
                        "category_id": 0, 
                        "bbox": [bbox_left, bbox_top, bbox_width, bbox_height], 
                        "area": float(bbox_width * bbox_height), 
                        "iscrowd": 0
                    })
                    annotation_id_counter += 1
            image_id_counter += 1

        # Save the monolithic COCO dataset
        with open(output_json_path, 'w') as f: json.dump(coco_output, f)
        print(f"\nConversion completed. Total {image_id_counter} images and {annotation_id_counter} annotations processed.")

    output_json_file = os.path.join(DATA_DIR, 'train.json')
    convert_visdrone_to_coco(IMAGE_DIR, ANNOTATION_DIR, output_json_file)

    # ------------------------- TRAIN/VAL SPLIT (Zero Overlap) -------------------------
    # Load the monolithic train.json, generate a deterministic 90% train / 10% val split,
    # overwrite train.json with the training subset, and write val.json separately.
    with open(output_json_file, 'r') as f:
        all_data = json.load(f)

    # Sort images by filename to ensure a deterministic split across runs
    images_sorted = sorted(all_data["images"], key=lambda x: x["file_name"])
    n_total = len(images_sorted)
    n_train = int(0.9 * n_total)
    
    # Use sets for O(1) lookup speeds when filtering annotations
    train_ids = set(img["id"] for img in images_sorted[:n_train])
    val_ids = set(img["id"] for img in images_sorted[n_train:])

    def split_coco(data, keep_ids):
        """Helper to extract a subset of images and annotations based on image IDs."""
        imgs = [img for img in data["images"] if img["id"] in keep_ids]
        anns = [ann for ann in data["annotations"] if ann["image_id"] in keep_ids]
        return {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "images": imgs,
            "annotations": anns,
            "categories": data.get("categories", [])
        }

    train_data = split_coco(all_data, train_ids)
    val_data = split_coco(all_data, val_ids)

    # Overwrite train.json and output val.json
    with open(os.path.join(DATA_DIR, 'train.json'), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(DATA_DIR, 'val.json'), 'w') as f:
        json.dump(val_data, f)

    print(f"\nSplit completed: train={len(train_data['images'])} images, val={len(val_data['images'])} images")

    print("\nSTEP 2 COMPLETED: Dataset is ready.")
    time.sleep(1)

    # --- STEP 3: CREATING CUSTOM MODEL AND CONFIGURATION FILES ---
    print("\n" + "="*80)
    print("STEP 3: CREATING CUSTOM MODEL AND CONFIGURATION FILES...")
    print("="*80)

    # Define paths for copying the custom model file
    source_model_path = "/content/drive/MyDrive/visdrone_gmm/resnet_halvit_for_fpn_compatible.py"
    target_model_path = "/content/mmdetection/mmdet/models/backbones/resnet_halvit_for_fpn_compatible.py"

    with open(source_model_path, 'r') as f: content = f.read()

    # Inject MMDetection's registry decorators so the architecture is recognized by the configs
    import_str = "from mmdet.registry import MODELS\n"
    if "from mmdet.registry import MODELS" not in content: content = import_str + content

    content = content.replace("def __init__(self, num_classes=1000):", "def __init__(self, num_classes=1000, **kwargs):")
    if "@MODELS.register_module()" not in content:
        content = content.replace("class ResNet50(nn.Module):", "@MODELS.register_module()\nclass ResNet50(nn.Module):")

    with open(target_model_path, 'w') as f: f.write(content)
    print(f"Custom model copied and updated in '{target_model_path}'.")

    # Generate the dataset configuration file tailored for the Focus stage
    os.makedirs('/content/mmdetection/configs/_base_/datasets/', exist_ok=True)
    with open("/content/mmdetection/configs/_base_/datasets/visdrone_gmm_coco.py", "w") as f:
        f.write("""
metainfo = {'classes': ('focal_region',), 'palette': [(220, 20, 60)]}
train_dataloader = dict(
    batch_size=8, num_workers=2, persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='CocoDataset', data_root='data/visdrone_gmm/', metainfo=metainfo,
                 ann_file='train.json', data_prefix=dict(img='images/'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True),
                    # Multi-scale training via RandomChoiceResize (400x1400 ~ 1200x1400)
                    dict(type='RandomChoiceResize',
                         scales=[(1400, 400), (1400, 600), (1400, 800), (1400, 1000), (1400, 1200)],
                         keep_ratio=True),
                    dict(type='RandomFlip', prob=0.5),
                    dict(type='PackDetInputs')
                 ]))
val_dataloader = dict(
    batch_size=4, num_workers=2, persistent_workers=True, drop_last=False, sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root='data/visdrone_gmm/', metainfo=metainfo,
                 # Utilize the separate, non-overlapping validation split
                 ann_file='val.json', data_prefix=dict(img='images/'),
                 test_mode=True, pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
                 ]))
test_dataloader = val_dataloader
# Val evaluator evaluates exclusively against val.json
val_evaluator = dict(type='CocoMetric', ann_file='data/visdrone_gmm/val.json', metric='bbox', classwise=True)
test_evaluator = val_evaluator
""")

    # Generate the model configuration linking GFL with the ResNet-HalVit backbone
    os.makedirs('/content/mmdetection/configs/gfl/', exist_ok=True)
    with open("/content/mmdetection/configs/gfl/gfl_halvit_fpn_1x_visdrone_gmm.py", "w") as f:
        f.write("""
_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../_base_/datasets/visdrone_gmm_coco.py'
]
custom_imports = dict(imports=['mmdet.models.backbones.resnet_halvit_for_fpn_compatible'], allow_failed_imports=False)

model = dict(
    type='GFL',
    data_preprocessor=dict(type='DetDataPreprocessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True, pad_size_divisor=32),
    backbone=dict(
        type='ResNet50',
        init_cfg=None,
        # Backbone utilizes Synchronized Batch Normalization (SyncBN)
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False),
    neck=dict(
        type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, start_level=1,
        add_extra_convs='on_output', num_outs=5,
        # Feature Pyramid Network utilizes Group Normalization (GN)
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    bbox_head=dict(
        type='GFLHead', num_classes=1, in_channels=256, stacked_convs=4, feat_channels=256,
        anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], octave_base_scale=8, scales_per_octave=1, strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        # Non-Maximum Suppression (NMS) setup
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

load_from = None
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# Extended training to 24 epochs and adjusted Step LR schedule mapping [16, 22]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
param_scheduler = [dict(type='MultiStepLR', milestones=[16, 22], gamma=0.1)]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
""")
    print("\nSTEP 3 COMPLETED: Configuration files are ready.")
    time.sleep(1)

    # --- STEP 4: INITIATING TRAINING ---
    print("\n" + "="*80)
    print("STEP 4: INITIATING TRAINING...")
    print("="*80)

    config_file = '/content/mmdetection/configs/gfl/gfl_halvit_fpn_1x_visdrone_gmm.py'
    work_dir = '/content/drive/MyDrive/visdrone_gmm/gfl_focus_model_checkpoints' 

    run_command(f"python /content/mmdetection/tools/train.py {config_file} --work-dir {work_dir}")

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print(f"Model weights and log files are saved in the following Google Drive folder: {work_dir}")
    print("="*80)

except Exception as e:
    # Safely catch and report pipeline failures
    print("\n" + "*"*80, file=sys.stderr)
    print("AN ERROR OCCURRED!", file=sys.stderr)
    print(f"Error Detail: {e}", file=sys.stderr)
    print("*"*80, file=sys.stderr)
