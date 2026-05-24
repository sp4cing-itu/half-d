# -*- coding: utf-8 -*-
"""
================================================================================
Focus-and-Detect Framework: Detection Stage Automated Training Pipeline
================================================================================

Description:
This script completely automates the training pipeline for the detection stage 
of the "Focus-and-Detect" framework. It trains a custom ResNeXt-101 HalVit model 
on pre-cropped focal region images.

Pipeline Overview:
------------------
* STEP 1: Environment Setup
  Installs a dedicated Python 3.11 virtual environment, resolves dependency conflicts 
  (pinning NumPy, PyTorch, MMCV), and installs the OpenMMLab MMDetection framework.

* STEP 2: Dataset Preparation
  Extracts zipped images and custom TXT annotations from Google Drive, flattens the 
  directory structures, converts the VisDrone-style annotations into a monolithic 
  COCO JSON format, and securely splits it into training (90%) and validation (10%) sets.

* STEP 3: Model Architecture & Configuration
  Injects the custom ResNeXt-101 HalVit backbone into the MMDetection registry. 
  Generates the necessary Python-based configuration files for the dataset, model 
  (Generalized Focal Loss - GFL), and training schedules.

* STEP 4: Training Execution
  Initiates the MMDetection training sequence using the generated configs and 
  saves the output checkpoints and logs back to Google Drive.
================================================================================
"""

import os
import subprocess
import sys
import time
import shutil
import json
import cv2
from tqdm import tqdm
import datetime
import random 

def run_command(command, **kwargs):
    """
    Executes a shell command and streams the standard output in real-time.
    Raises a CalledProcessError if the command fails.
    """
    print(f"\n> Executing: {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True, bufsize=1, **kwargs)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

def run_venv_command(venv_path, command, **kwargs):
    """
    Executes a shell command specifically within the activated virtual environment.
    """
    full_command = f"bash -c 'source {venv_path}/bin/activate && {command}'"
    return run_command(full_command, **kwargs)

def split_coco_json(json_path, train_ratio=0.9):
    """
    Splits a single monolithic COCO format JSON file into separate training and 
    validation JSON files based on the specified ratio.
    """
    print(f"\nSplitting '{json_path}' into {train_ratio*100}% train / {(1-train_ratio)*100}% val...")

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']

    # Shuffle the images to ensure a random and unbiased distribution
    random.shuffle(images)

    # Calculate the split index and divide the image list
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"Out of {len(images)} total images, {len(train_images)} allocated for training and {len(val_images)} for validation.")

    # Store IDs in sets for O(1) lookup performance when filtering annotations
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}

    # Distribute annotations to respective sets based on their associated image_id
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]

    # Extract common COCO structure dictionaries (metadata, licenses, classes)
    common_data = {
        "info": coco_data.get('info', {}),
        "licenses": coco_data.get('licenses', []),
        "categories": coco_data['categories']
    }

    # Construct the final dictionaries for the new JSON files
    train_json = {**common_data, 'images': train_images, 'annotations': train_annotations}
    val_json = {**common_data, 'images': val_images, 'annotations': val_annotations}

    # Determine output paths based on the original file's directory
    base_dir = os.path.dirname(json_path)
    train_output_path = os.path.join(base_dir, 'focal_regions_train.json')
    val_output_path = os.path.join(base_dir, 'focal_regions_val.json')

    # Serialize and save the split datasets to disk
    with open(train_output_path, 'w') as f:
        json.dump(train_json, f)
    print(f"Training JSON file saved: {train_output_path}")

    with open(val_output_path, 'w') as f:
        json.dump(val_json, f)
    print(f"Validation JSON file saved: {val_output_path}")

try:
    # --- STEP 1: PYTHON 3.11 VIRTUAL ENVIRONMENT SETUP ---
    print("="*80)
    print("DETECTION STAGE TRAINING - STEP 1: PYTHON 3.11 VIRTUAL ENVIRONMENT SETUP...")
    print("="*80)

    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    print("Installing Python 3.11...")
    run_command("sudo apt update")
    run_command("sudo apt install -y software-properties-common")
    run_command("sudo add-apt-repository -y ppa:deadsnakes/ppa")
    run_command("sudo apt update")
    run_command("sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils")

    print("Creating Python 3.11 virtual environment...")
    venv_path = "/content/py311_detection_env"
    run_command(f"python3.11 -m venv {venv_path}")

    print("Upgrading pip and installing basic tools...")
    run_venv_command(venv_path, "python -m pip install --upgrade pip")
    run_venv_command(venv_path, "pip install wheel setuptools")

    # Critical Step: Pin NumPy to version 1.26 to prevent compatibility issues
    print("Installing NumPy 1.26 (to prevent NumPy 2.x incompatibilities)...")
    run_venv_command(venv_path, "pip install 'numpy==1.26'")

    print("Installing PyTorch and compatible packages...")
    run_venv_command(venv_path, "pip install 'matplotlib==3.7.2'")
    run_venv_command(venv_path, "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121")

    print("Installing MMCV...")
    run_venv_command(venv_path, "pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html")

    run_venv_command(venv_path, "pip install mmengine")
    run_venv_command(venv_path, "pip install opencv-python")
    run_venv_command(venv_path, "pip install Pillow")
    run_venv_command(venv_path, "pip install scipy")
    run_venv_command(venv_path, "pip install scikit-image")

    print("Cloning and installing MMDetection...")
    if not os.path.exists('/content/mmdetection'):
        run_command("git clone https://github.com/open-mmlab/mmdetection.git /content/mmdetection")

    os.chdir('/content/mmdetection')

    # Clean up MMDetection's requirements.txt to prevent implicit version upgrades
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            original_lines = f.readlines()

        with open('requirements.txt', 'w') as f:
            for line in original_lines:
                line_lower = line.lower()
                if not any(pkg in line_lower for pkg in ['numpy', 'torch', 'matplotlib', 'opencv', 'pillow', 'scipy']):
                    f.write(line)

    print("Pinning NumPy version to 1.26...")
    run_venv_command(venv_path, "pip install --force-reinstall 'numpy==1.26'")

    run_venv_command(venv_path, "pip install -e .")

    print("\nSTEP 1 COMPLETED: Python 3.11 virtual environment and MMDetection installed.")
    time.sleep(1)

    # --- STEP 2: DETECTION STAGE DATASET PREPARATION ---
    print("\n" + "="*80)
    print("STEP 2: DETECTION STAGE DATASET PREPARATION STARTING...")
    print("="*80)

    # Define target directories for the focal region dataset
    DETECTION_DATA_DIR = '/content/mmdetection/data/visdrone_detection_focal_regions/'
    DETECTION_IMAGE_DIR = os.path.join(DETECTION_DATA_DIR, 'images')
    DETECTION_ANNOTATION_DIR = os.path.join(DETECTION_DATA_DIR, 'annotations')
    os.makedirs(DETECTION_IMAGE_DIR, exist_ok=True)
    os.makedirs(DETECTION_ANNOTATION_DIR, exist_ok=True)

    print("Extracting focal region images...")
    if os.path.exists('/content/drive/MyDrive/visdrone_gmm/detection_images.zip'):
        run_command("unzip -q -o /content/drive/MyDrive/visdrone_gmm/detection_images.zip -d " + DETECTION_IMAGE_DIR)
    elif os.path.exists('/content/drive/MyDrive/visdrone_gmm/images.zip'):
        print("detection_images.zip not found, using images.zip instead...")
        run_command("unzip -q -o /content/drive/MyDrive/visdrone_gmm/images.zip -d " + DETECTION_IMAGE_DIR)

    print("Extracting focal region annotations...")
    if os.path.exists('/content/drive/MyDrive/visdrone_gmm/detection_annotations.zip'):
        run_command("unzip -q -o /content/drive/MyDrive/visdrone_gmm/detection_annotations.zip -d " + DETECTION_ANNOTATION_DIR)
    elif os.path.exists('/content/drive/MyDrive/visdrone_gmm/annotations.zip'):
        print("detection_annotations.zip not found, using annotations.zip instead...")
        run_command("unzip -q -o /content/drive/MyDrive/visdrone_gmm/annotations.zip -d " + DETECTION_ANNOTATION_DIR)

    def flatten_directory(base_path, extension):
        """
        Helper function to recursively move files from nested subdirectories
        to the base directory.
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
                if os.path.exists(d):
                    os.remove(d)
                shutil.move(s, d)
            try:
                shutil.rmtree(src_path)
            except OSError:
                pass

    flatten_directory(DETECTION_IMAGE_DIR, '.jpg')
    flatten_directory(DETECTION_ANNOTATION_DIR, '.txt')

    def convert_focal_regions_to_coco(img_dir, ann_dir, output_json_path):
        """
        Converts the custom VisDrone annotation text format into a COCO JSON format.
        """
        categories = [
            {"id": 1, "name": "pedestrian", "supercategory": "person"},
            {"id": 2, "name": "person", "supercategory": "person"},
            {"id": 3, "name": "bicycle", "supercategory": "vehicle"},
            {"id": 4, "name": "car", "supercategory": "vehicle"},
            {"id": 5, "name": "van", "supercategory": "vehicle"},
            {"id": 6, "name": "truck", "supercategory": "vehicle"},
            {"id": 7, "name": "tricycle", "supercategory": "vehicle"},
            {"id": 8, "name": "awning-tricycle", "supercategory": "vehicle"},
            {"id": 9, "name": "bus", "supercategory": "vehicle"},
            {"id": 10, "name": "motor", "supercategory": "vehicle"}
        ]

        coco_output = {
            "info": {
                "description": "VisDrone Detection Stage - Focal Regions Dataset",
                "url": "", "version": "1.0", "year": datetime.date.today().year,
                "contributor": "Focus-and-Detect Framework",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }

        image_id_counter, annotation_id_counter = 0, 0

        image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not image_files:
            raise FileNotFoundError(f"No image files found in the specified directory: {img_dir}")

        for image_file in tqdm(image_files, desc="Processing focal region files"):
            image_path = os.path.join(img_dir, image_file)
            txt_file = os.path.splitext(image_file)[0] + '.txt'
            annotation_path = os.path.join(ann_dir, txt_file)

            try:
                image_cv = cv2.imread(image_path)
                if image_cv is None: continue
                height, width, _ = image_cv.shape
            except Exception:
                continue

            coco_output["images"].append({
                "id": image_id_counter,
                "file_name": image_file,
                "height": height,
                "width": width
            })

            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            parts = line.split(',')
                            if len(parts) < 6: continue
                            bbox_left, bbox_top, bbox_width, bbox_height = map(int, parts[:4])
                            object_category = int(parts[5])
                            
                            if object_category in [0, 11]: continue
                            
                            area = float(bbox_width * bbox_height)
                            if area > 0:
                                coco_output["annotations"].append({
                                    "id": annotation_id_counter,
                                    "image_id": image_id_counter,
                                    "category_id": object_category,
                                    "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                                    "area": area,
                                    "iscrowd": 0
                                })
                                annotation_id_counter += 1
                        except (ValueError, IndexError):
                            continue
            image_id_counter += 1

        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f)

        print(f"\nDetection stage conversion completed.")
        print(f"Total {image_id_counter} images and {annotation_id_counter} annotations processed.")

    # Generate a temporary monolithic JSON file containing the full dataset
    full_dataset_json = os.path.join(DETECTION_DATA_DIR, 'focal_regions_full.json')
    convert_focal_regions_to_coco(DETECTION_IMAGE_DIR, DETECTION_ANNOTATION_DIR, full_dataset_json)

    # Apply the splitting function (90% train, 10% validation)
    split_coco_json(full_dataset_json, train_ratio=0.9) 

    # Clean up by removing the temporary monolithic JSON file
    os.remove(full_dataset_json)
    print(f"Temporary file deleted: {full_dataset_json}")

    print("\nSTEP 2 COMPLETED: Detection stage dataset prepared and split.")
    time.sleep(1)

    # --- STEP 3: CREATING RESNEXT-101 HALVIT MODEL AND CONFIGURATION FILES ---
    print("\n" + "="*80)
    print("STEP 3: CREATING ResNeXt-101 HalVit MODEL AND CONFIGURATION FILES...")
    print("="*80)

    source_resnext_halvit_path = "/content/drive/MyDrive/visdrone_gmm/resnext_halvit_test.py"
    target_resnext_halvit_path = "/content/mmdetection/mmdet/models/backbones/resnext_halvit_test.py"

    if not os.path.exists(source_resnext_halvit_path):
        raise FileNotFoundError(f"ResNeXt HalVit model file not found: {source_resnext_halvit_path}")

    with open(source_resnext_halvit_path, 'r', encoding='utf-8') as f:
        resnext_content = f.read()

    import_str = "from mmdet.registry import MODELS\n"
    if "from mmdet.registry import MODELS" not in resnext_content:
        resnext_content = import_str + resnext_content

    if "@MODELS.register_module()" not in resnext_content:
        resnext_content = resnext_content.replace(
          "class ResNeXt101Halvit(nn.Module):",
          "@MODELS.register_module()\nclass ResNeXt101Halvit(nn.Module):"
        )

    resnext_content = resnext_content.replace(
        "def __init__(self, num_classes=1000, groups=32, width_per_group=8):",
        "def __init__(self, num_classes=1000, groups=32, width_per_group=8, **kwargs):"
    )

    with open(target_resnext_halvit_path, 'w', encoding='utf-8') as f:
        f.write(resnext_content)

    print(f"ResNeXt-101 HalVit model copied and updated in '{target_resnext_halvit_path}'.")

    os.makedirs('/content/mmdetection/configs/_base_/datasets/', exist_ok=True)
    detection_dataset_config = """
# VisDrone Detection Stage Dataset Configuration - Focal Regions
metainfo = {
    'classes': ('pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
}

# TRAINING DATALOADER CONFIGURATION
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root='data/visdrone_detection_focal_regions/',
        metainfo=metainfo,
        ann_file='focal_regions_train.json', # Directing dataloader to the training split
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

# VALIDATION DATALOADER CONFIGURATION
val_dataloader = dict(
    batch_size=1, # Validation should be run with a batch size of 1
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/visdrone_detection_focal_regions/',
        metainfo=metainfo,
        ann_file='focal_regions_val.json', # Directing dataloader to the validation split
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            # Keep image dimensions constant during testing/validation as specified by the paper
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
            )
        ]
    )
)
test_dataloader = val_dataloader

# VALIDATION EVALUATOR CONFIGURATION
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/visdrone_detection_focal_regions/focal_regions_val.json', # Evaluate against validation split
    metric='bbox',
    classwise=True
)
test_evaluator = val_evaluator
"""

    with open("/content/mmdetection/configs/_base_/datasets/visdrone_detection_focal_regions.py", "w") as f:
        f.write(detection_dataset_config)

    os.makedirs('/content/mmdetection/configs/gfl/', exist_ok=True)
    detection_gfl_config = """
# Focus-and-Detect Detection Stage Configuration
_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/visdrone_detection_focal_regions.py'
]

custom_imports = dict(
    imports=['mmdet.models.backbones.resnext_halvit_test'],
    allow_failed_imports=False
)

model = dict(
    type='GFL',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNeXt101Halvit',
        groups=32,
        width_per_group=8,
        init_cfg=None,
        use_halvit_from_stage=5,
        # Utilize Synchronized Batch Normalization for the backbone
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        # Utilize Group Normalization within the FPN layers
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    ),
    bbox_head=dict(
        type='GFLHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]
        ),
        loss_cls=dict(
            type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)
    ),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000, min_bbox_size=0, score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6), max_per_img=100)
)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP'),
    logger=dict(type='LoggerHook', interval=50)
)

load_from = None
resume = False

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
"""

    with open("/content/mmdetection/configs/gfl/gfl_resnext101_halvit_detection_focal_regions.py", "w") as f:
        f.write(detection_gfl_config)

    print("\nSTEP 3 COMPLETED: ResNeXt-101 HalVit configuration files are ready.")
    time.sleep(1)

except Exception as e:
    print("\n" + "*"*80, file=sys.stderr)
    print("AN ERROR OCCURRED!", file=sys.stderr)
    print(f"Error Detail: {e}", file=sys.stderr)
    print("*"*80, file=sys.stderr)
    import traceback
    traceback.print_exc()

# --- STEP 4: INITIATING DETECTION STAGE TRAINING ---
print("\n" + "="*80)
print("STEP 4: INITIATING DETECTION STAGE TRAINING...")
print("Training on focal regions with ResNeXt-101 HalVit model...")
print("="*80)

run_venv_command(venv_path, "pip install --force-reinstall 'numpy==1.26'")

config_file = '/content/mmdetection/configs/gfl/gfl_resnext101_halvit_detection_focal_regions.py'
work_dir = '/content/drive/MyDrive/visdrone_gmm/detection_resnext_checkpoints'

os.environ['MPLBACKEND'] = 'Agg'

train_command = f"export MPLBACKEND=Agg && python /content/mmdetection/tools/train.py {config_file} --work-dir {work_dir} --resume"
run_venv_command(venv_path, train_command)

print("\n" + "="*80)
print("DETECTION STAGE TRAINING COMPLETED!")
print(f"Model weights and log files saved to the following Google Drive folder: {work_dir}")
print("This model is ready for the detection stage of the Focus-and-Detect framework.")
print("It can perform object detection on focal regions.")
print("="*80)
