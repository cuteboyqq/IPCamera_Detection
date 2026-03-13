# 🏷️ Multi-Task Labeling Tool User Guide

## 🌟 Overview

This labeling tool is designed to automatically generate detection labels for computer vision datasets using pre-trained YOLO models. It supports three main detection tasks:

- **🎯 COCO Object Detection**: Detects 80 common object classes from the COCO2017 dataset
- **👤 Face Detection**: Specialized face detection capabilities  
- **🤸 Pose Detection**: Human keypoint/pose estimation
- **🚀 Multi-Task Mode**: Combines all three detection types in a single run

## 🚀 Set up environment 

Setting up a virtual environment will keep your YOLO + OpenCV project clean and isolated from system packages. Since you're on Ubuntu/Linux, here's how you can do it step by step:

🔹 **1. Install venv (if not already installed)** 📦
```python
sudo apt update
sudo apt install python3-venv -y
```

🔹 **2. Create a virtual environment** 🏗️
Navigate to your project folder, then run:
```python
python3 -m venv env
```
This will create a folder named `env/` in your project. 📁

🔹 **3. Activate the virtual environment** ⚡
```python
source env/bin/activate
```
When activated, your shell prompt will show `(env)` at the beginning. ✅

**To deactivate later:** 🛑
```python
deactivate
```

🔹 **4. Install dependencies** 📥
```python
# Upgrade pip and install packages
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

🔹 **5. Verify environment** 🔍
```python
which python
which pip
```
Both should point to your project's `env/` directory, not system Python. ✔️

⚡ **Pro tip:** If you use VS Code or PyCharm, you can set this env as your interpreter so scripts automatically run inside it. 💡🎯


## Quick Start

1. **📂 Must Do: Copy Required Files into Ultralytics Repository**
   Before running the auto-labeling tools, you must copy the following folders, scripts, and model into the Ultralytics repository https://github.com/ultralytics/ultralytics
, because the auto-labeling process depends on Ultralytics models.
  **Files and folders to copy:**
   1. tasks/ folder
   2. config/ folder
   3. engine/ folder
   4. main_label.py script
   5. yolo11l-face.pt model (download from https://github.com/YapaLab/yolo-face)
   
  **Example directory structure after copying:**
```yaml
ultralytics/
├── tasks/
├── config/
├── engine/
├── main_label.py
├── yolo11l-face.pt
...
# (Other original Ultralytics files and folders remain unchanged)
```
⚠️ Warning: Do not overwrite existing Ultralytics files unless specifically instructed. Only add the required folders, scripts, and model listed above.

2. **Activate virtual environment**: When activated, your shell prompt will show `(env)` at the beginning. ✅
  
3. **Configure the tool**: Edit `config.yaml` to match your dataset paths and requirements
   
4. **Prepare your dataset**: Ensure images are in the specified directory
   
5. **Run the labeling process**: Execute the main script

```python
python main_label.py
# or python3.8 main_label.py # Based on which python version you are using
```
5. **Review results**: Check generated labels and visualization images

## 📋 Configuration Guide

### 🎯 Task Selection

Choose which detection tasks to enable:

```yaml
GENERATE_LABEL_TASK:
  TASK_COCO_DETECTION: False   # COCO objects only
  TASK_FACE_DETECTION: False   # Face detection only  
  TASK_POSE_DETECTION: False   # Pose detection only
  TASK_MULTI_DETECTION: True   # All tasks combined (recommended)
```

**💡 Recommendation**: Use `TASK_MULTI_DETECTION: True` for comprehensive labeling.

### 🎛️ Multi-Task Configuration

When using multi-task mode, specify which detection types to include:

```yaml
MD_ENABLE_LABEL:
  COCO2017: True   # Include COCO object detection
  FACE: True       # Include face detection
  POSE: True       # Include pose detection
```

### 🎚️ Detection Confidence

Set confidence thresholds for each detection type (0.0-1.0):

```yaml
CONFIDENCE_THRESHOLD:
  COCO2017: 0.25   # Lower = more detections, higher = more precise
  FACE: 0.25
  POSE: 0.25
```

**📊 Guidelines**:
- **🟢 0.1-0.3**: More detections, some false positives
- **🟡 0.4-0.6**: Balanced precision/recall
- **🔴 0.7-0.9**: High precision, may miss some objects

### 🧠 Model Configuration

Specify the pre-trained models to use:

```yaml
MODEL:
  DETECT_MODEL: yolo11x.pt       # COCO detection model
  FACE_MODEL: yolov11l-face.pt   # Face detection model  
  POSE_MODEL: yolo11x-pose.pt    # Pose estimation model
```

**🏃 Model Options**:
- **⚡ Nano (n)**: Fastest, least accurate
- **🏃 Small (s)**: Good speed/accuracy balance
- **🚶 Medium (m)**: Better accuracy
- **🐌 Large (l)**: High accuracy
- **🐢 Extra Large (x)**: Best accuracy, slowest

### 📁 Dataset Paths

Configure your dataset directory structure:

```yaml
DATA:
  DATA_NUM: 100000                    # Max images to process
  DATA_DIR: "/path/to/dataset"        # Root dataset directory
  DATA_TYPE: "train"                  # Split type: 'train', 'val', or ''
  DATA_IMG_DIR: "/path/to/images"     # Source images folder
  DATA_SAVE_DETECT_TXT_DIR: "/path/to/detection_labels"  # Detection output
  DATA_SAVE_POSE_TXT_DIR: "/path/to/pose_labels"         # Pose output
```

### 🏷️ Label Configuration

Set category indices for different detection types:

```yaml
LABEL_VALUE:
  FACE: 10   # Face class ID (usually after COCO classes)
  POSE: 0    # Pose class ID (0 if separate from other classes)
```

### 👀 Visualization Settings

Control result visualization:

```yaml
VISUALIZE:
  SHOW_RESULT_IM: False   # Display images during processing (slower)

SAVE:
  RESULT_IMAGE: True                 # Save annotated images
  RESULT_IM_RESOLUTION: [720,1280]   # Output image size [height, width]
  RESULT_MAPPING_IMAGE: True         # Save class mapping visualizations
```

### 🔄 Class Mapping

Remap COCO classes to a custom set:

```yaml
MAPPING:
  ENABLE: True
  LABEL_MAPPING: {0: 0, 2: 1, 1: 2, 3: 3, 7: 4, 5: 5, 56: 6, 16: 7, 24: 8, 15: 9, 80: 10}
  MAPPING_LABEL_NAME: ["person", "car", "bicycle", "motorcycle", "truck", "bus", "chair", "dog", "backpack", "cat", "face"]
```

**Current Mapping**:
- Original COCO class 0 (person) → ID 0
- Original COCO class 2 (car) → ID 1
- Original COCO class 1 (bicycle) → ID 2
- And so on...
- Face detection → ID 10

## Output Formats

### Detection Labels (YOLO Format)

Each text file contains one line per detection:
```
class_id center_x center_y width height confidence
```

Example:
```
0 0.5 0.3 0.2 0.4 0.85
1 0.7 0.6 0.15 0.25 0.92
```

### 🤸 Pose Labels

Keypoint format varies by model but typically includes:
```
class_id center_x center_y width height keypoint_1_x keypoint_1_y keypoint_1_visible ...
```

## 📁 Directory Structure

Organize your dataset as follows:

```
dataset/
├── images/
│   └── train2017/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   ├── detection/
│   │   └── train2017-10cls/
│   │       ├── image1.txt
│   │       ├── image2.txt
│   │       └── ...
│   └── lane/
│       └── points/
│           └── train2017-2025-09-19/
│               ├── image1.txt
│               ├── image2.txt
│               └── ...
└── results/ (auto-generated)
    ├── annotated_images/
    └── mapping_images/
```

## 🎮 Usage Examples

### 🚀 Basic Multi-Task Labeling

1. Set up multi-task detection:
```yaml
GENERATE_LABEL_TASK:
  TASK_MULTI_DETECTION: True

MD_ENABLE_LABEL:
  COCO2017: True
  FACE: True
  POSE: True
```

2. Configure paths:
```yaml
DATA_IMG_DIR: "/your/images/folder"
DATA_SAVE_DETECT_TXT_DIR: "/your/output/labels"
```

3. Run the tool

### 🎯 COCO Detection Only

```yaml
GENERATE_LABEL_TASK:
  TASK_COCO_DETECTION: True
  TASK_MULTI_DETECTION: False
```

### 🎨 Custom Class Mapping

To create a custom 5-class dataset from COCO:

```yaml
MAPPING:
  ENABLE: True
  LABEL_MAPPING: {0: 0, 2: 1, 5: 2, 7: 3, 15: 4}
  MAPPING_LABEL_NAME: ["person", "car", "bus", "truck", "cat"]
```

## 🔧 Troubleshooting

### ⚠️ Common Issues

**❌ No labels generated**:
- Check confidence thresholds (try lowering to 0.1)
- Verify image paths are correct
- Ensure models are downloaded/accessible

**❗ Too many false positives**:
- Increase confidence thresholds (0.4-0.6)
- Use smaller, more precise models

**😔 Missing detections**:
- Lower confidence thresholds
- Use larger models (yolo11x vs yolo11n)

**📂 Path errors**:
- Use absolute paths in configuration
- Ensure directories exist or can be created
- Check file permissions

### 🚀 Performance Optimization

**⚡ Speed up processing**:
- Use smaller models (yolo11n, yolo11s)
- Set `SHOW_RESULT_IM: False`
- Reduce `DATA_NUM` for testing
- Use GPU if available

**🎯 Improve accuracy**:
- Use larger models (yolo11l, yolo11x)
- Fine-tune confidence thresholds
- Enable all detection types in multi-task mode

## ✨ Best Practices

1. **🐣 Start small**: Test with a few images first (`DATA_NUM: 10`)
2. **✅ Validate thresholds**: Review generated labels manually
3. **💾 Backup original data**: Keep copies before processing
4. **💽 Monitor disk space**: Large datasets generate many label files
5. **📝 Use version control**: Track configuration changes
6. **📋 Document mapping**: Keep records of class remapping decisions

## 🆘 Support

For issues or questions:
- Check configuration syntax against this guide
- Verify model files are valid and accessible
- Ensure sufficient disk space and permissions
- Review log output for specific error messages