import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2

class BaseDataset:
    """
    BaseDataset class initializes dataset parameters from Args config.
    """

    def __init__(self, args):
        """
        Initialize dataset parameters from Args object.

        Parameters
        ----------
        args : Args
            Parsed configuration object containing dataset and task settings.
        """
        # --- Task settings ---
        self.task_coco_detection  = args.task_coco_detection
        self.task_face_detection  = args.task_face_detection
        self.task_pose_detection  = args.task_pose_detection
        self.task_multi_detection = args.task_multi_detection

        # --- Multi-detection enable flags ---
        self.md_enable_coco2017 = args.md_enable_coco2017
        self.md_enable_face     = args.md_enable_face
        self.md_enable_pose     = args.md_enable_pose
        
        # --- Label value ---
        self.face_label_value = args.face_label_value
        self.pose_label_value = args.pose_label_value

        # --- Dataset info ---
        self.data_num               = args.data_num
        self.data_dir               = args.data_dir
        self.data_type              = args.data_type
        self.data_img_dir           = args.data_img_dir
        self.data_save_txt_dir      = args.data_save_txt_dir
        self.data_pose_save_txt_dir = args.data_pose_save_txt_dir
        
        # --- Model info ---
        self.detect_model = YOLO(args.detect_model)
        if Path(args.face_model).is_file():
            self.face_model   = YOLO(args.face_model)
        else:
            print(f"face model {args.face_model} does not exist...")
        self.pose_model   = YOLO(args.pose_model)
        
        self.show_result_im = args.show_result_im
        
      
        
        # --- Mapping ---
        self.enable_mapping = args.enable_mapping
        self.label_mapping = args.label_mapping
        self.mapping_input_label_dir = args.mapping_input_label_dir
        self.mapping_output_label_dir = args.mapping_output_label_dir
        
        self.save_result_im = args.save_result_im
        self.save_result_im_resolution =args.save_result_im_resolution
        
    def Save_YOLO_txt_Labels(self, img_path, image):
        """
        Save YOLO-format labels for a given image.

        This function should be implemented by dataset-specific subclasses 
        (e.g., COCO detection, face detection, pose detection). It runs the 
        corresponding detection/pose model on the input image, generates 
        normalized bounding box (and keypoint, if applicable) labels, and 
        saves them in YOLO text file format.

        Args:
            img_path (str | Path):
                Path to the input image file.
            image (numpy.ndarray | None):
                Optional pre-loaded image (BGR format). If None, the image 
                will be loaded from `img_path`.

        Returns:
            numpy.ndarray:
                The image with visualized results (bounding boxes, class 
                labels, and/or skeletons) if visualization is enabled. 
                Otherwise, returns the original image.
        """
        return NotImplemented

    def filter_and_remap_yolo_labels(self):
        """
        Filters and remaps YOLO labels using pathlib.

        Args:
            label_dir (Path or str): Path to input label directory.
            output_dir (Path or str): Path to save filtered & remapped labels.
            mapping (dict): Maps original class IDs to new class IDs.
        """
        label_dir = Path(self.mapping_input_label_dir)
        output_dir = Path(self.mapping_output_label_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        #wanted_cls = set(self.label_mapping).keys()
        wanted_cls = set(self.label_mapping.keys())
        for label_file in label_dir.glob("*.txt"):
            with label_file.open("r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id in wanted_cls:
                    new_class_id = self.label_mapping[class_id]
                    rest = parts[1:]  # bbox: x_center, y_center, width, height
                    new_line = f"{new_class_id} {' '.join(rest)}\n"
                    new_lines.append(new_line)

            output_file = output_dir / label_file.name
            with output_file.open("w") as f:
                f.writelines(new_lines)

        # print(f"Filtered and remapped labels saved to: {output_dir}")