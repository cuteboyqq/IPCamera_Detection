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
        
        # -- Label-confidence threshold ---
        self.coco2017_conf_th = args.coco2017_conf_th
        self.face_conf_th     = args.face_conf_th
        self.pose_conf_th     = args.pose_conf_th
        
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
        self.mapping_label_name = args.mapping_label_name
        
        # --- Save option ---
        self.save_result_im = args.save_result_im
        self.save_result_im_resolution =args.save_result_im_resolution
        self.save_mapping_result_im = args.save_mapping_result_im
        
        
    def _bool_str(self, value):
        """Return colored check or cross for booleans."""
        return "✅ True" if value else "❌ False"

    def show_config(self,args):
        """
        Print all parameter settings with emoji and colors for better readability.
        """
        GREEN = "\033[92m"
        RED = "\033[91m"
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        END = "\033[0m"

        print(f"\n{BOLD}{CYAN}📌 ====== Dataset Configuration ======{END}\n")

        # --- Task settings ---
        print(f"{YELLOW}📝 Task Settings:{END}")
        print(f"   🐼 COCO Detection : {self._bool_str(self.task_coco_detection)}")
        print(f"   🙂 Face Detection : {self._bool_str(self.task_face_detection)}")
        print(f"   🕺 Pose Detection : {self._bool_str(self.task_pose_detection)}")
        print(f"   🔀 Multi Detection: {self._bool_str(self.task_multi_detection)}\n")

        # --- Multi-detection flags ---
        print(f"{YELLOW}🛠️ Multi-Detection Enable Flags:{END}")
        print(f"   🐼 COCO2017 : {self._bool_str(self.md_enable_coco2017)}")
        print(f"   🙂 FACE     : {self._bool_str(self.md_enable_face)}")
        print(f"   🕺 POSE     : {self._bool_str(self.md_enable_pose)}\n")

        # --- Thresholds ---
        print(f"{YELLOW}🎯 Confidence Thresholds:{END}")
        print(f"   🐼 COCO2017 : {GREEN}{self.coco2017_conf_th}{END}")
        print(f"   🙂 FACE     : {GREEN}{self.face_conf_th}{END}")
        print(f"   🕺 POSE     : {GREEN}{self.pose_conf_th}{END}\n")

        # --- Label values ---
        print(f"{YELLOW}🏷️ Label Values:{END}")
        print(f"   🙂 Face Label : {GREEN}{self.face_label_value}{END}")
        print(f"   🕺 Pose Label : {GREEN}{self.pose_label_value}{END}\n")

        # --- Dataset info ---
        print(f"{YELLOW}📂 Dataset Info:{END}")
        print(f"   🔢 Num Images   : {GREEN}{self.data_num}{END}")
        print(f"   📁 Data Dir     : {self.data_dir}")
        print(f"   📑 Data Type    : {self.data_type}")
        print(f"   🖼️ Image Dir    : {self.data_img_dir}")
        print(f"   📝 Label Save   : {self.data_save_txt_dir}")
        print(f"   📝 Pose Save    : {self.data_pose_save_txt_dir}\n")

        # --- Models ---
        print(f"{YELLOW}🤖 Models:{END}")
        print(f"   📦 Detection : {args.detect_model if args.detect_model else RED+'Not Loaded'+END}")
        print(f"   🙂 Face      : {getattr(args, 'face_model', RED+'Not Loaded'+END)}")
        print(f"   🕺 Pose      : {args.pose_model if args.pose_model else RED+'Not Loaded'+END}\n")

        # --- Results ---
        print(f"{YELLOW}🖼️ Result Options:{END}")
        print(f"   👀 Show Images        : {self._bool_str(self.show_result_im)}")
        print(f"   💾 Save Images        : {self._bool_str(self.save_result_im)}")
        print(f"   📏 Save Resolution    : {GREEN}{self.save_result_im_resolution}{END}")
        print(f"   💾 Save Mapping Result: {self._bool_str(self.save_mapping_result_im)}\n")

        # --- Mapping ---
        print(f"{YELLOW}🗺️ Mapping Settings:{END}")
        print(f"   🔄 Enable Mapping      : {self._bool_str(self.enable_mapping)}")
        print(f"   📋 Label Mapping       : {self.label_mapping}")
        print(f"   📥 Mapping Input Dir   : {self.mapping_input_label_dir}")
        print(f"   📤 Mapping Output Dir  : {self.mapping_output_label_dir}")
        print(f"   🏷️ Mapping Label Name  : {self.mapping_label_name}\n")

        print(f"{BOLD}{CYAN}✅ ====== End of Config ======{END}\n")
    
    
    
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