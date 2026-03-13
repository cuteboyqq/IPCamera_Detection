import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Define COCO-style skeleton (17 keypoints) ---
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

# assign distinct colors per limb
COLORS = [tuple(int(c*255) for c in plt.cm.tab20(i)[:3]) for i in range(len(SKELETON))]

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
        self.data_type              = args.data_type
        self.data_video_dir         = args.data_video_dir
        self.data_skip_frame        = args.data_skip_frame
        self.data_num               = args.data_num
        self.data_dir               = args.data_dir
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
        
        self.detect_resolution  = args.detect_resolution
        self.face_resolution    = args.face_resolution
        self.pose_resolution    = args.pose_resolution
        
        self.show_result_im = args.show_result_im
        self.waitkey = args.waitkey
        
        # --- Mapping ---
        self.copy_label_to_new_label = args.copy_label_to_new_label
        self.enable_mapping = args.enable_mapping
        self.label_mapping = args.label_mapping
        self.mapping_input_label_dir = args.mapping_input_label_dir
        self.mapping_output_label_dir = args.mapping_output_label_dir
        self.mapping_label_name = args.mapping_label_name
        
        # --- Save option ---
        self.save_result_im = args.save_result_im
        self.save_result_im_resolution =args.save_result_im_resolution
        self.save_mapping_result_im = args.save_mapping_result_im
        
        # ---post proc pose label---
        self.filter_empty_label = args.filter_empty_label
        self.post_proc_pose_label = args.post_proc_pose_label
        self.iou_th = args.iou_th
        
        
    def _bool_str(self, value):
        """Return colored check or cross for booleans."""
        return "✅ True" if value else "❌ False"

    def show_config(self, args):
        """
        Print all dataset and model configuration parameters with emojis and colors.
        """
        # --- ANSI colors ---
        GREEN = "\033[92m"
        RED = "\033[91m"
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        END = "\033[0m"

        print(f"\n{BOLD}{CYAN}📌 ====== Dataset & Model Configuration ======{END}\n")

        # --- Task settings ---
        print(f"{YELLOW}📝 Task Settings:{END}")
        print(f"   🐼 COCO Detection   : {self._bool_str(self.task_coco_detection)}")
        print(f"   🙂 Face Detection   : {self._bool_str(self.task_face_detection)}")
        print(f"   🕺 Pose Detection   : {self._bool_str(self.task_pose_detection)}")
        print(f"   🔀 Multi Detection  : {self._bool_str(self.task_multi_detection)}\n")

        # --- Multi-detection flags ---
        print(f"{YELLOW}🛠️ Multi-Detection Enable Flags:{END}")
        print(f"   🐼 COCO2017 Enabled : {self._bool_str(self.md_enable_coco2017)}")
        print(f"   🙂 Face Enabled     : {self._bool_str(self.md_enable_face)}")
        print(f"   🕺 Pose Enabled     : {self._bool_str(self.md_enable_pose)}\n")

        # --- Confidence thresholds ---
        print(f"{YELLOW}🎯 Confidence Thresholds:{END}")
        print(f"   🐼 COCO2017 : {GREEN}{self.coco2017_conf_th}{END}")
        print(f"   🙂 Face     : {GREEN}{self.face_conf_th}{END}")
        print(f"   🕺 Pose     : {GREEN}{self.pose_conf_th}{END}\n")

        # --- Label values ---
        print(f"{YELLOW}🏷️ Label Values:{END}")
        print(f"   🙂 Face Label : {GREEN}{self.face_label_value}{END}")
        print(f"   🕺 Pose Label : {GREEN}{self.pose_label_value}{END}\n")

        # --- Dataset info ---
        print(f"{YELLOW}📂 Dataset Info:{END}")
        print(f"   📑 Data Type          : {GREEN}{self.data_type}{END}")
        print(f"   🎞️ Video Dir          : {self.data_video_dir}")
        print(f"   ⏭️ Skip Frames        : {GREEN}{self.data_skip_frame}{END}")
        print(f"   🔢 Max Num Images     : {GREEN}{self.data_num}{END}")
        print(f"   📁 Data Root Dir      : {self.data_dir}")
        print(f"   🖼️ Image Dir          : {self.data_img_dir}")
        print(f"   📝 Label Save Dir     : {self.data_save_txt_dir}")
        print(f"   📝 Pose Label Save Dir: {self.data_pose_save_txt_dir}\n")

        # --- Models ---
        print(f"{YELLOW}🤖 Models:{END}")
        detect_status = getattr(self, "detect_model", None)
        face_status   = getattr(self, "face_model", None)
        pose_status   = getattr(self, "pose_model", None)

        # Print load status + resolution info
        print(f"   📦 Detection : {GREEN+'Loaded'+END if detect_status else RED+'Not Loaded'+END}"
            f"   (res={self.detect_resolution})")
        print(f"   🙂 Face      : {GREEN+'Loaded'+END if face_status else RED+'Not Loaded'+END}"
            f"   (res={self.face_resolution})")
        print(f"   🕺 Pose      : {GREEN+'Loaded'+END if pose_status else RED+'Not Loaded'+END}"
            f"   (res={self.pose_resolution})\n")

        # --- Visualization & save options ---
        print(f"{YELLOW}🖼️ Visualization & Save Options:{END}")
        print(f"   👀 Show Images        : {self._bool_str(self.show_result_im)}")
        print(f"   ⏱️ WaitKey (ms)       : {GREEN}{self.waitkey}{END}")
        print(f"   💾 Save Images        : {self._bool_str(self.save_result_im)}")
        print(f"   📏 Save Resolution    : {GREEN}{self.save_result_im_resolution}{END}")
        print(f"   💾 Save Mapping Result: {self._bool_str(self.save_mapping_result_im)}\n")

        # --- Mapping ---
        print(f"{YELLOW}🗺️ Mapping Settings:{END}")
        print(f"   🔄 Enable Mapping     : {self._bool_str(self.enable_mapping)}")
        print(f"   📋 Label Mapping      : {self.label_mapping}")
        print(f"   📥 Mapping Input Dir  : {self.mapping_input_label_dir}")
        print(f"   📤 Mapping Output Dir : {self.mapping_output_label_dir}")
        print(f"   🏷️ Mapping Label Name : {self.mapping_label_name}\n")

        # --- Post-processing ---
        print(f"{YELLOW}🧹 Post-Processing:{END}")
        print(f"   🚫 Filter Empty Label : {self._bool_str(self.filter_empty_label)}")
        print(f"   🕺 Filter FP Pose     : {self._bool_str(self.post_proc_pose_label)}")
        print(f"   📏 IoU Threshold      : {GREEN}{self.iou_th}{END}\n")

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
    
    
    def Auto_labeling_tools(self):
        """
        Automatic labeling entry point for dataset subclasses.

        This method should be implemented by dataset-specific subclasses 
        (e.g., COCO2017 detection, face detection, pose detection). 
        It provides the main pipeline to:
        
        1. Load images from the dataset directory.
        2. Run the corresponding detection or pose estimation model.
        3. Generate YOLO-format labels (bounding boxes and/or keypoints).
        4. Save labels as `.txt` files in the configured output directory.
        5. Optionally visualize and/or save annotated result images.

        Subclasses should implement dataset-specific logic, for example:
            - **COCO2017Detection**: Save object detection bounding boxes 
            using COCO-trained model.
            - **FaceDetection**: Save face bounding boxes with the face model.
            - **PoseDetection**: Save human pose keypoints and optionally 
            skeleton connections.

        Args:
            None

        Returns:
            None
                The function runs the auto-labeling process and saves results 
                to disk (labels and optionally annotated images).
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
        
    
    def extract_frames(self):
        video_dir = Path(self.data_video_dir)
        output_dir = Path(self.data_img_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Skip {self.data_skip_frame} frames")
        # Iterate over all video files in directory
        for video_file in video_dir.glob("*.*"):
            if not video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                continue  # skip non-video files

            cap = cv2.VideoCapture(str(video_file))
            frame_count = 0
            video_name = video_file.stem  # filename without extension

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_filename = output_dir / f"{video_name}_{frame_count:06d}.jpg"
                if frame_count % self.data_skip_frame == 0:
                    cv2.imwrite(str(frame_filename), frame)
                frame_count += 1

            cap.release()
            print(f"✅ Extracted {int(frame_count/self.data_skip_frame)} frames from {video_file.name}, save to dir : {output_dir}")
            
    
    def yolo_to_xyxy(self,box, img_w=1, img_h=1):
        """Convert YOLO box (cx,cy,w,h) normalized to (x1,y1,x2,y2)."""
        cx, cy, w, h = box
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        return [x1, y1, x2, y2]

    def iou(self, box1, box2):
        """Compute IoU between two [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        return inter/union if union > 0 else 0

    def filter_pose_labels(self, ped_class=0):
        pose_dir = Path(self.data_pose_save_txt_dir)
        det_dir = Path(self.data_save_txt_dir)
        out_dir = pose_dir.parent / "train2017-filtered"
        out_dir.mkdir(parents=True, exist_ok=True)
        iou_thr = self.iou_th

        print(f"🔍 Filtering pose labels in: {pose_dir}")
        print(f"📂 Saving filtered results to: {out_dir}\n")

        
        pose_files_list = list(pose_dir.glob("*.txt"))
        pose_files_list = sorted(pose_files_list, key=lambda p: str(p))
        
        for pose_file in tqdm(pose_files_list, desc="Filtering pose label...",unit="File"):
            det_file = det_dir / pose_file.name
            if not det_file.exists():
                print(f"⚠️ No detection file for {pose_file.name}, skipping...")
                continue

            # --- Load detection boxes (pedestrians only) ---
            det_boxes = []
            for line in det_file.read_text().splitlines():
                parts = line.strip().split()
                cls, vals = int(parts[0]), list(map(float, parts[1:5]))
                if cls == ped_class:
                    det_boxes.append(self.yolo_to_xyxy(vals))

            # --- Load pose labels ---
            pose_lines = pose_file.read_text().splitlines()
            keep_lines = []
            for line in pose_lines:
                parts = line.strip().split()
                vals = list(map(float, parts[1:5]))  # cx, cy, w, h
                pose_box = self.yolo_to_xyxy(vals)

                if any(self.iou(pose_box, d) > iou_thr for d in det_boxes):
                    keep_lines.append(line)

            # --- Save filtered pose labels ---
            out_file = out_dir / pose_file.name
            out_file.write_text("\n".join(keep_lines) + ("\n" if keep_lines else ""))

            # --- Log result ---
            # if keep_lines:
            #     print(f"✅ {pose_file.name}: {len(keep_lines)} kept / {len(pose_lines)} total")
            # else:
            #     print(f"🗑️ {pose_file.name}: all {len(pose_lines)} removed (no matching pedestrian)")
                
            # draw + filter
            img_path = Path(self.data_img_dir) / f"{pose_file.stem}.jpg"
            if img_path.exists():
                self.draw_pose_on_image(img_path, keep_lines)

        print("\n🎉 Filtering complete!")
        

    def draw_pose_on_image(self,img_path: Path, pose_lines):
        
        out_vis_dir = Path(self.data_pose_save_txt_dir).parent / "filtered_train2017_visual"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Image not found: {img_path}")
            return

        h, w = img.shape[:2]

        # --- Draw pose detections ---
        for line in pose_lines:
            parts = line.strip().split()
            cx, cy, bw, bh = map(float, parts[1:5])

            # Bounding box (blue)
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Keypoints
            kps = list(map(float, parts[5:]))
            keypoints = []
            for i in range(0, len(kps), 3):
                kx, ky, v = kps[i:i+3]
                px, py = int(kx * w), int(ky * h)
                if v > 0.5:  # visible
                    cv2.circle(img, (px, py), 3, (0, 255, 0), -1)
                    keypoints.append((px, py))
                else:
                    keypoints.append(None)

            # Draw skeleton connections
            for i, (j1, j2) in enumerate(SKELETON):
                if j1 < len(keypoints) and j2 < len(keypoints):
                    if keypoints[j1] and keypoints[j2]:
                        cv2.line(img, keypoints[j1], keypoints[j2], COLORS[i], 2)

        # --- Save visualization to specific directory ---
        out_vis_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_vis_dir / img_path.name
        im_h,im_w = self.save_result_im_resolution[:2]
        vis_img = cv2.resize(img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Filtered Pose detection label",vis_img)
        cv2.waitKey(self.waitkey * 2)
        cv2.imwrite(str(out_file), vis_img)
        # print(f"🖼️ Saved visualization: {out_file}")