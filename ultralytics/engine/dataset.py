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
        
        self.label_mapping = args.label_mapping
        
        # --- Mapping ---
        self.label_mapping = args.label_mapping
        self.mapping_input_label_dir = args.mapping_input_label_dir
        self.mapping_output_label_dir = args.mapping_output_label_dir
        
        
    def Save_COCO2017_Detection_YOLO_Txt_Label(self, img_path):
        img = cv2.imread(img_path)
        img_h,img_w = img.shape[:2]
        # print(f"img_h:{img_h}, img_w:{img_w}")
        Path(self.data_save_txt_dir).mkdir(parents=True, exist_ok=True)
        new_label_path = Path(self.data_save_txt_dir) / (Path(img_path).stem + ".txt")
      
        
        results = self.detect_model.predict(img, conf=0.25, verbose=False)
        
        with new_label_path.open('w') as f:
            for r in results:
                boxes = r.boxes.xywhn.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                for cls_id, box in zip(cls_ids,boxes):
                    cx,cy,w,h = box
                    label_line = f"{cls_id} {cx:6f} {cy:6f} {w:6f} {h:6f}\n"
                    f.write(label_line)

                    if self.show_result_im:
                        x1 = int((cx - w/2)*img_w)
                        y1 = int((cy - h/2)*img_h)
                        x2 = int((cx + w/2)*img_w)
                        y2 = int((cy + h/2)*img_h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(img, str(cls_id), (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
                 
        if self.show_result_im:
            cv2.imshow("Detection Auto Label",img)
            key = cv2.waitKey(0)
        return img
    
    def Save_Face_Detection_YOLO_Txt_Label(self, img_path):
        img = cv2.imread(img_path)
        img_h,img_w = img.shape[:2]
        # print(f"img_h:{img_h}, img_w:{img_w}")
        Path(self.data_save_txt_dir).mkdir(parents=True, exist_ok=True)
        new_label_path = Path(self.data_save_txt_dir) / (Path(img_path).stem + ".txt")
      
        
        results = self.detect_model.predict(img, conf=0.25, verbose=False)
        
        with new_label_path.open('a') as f:
            for r in results:
                boxes = r.boxes.xywhn.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                for cls_id, box in zip(cls_ids,boxes):
                    cx,cy,w,h = box
                    label_line = f"{self.face_label_value} {cx:6f} {cy:6f} {w:6f} {h:6f}\n"
                    f.write(label_line)

                    if self.show_result_im:
                        x1 = int((cx - w/2)*img_w)
                        y1 = int((cy - h/2)*img_h)
                        x2 = int((cx + w/2)*img_w)
                        y2 = int((cy + h/2)*img_h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
                        cv2.putText(img, str(cls_id), (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        return img
    
    def Save_Pose_Detection_YOLO_Txt_Label(self, img_path):
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        new_label_path = Path(self.data_pose_save_txt_dir) / (Path(img_path).stem + ".txt")
        Path(self.data_pose_save_txt_dir).mkdir(parents=True, exist_ok=True)
        # Run pose detection
        results = self.pose_model.predict(img, conf=0.25, verbose=False)

        with new_label_path.open("w") as f:
            for r in results:
                boxes = r.boxes.xywhn.cpu().numpy()   # normalized bboxes (cx,cy,w,h)
                kpts = r.keypoints.xyn.cpu().numpy()  # normalized keypoints (num,17,2)
                kpts_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones_like(kpts[:,:,0])

                for box, kpt, confs in zip(boxes, kpts, kpts_conf):
                    cx, cy, w, h = box

                    # Build YOLO line: cls cx cy w h kpt1_x kpt1_y v1 ...
                    label_parts = [str(self.pose_label_value), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

                    for (x, y), v in zip(kpt, confs):
                        v_flag = 2 if v > 0.5 else 1  # visibility flag (you can tweak threshold)
                        label_parts += [f"{x:.6f}", f"{y:.6f}", str(v_flag)]

                        # Optionally draw on image
                        if self.show_result_im and v > 0.5:
                            cv2.circle(img, (int(x * img_w), int(y * img_h)), 2, (0,255,0), -1)

                    f.write(" ".join(label_parts) + "\n")

        if self.show_result_im:
            cv2.imshow("Pose Auto-Label", img)
            key = cv2.waitKey(0)
         

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

        wanted_cls = set(self.label_mapping).keys()

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

        print(f"Filtered and remapped labels saved to: {output_dir}")