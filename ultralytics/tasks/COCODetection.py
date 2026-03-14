from pathlib import Path
from engine.dataset import BaseDataset
import cv2
import random
from tqdm import tqdm

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Pre-generate random colors for each class
COLORS = {}
def get_color(cls_id):
    if cls_id not in COLORS:
        # generate a bright random color
        COLORS[cls_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return COLORS[cls_id]

img_extension = ['*.jpg', '*.png', '*.bmp', '*.jpeg']

class COCODetection(BaseDataset):
    def __init__(self,args):
        super().__init__(args)
        # Add an output directory for saving visualized results
        self.result_img_dir = Path(args.data_save_txt_dir).parent / "vis_results"
        self.result_img_dir.mkdir(parents=True, exist_ok=True)
    
    
    def Auto_labeling_tools(self):
        img_path_list=[]
        for ext in img_extension:
            paths = list(Path(self.data_img_dir).glob(ext))
            img_path_list.extend(paths)
        
        img_path_list = sorted(img_path_list,key=lambda p: str(p))
        
        img_path_list = img_path_list[:self.data_num] if self.data_num<=len(img_path_list) else img_path_list
        
        for img_path in tqdm(img_path_list,desc="Auto Labeling COCO2017 Detection..."):
            self.Save_YOLO_txt_Labels(img_path=img_path)    
    
    def Save_YOLO_txt_Labels_ver1(self, img_path, image=None):
        img = cv2.imread(img_path)
        img_h,img_w = img.shape[:2]
        raw_img = img.copy() # ⬅️ keep original (no drawings)

        # print(f"img_h:{img_h}, img_w:{img_w}")

        # if new_label_path.exists():
        #     print(f"File {new_label_path} exist..PASS")
        #     return None
        results = self.detect_model.predict(img, conf=self.coco2017_conf_th, verbose=False)
        
        # === robust check for empty detections ===
        # if results is None or len(results) == 0 or all(
        #     (r.boxes is None or r.boxes.shape[0] == 0) for r in results
        # ):
        #     print(f"[INFO] No detection found for {Path(img_path).name}, skip creating label file")
        #     return None     
        
        Path(self.data_save_txt_dir).mkdir(parents=True, exist_ok=True)
        new_label_path = Path(self.data_save_txt_dir) / (Path(img_path).stem + ".txt")
        
        wanted_cls = set(self.label_mapping.keys())
            
        wrote_any = False
        
        with new_label_path.open('w') as f:
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xywhn.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy() # ✅ get confidences
                
                for cls_id, box, conf in zip(cls_ids,boxes,confs):
                    cx,cy,w,h = box
                    new_class_id = None
                    if self.enable_mapping:
                        if cls_id in wanted_cls:
                            new_class_id = self.label_mapping[cls_id]
                            label_line = f"{new_class_id} {cx:6f} {cy:6f} {w:6f} {h:6f}\n"
                            f.write(label_line)
                            wrote_any = True
                    else:
                        label_line = f"{cls_id} {cx:6f} {cy:6f} {w:6f} {h:6f}\n"
                        f.write(label_line)
                        wrote_any = True            
                    

                    if self.show_result_im or self.save_result_im:
                        color =get_color(int(cls_id))
                        x1 = int((cx - w/2)*img_w)
                        y1 = int((cy - h/2)*img_h)
                        x2 = int((cx + w/2)*img_w)
                        y2 = int((cy + h/2)*img_h)
                        
                        if self.enable_mapping and new_class_id is not None:
                            label_name = self.mapping_label_name[int(new_class_id)] if int(new_class_id) < len(self.mapping_label_name) else str(new_class_id)
                        elif not self.enable_mapping:
                            label_name = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else str(cls_id)
                        else:
                            label_name = None
                        
                        if label_name is not None:
                            txt = f"{label_name} {conf:.2f}"
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                img,
                                txt,
                                (x1, y1 - 10),              # shift a little higher above box
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,                        # ⬅️ font scale (was 0.5)
                                color,
                                2                           # ⬅️ thickness (was 1)
                            )
        
        
        # ✅ Remove file if nothing was written
        if not wrote_any and self.filter_empty_label:
            new_label_path.unlink(missing_ok=True) # delete new_label_path if it exists
            print(f"[INFO] {Path(img_path).name}: detections filtered out — deleted empty label file.")
            return None
        
        if self.filter_empty_label:
            # === Save only images that have labels ===
            labeled_img_dir = Path(self.data_save_txt_dir).parent / "has_labels"
            labeled_img_dir.mkdir(parents=True, exist_ok=True)
            labeled_img_path = labeled_img_dir / Path(img_path).name
            cv2.imwrite(str(labeled_img_path), raw_img)
            print(f"[INFO] Saved labeled image: {labeled_img_path.name}")
                 
        if self.show_result_im and not self.md_enable_face and not self.md_enable_pose:
            im_h,im_w = self.save_result_im_resolution[:2]
            vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Detection Auto Label",vis_img)
            cv2.waitKey(self.waitkey)
            
        # === Save annotated image if enabled ===
        if self.save_result_im and not self.task_face_detection :
            im_h,im_w = self.save_result_im_resolution[:2]
            result_path = self.result_img_dir / Path(img_path).name
            if not result_path.exists():
                vis_img = cv2.resize(img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(result_path), vis_img)
            # print(f"[INFO] Saved result image: {result_path}")
        return img
    
    
    
    def Save_YOLO_txt_Labels(self, img_path, image=None):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}")
            return None
        img_h, img_w = img.shape[:2]
        raw_img = img.copy()

        # === Configurable parameters (adjustable like in PoseDetection) ===
        imgsz_high = getattr(self, "detect_resolution", 1920)   # better for small/far objects
        imgsz_low  = getattr(self, "imgsz_low", 640)     # better for large/close objects
        area_th    = getattr(self, "dynamic_area_th", 0.04)  # normalized area threshold

        Path(self.data_save_txt_dir).mkdir(parents=True, exist_ok=True)
        new_label_path = Path(self.data_save_txt_dir) / (Path(img_path).stem + ".txt")

        wanted_cls = set(self.label_mapping.keys())
        wrote_any = False

        # === Run detection at two resolutions ===
        try:
            high_results_list = self.detect_model.predict(img, imgsz=imgsz_high, conf=self.coco2017_conf_th, verbose=False)
            low_results_list  = self.detect_model.predict(img, imgsz=imgsz_low,  conf=self.coco2017_conf_th, verbose=False)
        except Exception as e:
            print(f"Inference error: {e}")
            return None

        # === Extract detections into uniform list ===
        def extract_dets(results_list):
            dets = []
            if results_list is None or len(results_list) == 0:
                return dets
            r = results_list[0]
            if r.boxes is None or len(r.boxes) == 0:
                return dets
            boxes = r.boxes.xywhn.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for cls_id, box, conf in zip(cls_ids, boxes, confs):
                dets.append({
                    "cls_id": cls_id,
                    "box": box,   # normalized cx,cy,w,h
                    "conf": conf
                })
            return dets

        high_dets = extract_dets(high_results_list)
        low_dets  = extract_dets(low_results_list)

        # === Select detections by normalized box area ===
        selected = []

        for d in high_dets:
            _, _, w, h = d["box"]
            if w * h <= area_th:
                d["source"] = "high"
                selected.append(d)

        for d in low_dets:
            _, _, w, h = d["box"]
            if w * h > area_th:
                d["source"] = "low"
                selected.append(d)

        # Fallback: if empty, prefer low_dets then high_dets
        if len(selected) == 0:
            selected = low_dets if len(low_dets) > 0 else high_dets
            for d in selected:
                d["source"] = "fallback"

   
        with new_label_path.open('w') as f:
            for d in selected:
                cx, cy, w, h = d["box"]
                cls_id = d["cls_id"]
                conf = d["conf"]

                # Class mapping if enabled
                new_class_id = None
                if self.enable_mapping:
                    if cls_id in wanted_cls:
                        new_class_id = self.label_mapping[cls_id]
                        f.write(f"{new_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        wrote_any = True
                else:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    wrote_any = True

                # === Draw detections ===
                if self.show_result_im or self.save_result_im:
                    color = get_color(int(cls_id))
                    x1, y1 = int((cx - w/2)*img_w), int((cy - h/2)*img_h)
                    x2, y2 = int((cx + w/2)*img_w), int((cy + h/2)*img_h)

                    # Get label name
                    if self.enable_mapping and new_class_id is not None:
                        label_name = self.mapping_label_name[int(new_class_id)] if int(new_class_id) < len(self.mapping_label_name) else str(new_class_id)
                    elif not self.enable_mapping:
                        label_name = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else str(cls_id)
                    else:
                        label_name = None

                    if label_name is not None:
                        txt = f"{label_name} {conf:.2f}"
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, txt, (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    else:
                        print("label_name is None~~~")

        # === Clean up ===
        if not wrote_any and self.filter_empty_label:
            new_label_path.unlink(missing_ok=True)
            # print(f"[INFO] {Path(img_path).name}: no detections — deleted empty label file.")
            return None

        # === Optional image saving ===
        if self.filter_empty_label:
            labeled_img_dir = Path(self.data_save_txt_dir).parent / "has_labels"
            labeled_img_dir.mkdir(parents=True, exist_ok=True)
            labeled_img_path = labeled_img_dir / Path(img_path).name
            cv2.imwrite(str(labeled_img_path), raw_img)
            # print(f"[INFO] Saved labeled image: {labeled_img_path.name}")

        if self.show_result_im and not self.md_enable_face and not self.md_enable_pose:
            im_h, im_w = self.save_result_im_resolution[:2]
            vis_img = cv2.resize(image if image is not None else img, (im_w, im_h))
            cv2.imshow("Detection Auto Label", vis_img)
            cv2.waitKey(self.waitkey)

        if self.save_result_im and not self.task_face_detection:
            im_h, im_w = self.save_result_im_resolution[:2]
            result_path = self.result_img_dir / Path(img_path).name
            if not result_path.exists():
                vis_img = cv2.resize(img, (im_w, im_h))
                cv2.imwrite(str(result_path), vis_img)

        # print(f"Processed {img_path.name} | kept {len(selected)} detections | area_th={area_th} (high={imgsz_high}, low={imgsz_low})")
        return img
