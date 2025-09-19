from pathlib import Path
from engine.dataset import BaseDataset
import cv2
import random

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


class COCODetection(BaseDataset):
    def __init__(self,args):
        super().__init__(args)
        # Add an output directory for saving visualized results
        self.result_img_dir = Path(args.data_save_txt_dir).parent / "vis_results"
        self.result_img_dir.mkdir(parents=True, exist_ok=True)
        
    def Save_YOLO_txt_Labels(self, img_path, image):
        img = cv2.imread(img_path)
        img_h,img_w = img.shape[:2]
        # print(f"img_h:{img_h}, img_w:{img_w}")
        new_label_path = Path(self.data_save_txt_dir) / (Path(img_path).stem + ".txt")

        # if new_label_path.exists():
        #     print(f"File {new_label_path} exist..PASS")
        #     return None
        
        Path(self.data_save_txt_dir).mkdir(parents=True, exist_ok=True)
        results = self.detect_model.predict(img, conf=self.coco2017_conf_th, verbose=False)
        
        wanted_cls = set(self.label_mapping.keys())
             
        with new_label_path.open('w') as f:
            for r in results:
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
                    else:
                        label_line = f"{cls_id} {cx:6f} {cy:6f} {w:6f} {h:6f}\n"
                        f.write(label_line)            
                    

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
                                    
                 
        if self.show_result_im:
            cv2.imshow("Detection Auto Label",img)
            key = cv2.waitKey(0)
            
        # === Save annotated image if enabled ===
        if self.save_result_im and not self.md_enable_face :
            im_h,im_w = self.save_result_im_resolution[:2]
            result_path = self.result_img_dir / Path(img_path).name
            if not result_path.exists():
                vis_img = cv2.resize(img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(result_path), vis_img)
            # print(f"[INFO] Saved result image: {result_path}")
        return img