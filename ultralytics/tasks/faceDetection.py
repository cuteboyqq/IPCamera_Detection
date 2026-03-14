from engine.dataset import BaseDataset
from pathlib import Path
import cv2
import random
from tqdm import tqdm
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

image_extension = ['*.jpg', '*.png', '*.bmp', '*.jpeg']
import shutil 
import os
class FaceDetection(BaseDataset):
    
    def __init__(self,args):
        super().__init__(args)
        # Add an output directory for saving visualized results
        self.result_img_dir = Path(args.data_save_txt_dir).parent / "vis_results"
        self.result_img_dir.mkdir(parents=True, exist_ok=True)

    def Auto_labeling_tools(self):
        img_path_list = []
        for ext in image_extension:
            paths = list(Path(self.data_img_dir).glob(ext))
            img_path_list.extend(paths)
            
        img_path_list = sorted(img_path_list, key=lambda p: str(p))
        
        img_path_list = img_path_list[:self.data_num] if self.data_num<=len(img_path_list) else img_path_list
        
        for img_path in tqdm(img_path_list, desc="Auto Labeling Face Detection..."):
            self.Save_YOLO_txt_Labels(img_path)

    def Save_YOLO_txt_Labels(self, img_path, image=None):
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        new_label_path = Path(self.data_save_txt_dir) / (Path(img_path).stem + ".txt")
        
        
        if new_label_path.exists() and not self.task_multi_detection:
            #print("txt file exist , skip it ~~~~")
            return image if image is not None else img
        
        # original_label_path = Path(self.data_img_dir) / (Path(img_path).stem + ".txt")
        img_path = Path(img_path)

        # convert image path → label path
        original_label_path = Path(
            str(img_path)
            .replace("/images/", "/labels/detection/")
        ).with_suffix(".txt")
        # print(f"original_label_path:{original_label_path}")

        Path(self.data_save_txt_dir).mkdir(parents=True, exist_ok=True)

        
        
        # -------- copy original label first --------
        if original_label_path.exists() and self.copy_label_to_new_label:
            shutil.copy(original_label_path, new_label_path)
        elif not self.task_multi_detection:
            # print("create empty file --> original doesn't exist")
            new_label_path.touch()   # create empty file if original doesn't exist

        # -------- run face detection --------
        results = self.face_model.predict(img, imgsz=1280, conf=self.face_conf_th, verbose=False)

        # -------- append face labels --------
        with new_label_path.open('a') as f:
            for r in results:
                boxes = r.boxes.xywhn.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)

                for cls_id, box in zip(cls_ids, boxes):
                    cx, cy, w, h = box
                    label_line = f"{self.face_label_value} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                    f.write(label_line)
                    
                    if self.show_result_im or self.save_result_im:
                            x1 = int((cx - w/2)*img_w)
                            y1 = int((cy - h/2)*img_h)
                            x2 = int((cx + w/2)*img_w)
                            y2 = int((cy + h/2)*img_h)
                            cv2.rectangle(image if image is not None else img, (x1, y1), (x2, y2),  (0,0,255), 2)
                            label_name = "Face"
                            cv2.putText(
                                image if image is not None else img,
                                label_name,
                                (x1, y1 - 10),              # shift a little higher above box
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,                        # ⬅️ font scale (was 0.5)
                                (0,0,255),
                                2                           # ⬅️ thickness (was 1)
                            )
        if self.show_result_im and not self.md_enable_pose:
            im_h,im_w = self.save_result_im_resolution[:2]
            vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Detection Auto Label",vis_img)
            key = cv2.waitKey(self.waitkey)
            
        # === Save annotated image if enabled ===
        if self.save_result_im and not self.task_pose_detection and not self.md_enable_pose:
            im_h,im_w = self.save_result_im_resolution[:2]
            result_path = self.result_img_dir / Path(img_path).name
            vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(result_path), vis_img)
            # print(f"[INFO] Saved result image: {result_path}")
        
        return image if image is not None else img
        
    # def Save_YOLO_txt_Labels(self, img_path, image=None):
    #     img = cv2.imread(img_path)
    #     img_h,img_w = img.shape[:2]
    #     # print(f"img_h:{img_h}, img_w:{img_w}")
    #     new_label_path = Path(self.data_save_txt_dir) / (Path(img_path).stem + ".txt")
        
    #     # if new_label_path.exists():
    #     #     print(f"File {new_label_path} exist..PASS")
    #     #     return None
        
    #     Path(self.data_save_txt_dir).mkdir(parents=True, exist_ok=True)
        
    #     results = self.face_model.predict(img, imgsz=1280, conf=self.face_conf_th, verbose=False)
        
    #     with new_label_path.open('a') as f:
    #         for r in results:
    #             boxes = r.boxes.xywhn.cpu().numpy()
    #             cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                
    #             for cls_id, box in zip(cls_ids,boxes):
    #                 cx,cy,w,h = box
    #                 label_line = f"{self.face_label_value} {cx:6f} {cy:6f} {w:6f} {h:6f}\n"
    #                 f.write(label_line)

    #                 if self.show_result_im or self.save_result_im:
    #                     x1 = int((cx - w/2)*img_w)
    #                     y1 = int((cy - h/2)*img_h)
    #                     x2 = int((cx + w/2)*img_w)
    #                     y2 = int((cy + h/2)*img_h)
    #                     cv2.rectangle(image if image is not None else img, (x1, y1), (x2, y2),  (0,0,255), 2)
    #                     label_name = "Face"
    #                     cv2.putText(
    #                         image if image is not None else img,
    #                         label_name,
    #                         (x1, y1 - 10),              # shift a little higher above box
    #                         cv2.FONT_HERSHEY_SIMPLEX,
    #                         1.0,                        # ⬅️ font scale (was 0.5)
    #                         (0,0,255),
    #                         2                           # ⬅️ thickness (was 1)
    #                     )
                                    
                 
    #     if self.show_result_im and not self.md_enable_pose:
    #         im_h,im_w = self.save_result_im_resolution[:2]
    #         vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
    #         cv2.imshow("Detection Auto Label",vis_img)
    #         key = cv2.waitKey(self.waitkey)
            
    #     # === Save annotated image if enabled ===
    #     if self.save_result_im and not self.task_pose_detection:
    #         im_h,im_w = self.save_result_im_resolution[:2]
    #         result_path = self.result_img_dir / Path(img_path).name
    #         vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
    #         cv2.imwrite(str(result_path), vis_img)
    #         # print(f"[INFO] Saved result image: {result_path}")
        
    #     return image if image is not None else img