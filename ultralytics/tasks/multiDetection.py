from engine.dataset import BaseDataset
from pathlib import Path
from tasks.faceDetection import FaceDetection
from tasks.poseDetection import PoseDetection
from tasks.COCODetection import COCODetection

import sys
from ultralytics import YOLO
import cv2
from tqdm import tqdm

img_extension = ['*.jpg', '*.png', '*.bmp', '*.jpeg']

class MultiDetectTask(BaseDataset):
    
    def __init__(self,args):
        super().__init__(args)
        self.coco_det = COCODetection(args)
        self.face_det = FaceDetection(args)
        self.pose_det = PoseDetection(args)
        
    def Auto_labeling_tools(self):
        print(f"📂 Data type is {self.data_type}")
        
        # --- Validate data type ---
        assert self.data_type in ["videos", "images"], \
            "❌ Data Type is not correct! Data type should be 'images' or 'videos'"
        
        # --- Handle video case ---
        if self.data_type == "videos":
            print(f"🎥 Data type is {self.data_type}, starting frame extraction...")
            self.extract_frames()
        
        # --- Collect images ---
        img_path_list = []
        for ext in img_extension:
            paths = list(Path(self.data_img_dir).glob(ext))
            img_path_list.extend(paths)
        
        img_path_list = sorted(img_path_list, key=lambda p: str(p))
        img_path_list = (img_path_list[:self.data_num] if self.data_num <= len(img_path_list) else img_path_list)
        
        # --- Multi-task labeling loop ---
        for img_path in tqdm(img_path_list, desc=f"⚡ 🐼 🙂 🕺 Processing Multi-Task Labeling..."):
            result_img = None
            
            if self.md_enable_coco2017:
                result_img = self.coco_det.Save_YOLO_txt_Labels(
                    img_path=img_path, image=None
                )
            if result_img is not None:
                # print("result_img is not None")
                if self.md_enable_face:
                    result_img = self.face_det.Save_YOLO_txt_Labels(
                        img_path=img_path, image=result_img
                    )
                
                if self.md_enable_pose:
                    # print(f"md_enable_pose : {self.md_enable_pose}")
                    if not self.md_enable_coco2017 and not self.md_enable_face:
                        result_img = None
                
                    self.pose_det.Save_YOLO_txt_Labels(
                        img_path=img_path, image=result_img
                    )
            else:
                print("Result image is None !!")
        
        if self.post_proc_pose_label and self.md_enable_pose:
            print(f"Start filtering FP pose detection labels...")
            self.filter_pose_labels()

                
                
                
            
        
        
        
        
    