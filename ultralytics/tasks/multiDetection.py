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
        
    def Get_Multi_Detection_YOLO_Txt_Label(self):
        
        img_path_list = []  
        
        for ext in img_extension:
            paths = list(Path(self.data_img_dir).glob(ext))
            img_path_list.extend(paths)
        
        for img_path in tqdm(img_path_list,desc=f"Processing images..."):
            # print(img_path)
            if self.md_enable_coco2017:
                result_img = self.coco_det.Save_YOLO_txt_Labels(img_path=img_path, image=None)

                
            if self.md_enable_face:
                result_img = self.face_det.Save_YOLO_txt_Labels(img_path=img_path, image = result_img)
                # self.Save_Face_Detection_YOLO_Txt_Label(img_path=img_path)
            
            
            if self.enable_mapping:
                self.filter_and_remap_yolo_labels()
                
            if self.md_enable_pose:
                if not self.md_enable_coco2017 and not self.md_enable_face:
                    result_img = None 
                self.pose_det.Save_YOLO_txt_Labels(img_path=img_path, image = result_img)
                # self.Save_Pose_Detection_YOLO_Txt_Label(img_path=img_path)
                
                
                
            
        
        
        
        
    