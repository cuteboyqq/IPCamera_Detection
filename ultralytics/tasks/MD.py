from engine.dataset import BaseDataset
from pathlib import Path
import sys
from ultralytics import YOLO
import cv2
from tqdm import tqdm

img_extension = ['*.jpg', '*.png', '*.bmp', '*.jpeg']

class MultiDetectTask(BaseDataset):
    
    def __init__(self,args):
        super().__init__(args)
        
        
    def Get_Multi_Detection_YOLO_Txt_Label(self):
        
        img_path_list = []  
        
        for ext in img_extension:
            paths = list(Path(self.data_img_dir).glob(ext))
            img_path_list.extend(paths)
        
        for img_path in tqdm(img_path_list,desc=f"Processing images..."):
            # print(img_path)
            
            if self.md_enable_coco2017:
                result_img = self.Save_COCO2017_Detection_YOLO_Txt_Label(img_path=img_path)
                # if self.show_result_im:
                #     cv2.imshow("Detection Auto Label",result_img)
                #     key = cv2.waitKey(0)
                #     if key == ord('q'):
                #         break
                
                
            if self.md_enable_face:
                self.Save_Face_Detection_YOLO_Txt_Label(img_path=img_path)
                
                
            if self.md_enable_pose:
                self.Save_Pose_Detection_YOLO_Txt_Label(img_path=img_path)
                
                
                
            
        
        
        
        
    