import cv2
from engine.dataset import BaseDataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
img_extension=['*.jpg', '*.png', '*.bmp', '*.jpeg']


class PoseDetection(BaseDataset):
    
    def __init__(self, args):
        super().__init__(args)
        self.skeleton_save_dir = Path(self.data_pose_save_txt_dir).parent / "visual_points"  # save dir for drawn images
        self.skeleton_save_dir.mkdir(parents=True, exist_ok=True)

        self.SKELETON = [
            [0, 1], [0, 2], [1, 3], [2, 4], 
            [0, 5], [0, 6], [5, 7], [7, 9], 
            [6, 8], [8, 10], [5, 6], [5, 11], 
            [6, 12], [11, 12], [11, 13], [13, 15], 
            [12, 14], [14, 16]
        ]
        # Colors for skeleton lines
        self.COLORS = [(255,0,0), (0,255,0), (0,0,255),
                       (255,255,0), (255,0,255), (0,255,255),
                       (128,0,128), (255,165,0), (0,128,255)]
        
    
    def Auto_labeling_tools(self):
        img_path_list = []
        for ext in img_extension:
            paths = list(Path(self.data_img_dir).glob(ext))
            img_path_list.extend(paths)
            
        img_path_list = sorted(img_path_list,key=lambda p: str(p))
        img_path_list = img_path_list[:self.data_num] if self.data_num<=len(img_path_list) else img_path_list
        
        for img_path in tqdm(img_path_list,desc="Auto Labeling Pose Detection..."):
            self.Save_YOLO_txt_Labels(img_path)

    
    
    def Save_YOLO_txt_Labels(self, img_path, image=None):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}")
            return None
        img_h, img_w = img.shape[:2]

        new_label_path = Path(self.data_pose_save_txt_dir) / (Path(img_path).stem + ".txt")
        
        # ---- Skip if label already exists ----
        if new_label_path.exists():
            # print(f"[Skip] Label already exists: {new_label_path}")
            return None
        
        Path(self.data_pose_save_txt_dir).mkdir(parents=True, exist_ok=True)

        # Configurable params (you can set self.dynamic_area_th elsewhere)
        imgsz_high = getattr(self, "pose_resolution", 1280)   # for small/far people
        imgsz_low  = getattr(self, "imgsz_low", 640)     # for large/close people
        area_th = getattr(self, "dynamic_area_th", 0.04) # normalized area threshold (w*h)

        # ---- Run both inferences ----
        try:
            high_results_list = self.pose_model.predict(img, imgsz=imgsz_high, conf=self.pose_conf_th, verbose=False)
            low_results_list  = self.pose_model.predict(img, imgsz=imgsz_low,  conf=self.pose_conf_th, verbose=False)
        except Exception as e:
            print(f"Inference error: {e}")
            return None

        # Ultralytics returns list of results per image; we use first (single image)
        high_r = high_results_list[0] if len(high_results_list) > 0 else None
        low_r  = low_results_list[0]  if len(low_results_list)  > 0 else None

        # Helper to extract detections into a uniform list of dicts
        def extract_dets(r):
            dets = []
            if r is None:
                return dets
            boxes = r.boxes.xywhn.cpu().numpy() if hasattr(r, "boxes") and len(r.boxes) > 0 else np.zeros((0,4))
            boxes_conf = r.boxes.conf.cpu().numpy() if hasattr(r, "boxes") and len(r.boxes) > 0 else np.zeros((0,))
            kpts = r.keypoints.xyn.cpu().numpy() if hasattr(r, "keypoints") and r.keypoints is not None else np.zeros((0,17,2))
            kpts_conf = r.keypoints.conf.cpu().numpy() if (hasattr(r, "keypoints") and getattr(r.keypoints, "conf", None) is not None) else None

            n = boxes.shape[0]
            for i in range(n):
                box = boxes[i]            # cx,cy,w,h normalized
                box_conf = float(boxes_conf[i]) if boxes_conf is not None and len(boxes_conf)>i else 0.0
                kpt = kpts[i] if (kpts is not None and len(kpts)>i) else np.zeros((17,2))
                kpt_conf = kpts_conf[i] if (kpts_conf is not None and len(kpts_conf)>i) else np.ones((kpt.shape[0],))
                dets.append({
                    "box": box,           # normalized cx,cy,w,h
                    "box_conf": box_conf,
                    "kpt": kpt,           # normalized x,y
                    "kpt_conf": kpt_conf
                })
            return dets

        high_dets = extract_dets(high_r)
        low_dets  = extract_dets(low_r)

        # ---- Select detections by area ----
        # Keep high_res dets with area <= area_th (small/far)
        # Keep low_res dets with  area >  area_th (large/close)
        selected = []

        for d in high_dets:
            _,_,w,h = d["box"]
            area = float(w * h)
            if area <= area_th:
                d["source"] = "high"
                selected.append(d)

        for d in low_dets:
            _,_,w,h = d["box"]
            area = float(w * h)
            if area > area_th:
                d["source"] = "low"
                selected.append(d)

        # If nothing selected (edge case), fall back to low_dets then high_dets
        if len(selected) == 0:
            if len(low_dets) > 0:
                for d in low_dets:
                    d["source"] = "low"
                    selected.append(d)
            else:
                for d in high_dets:
                    d["source"] = "high"
                    selected.append(d)

        # ---- Optionally deduplicate overlapping boxes (simple IoU-based) ----
        # If you want to avoid duplicates where both runs detected same person with slightly different boxes,
        # you can dedupe here. For now we assume area split prevents duplicates.

        # ---- Write labels & draw ----
        with new_label_path.open("w") as f:
            for det in selected:
                cx, cy, w, h = det["box"]
                box_conf = det.get("box_conf", 0.0)
                kpt = det["kpt"]
                kpt_conf = det["kpt_conf"] if det.get("kpt_conf", None) is not None else np.ones((kpt.shape[0],))

                # YOLO-format line
                label_parts = [str(self.pose_label_value), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

                points = []
                for (x, y), v in zip(kpt, kpt_conf):
                    v_flag = 2 if v > 0.5 else (1 if v > 0 else 0)
                    label_parts += [f"{x:.6f}", f"{y:.6f}", str(v_flag)]

                    px, py = int(x * img_w), int(y * img_h)
                    points.append((px, py, float(v)))
                    # draw keypoint
                    cv2.circle(img, (px, py), 4, (0, 255, 0), -1)

                f.write(" ".join(label_parts) + "\n")

                # draw bbox
                x1 = int((cx - w/2) * img_w)
                y1 = int((cy - h/2) * img_h)
                x2 = int((cx + w/2) * img_w)
                y2 = int((cy + h/2) * img_h)
                # cv2.rectangle(image if image is not None else img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # draw skeleton
                for i, (j1, j2) in enumerate(self.SKELETON):
                    if j1 < len(points) and j2 < len(points):
                        x1_p, y1_p, v1 = points[j1]
                        x2_p, y2_p, v2 = points[j2]
                        color = self.COLORS[i % len(self.COLORS)]
                        cv2.line(image if image is not None else img, (x1_p, y1_p), (x2_p, y2_p), color, 2)

                # draw conf text (top-left of bbox)
                conf_text = f"{box_conf:.2f}"
                text_x = max(0, x1)
                text_y = max(0, y1 - 35)
                cv2.putText(image if image is not None else img, conf_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        # ---- show / save as before ----
        if self.show_result_im:
            im_h, im_w = self.save_result_im_resolution[:2]
            # vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            vis_img = image if image is not None else img
            cv2.imshow("Pose detection", vis_img)
            cv2.waitKey(self.waitkey)

        if self.save_result_im:
            im_h, im_w = self.save_result_im_resolution[:2]
            save_path = self.skeleton_save_dir / Path(img_path).name
            vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(save_path), vis_img)

        # print(f"Processed {img_path.name} | kept {len(selected)} detections | area_th={area_th} (high={imgsz_high}, low={imgsz_low})")
        return img
    
    
    def Save_YOLO_txt_Labels_ver1(self, img_path, image=None):
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]

        new_label_path = Path(self.data_pose_save_txt_dir) / (Path(img_path).stem + ".txt")
        # if new_label_path.exists():
        #     print(f"File {new_label_path} exist..PASS")
        #     return None
        Path(self.data_pose_save_txt_dir).mkdir(parents=True, exist_ok=True)
        
        # Run pose detection
        results = self.pose_model.predict(img, conf=self.pose_conf_th, verbose=False)

        with new_label_path.open("w") as f:
            for r in results:
                boxes = r.boxes.xywhn.cpu().numpy()
                boxes_confs = r.boxes.conf.cpu().numpy()
                kpts = r.keypoints.xyn.cpu().numpy()
                kpts_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones_like(kpts[:,:,0])

                for box, kpt, confs, box_conf in zip(boxes, kpts, kpts_conf, boxes_confs):
                    cx, cy, w, h = box

                    # Build YOLO line
                    label_parts = [str(self.pose_label_value), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

                    # Draw keypoints
                    points = []
                    for (x, y), v in zip(kpt, confs):
                        v_flag = 2 if v > 0.5 else (1 if v > 0 else 0)
                        label_parts += [f"{x:.6f}", f"{y:.6f}", str(v_flag)]

                        px, py = int(x * img_w), int(y * img_h)
                        points.append((px, py, v))

                        # if v > 0.5:
                        cv2.circle(img, (px, py), 4, (0,255,0), -1)

                    f.write(" ".join(label_parts) + "\n")
                    
                    
                    # Draw bounding box
                    if not self.task_coco_detection and not self.md_enable_coco2017:
                        x1 = int((cx - w/2)*img_w)
                        y1 = int((cy - h/2)*img_h)
                        x2 = int((cx + w/2)*img_w)
                        y2 = int((cy + h/2)*img_h)
                        cv2.rectangle(image if image is not None else img,(x1,y1),(x2,y2),(255,0,0),2)
                    
                    # Draw skeleton
                    for i, (j1, j2) in enumerate(self.SKELETON):
                        if j1 < len(points) and j2 < len(points):
                            x1, y1, v1 = points[j1]
                            x2, y2, v2 = points[j2]
                            # if v1 > 0.5 and v2 > 0.5:  # draw only visible joints
                            color = self.COLORS[i % len(self.COLORS)]
                            cv2.line(image if image is not None else img, (x1, y1), (x2, y2), color, 2)
                            
                    # Draw conf
                    conf = f"{box_conf:.2f}"
                    x = int((cx - w/2)*img_w)
                    y = int((cy - h/2)*img_h)
                    cv2.putText(image if image is not None else img,
                                conf, (x, y-35),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)

        if self.show_result_im:
            im_h,im_w = self.save_result_im_resolution[:2]
            vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Pose detection",vis_img)
            cv2.waitKey(self.waitkey)
        
        if self.save_result_im:
            # Save result image
            im_h,im_w = self.save_result_im_resolution[:2]
            save_path = self.skeleton_save_dir / Path(img_path).name
            # resize to (h=384, w=640) before saving
            vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(save_path), vis_img)
            # print(f"Saved visualization to {save_path}")

        return img



