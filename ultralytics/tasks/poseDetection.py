import cv2
from engine.dataset import BaseDataset
from pathlib import Path


class PoseDetection(BaseDataset):
    
    def __init__(self, args):
        super().__init__(args)
        self.skeleton_save_dir = Path(self.data_dir) / "labels" / "lane" / "points" / "visual_points"  # save dir for drawn images
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

    def Save_YOLO_txt_Labels(self, img_path, image):
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
                        v_flag = 2 if v > 0.5 else 1
                        label_parts += [f"{x:.6f}", f"{y:.6f}", str(v_flag)]

                        px, py = int(x * img_w), int(y * img_h)
                        points.append((px, py, v))

                        # if v > 0.5:
                        cv2.circle(img, (px, py), 4, (0,255,0), -1)

                    f.write(" ".join(label_parts) + "\n")
                    # Draw bounding box
                    # x1 = int(cx - w/2)
                    # y1 = int(cy - h/2)
                    # x2 = int(cx + w/2)
                    # y2 = int(cy + h/2)
                    # cv2.rectangle(image if image is not None else img,(x1,y1),(x2,y2),(255,0,0),2)
                    
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

        if self.save_result_im:
            # Save result image
            im_h,im_w = self.save_result_im_resolution[:2]
            save_path = self.skeleton_save_dir / Path(img_path).name
            # resize to (h=384, w=640) before saving
            vis_img = cv2.resize(image if image is not None else img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(save_path), vis_img)
            # print(f"Saved visualization to {save_path}")

        return img



