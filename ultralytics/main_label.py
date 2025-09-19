from tasks.multiDetection import MultiDetectTask
from tasks.multiDetection import COCODetection
from tasks.multiDetection import FaceDetection
from tasks.multiDetection import PoseDetection
from config.args import Args


if __name__=="__main__":
    
    args = Args(config_file='config/config.yaml')
    
    if args.task_multi_detection:
        MD = MultiDetectTask(args)
        MD.show_config(args)
        MD.Auto_labeling_tools()
        
    if args.task_coco_detection:
        CD = COCODetection(args)
        CD.show_config(args)
        CD.Auto_labeling_tools()
        
    if args.task_face_detection:
        FD = FaceDetection(args)
        FD.show_config(args)
        FD.Auto_labeling_tools()
        
    if args.task_pose_detection:
        PD = PoseDetection(args)
        PD.show_config(args)
        PD.Auto_labeling_tools()