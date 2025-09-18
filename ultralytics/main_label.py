from tasks.multiDetection import MultiDetectTask
from config.args import Args


if __name__=="__main__":
    
    args = Args(config_file='config/config.yaml')
    
    if args.task_multi_detection:
        MD = MultiDetectTask(args)
        MD.Get_Multi_Detection_YOLO_Txt_Label()