import yaml

class Args:
    def __init__(self, config_file: str):
        """
        Load configuration values from a YAML file and assign them
        as attributes of the Args object.

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file.
        """
        # --- Load YAML config file ---
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # --- Task settings ---
        self.task_coco_detection  = config['GENERATE_LABEL_TASK']['TASK_COCO_DETECTION']
        self.task_face_detection  = config['GENERATE_LABEL_TASK']['TASK_FACE_DETECTION']
        self.task_pose_detection  = config['GENERATE_LABEL_TASK']['TASK_POSE_DETECTION']
        self.task_multi_detection = config['GENERATE_LABEL_TASK']['TASK_MULTI_DETECTION']

        # --- Multi-detection label enable flags ---
        self.md_enable_coco2017 = config['MD_ENABLE_LABEL']['COCO2017']
        self.md_enable_face     = config['MD_ENABLE_LABEL']['FACE']
        self.md_enable_pose     = config['MD_ENABLE_LABEL']['POSE']
        
        # -- Label-confidence threshold ---
        self.coco2017_conf_th = config['CONFIDENCE_THRESHOLD']['COCO2017']
        self.face_conf_th     = config['CONFIDENCE_THRESHOLD']['FACE']
        self.pose_conf_th     = config['CONFIDENCE_THRESHOLD']['POSE']
        
        # --- Label value ---
        self.face_label_value = config['LABEL_VALUE']['FACE']
        self.pose_label_value = config['LABEL_VALUE']['POSE']
        
        ## --- Model ---
        self.detect_model = config['MODEL']['DETECT_MODEL']
        self.face_model = config['MODEL']['FACE_MODEL']
        self.pose_model = config['MODEL']['POSE_MODEL']
        
        ## --- Visualize ---
        self.show_result_im = config['VISUALIZE']['SHOW_RESULT_IM']

        # --- Dataset settings ---
        self.data_num                = config['DATA']['DATA_NUM']
        self.data_dir                = config['DATA']['DATA_DIR']
        self.data_type               = config['DATA']['DATA_TYPE']
        self.data_img_dir            = config['DATA']['DATA_IMG_DIR']
        self.data_save_txt_dir       = config['DATA']['DATA_SAVE_DETECT_TXT_DIR']
        self.data_pose_save_txt_dir  = config['DATA']['DATA_SAVE_POSE_TXT_DIR']
        
        
        # --- labelmapping ---
        self.enable_mapping = config['COCO2017_MAPPING']['ENABLE']
        self.label_mapping = config['COCO2017_MAPPING']['LABEL_MAPPING']
        self.mapping_input_label_dir = config['COCO2017_MAPPING']['INPUT_LABEL_DIR']
        self.mapping_output_label_dir = config['COCO2017_MAPPING']['OUTPUT_LABEL_DIR']
        self.mapping_label_name = config['COCO2017_MAPPING']['MAPPING_LABEL_NAME']
        
        # ---Save ---
        self.save_result_im = config['SAVE']['RESULT_IMAGE']
        self.save_result_im_resolution = config['SAVE']['RESULT_IM_RESOLUTION']
        self.save_mapping_result_im = config['SAVE']['RESULT_MAPPING_IMAGE']