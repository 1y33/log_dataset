import get_model
from args_configuration import COCOConfing,VisDroneConfig

def train():
    COCO_args = COCOConfing
    VisDrone = VisDroneConfig

    main_model = "yolov5n-p6.yaml"
    m = get_model.Model(main_model)

    # Training for COCO:
    # m.add_callbacks(COCO_args.project_name, COCO_args.experiment_name, COCO_args.tags)
    # m.get_dataset(COCO_args.PATH_TO_DATA)
    # m.train_model(COCO_args)


    # Training for VisDrone
    # m.add_callbacks(VOC_args.project_name, VOC_args.experiment_name, VOC_args.tags)
    m.get_dataset(VisDrone.PATH_TO_DATA)
    m.train_model(VisDrone)


train()
