import get_model
from args_configuration import COCOConfing,VOCConfing

def train():
    COCO_args = COCOConfing
    VOC_args = VOCConfing

    main_model = "./yolov5n-p6.yaml"
    m = get_model.Model(main_model)

    # Training for COCO:
    m.add_callbacks(COCO_args.project_name, COCO_args.experiment_name, COCO_args.tags)
    m.get_dataset(COCO_args.PATH_TO_DATA)
    m.train_model(COCO_args)

    # Training for VOC
    m.add_callbacks(VOC_args.project_name, VOC_args.experiment_name, VOC_args.tags)
    m.get_dataset(VOC_args.PATH_TO_DATA)
    m.train_model(VOC_args)


train()
