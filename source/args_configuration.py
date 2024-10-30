from dataclasses import dataclass


@dataclass
class LogDatasetConfig:
    PATH_TO_DATA = "../data/data.yaml"
    PATH_TO_MODEL = None
    run_name = "testing"
    name = f"{run_name}_stuff"
    epochs = 1
    batch_size = 0.80
    dropout = 0
    lr0 = 1e-3
    lrf = 1e-4
    imgsz = 1024
    cos_lr = True
    optimizer = "AdamW"
    save_period = 10
    cache = False
    seed = 36

    project_name = "1y33/training"
    experiment_name = run_name
    tags = ['Ultralytics', 'Training']

class COCOConfing:
    PATH_TO_DATA = "./coco.yaml"
    PATH_TO_MODEL = None
    run_name = "training_coco"
    name = f"{run_name}_"
    epochs = 1
    batch_size = 0.70
    dropout = 0.2
    lr0 = 1e-3
    lrf = 1e-5
    imgsz = 640
    cos_lr = True
    optimizer = "AdamW"
    save_period = 10
    cache = False
    seed = 36

    project_name = "1y33/training"
    experiment_name = run_name
    tags = [ 'Training','CoCo']

class VisDroneConfig:
    PATH_TO_DATA = "./VisDrone.yaml"
    PATH_TO_MODEL = None
    run_name = "training_visdrone"
    name = f"{run_name}_"
    epochs = 10
    batch_size = 0.70
    dropout = 0.1
    lr0 = 1e-3
    lrf = 1e-4
    imgsz = 256
    cos_lr = True
    optimizer = "AdamW"
    save_period = 10
    cache = False
    seed = 36

    project_name = "1y33/training"
    experiment_name = run_name
    tags = [ 'Training','VisDrone']