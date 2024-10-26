from dataclasses import dataclass


@dataclass
class LogDatasetConfig:
    PATH_TO_DATA = "data/data.yaml"
    PATH_TO_MODEL = None
    run_name = "testing"
    name = f"{run_name}_stuff"
    epochs = 1
    batch_size = 10
    dropout = 0
    lr0 = 1e-3
    lrf = 1e-4
    imgsz = 128
    cos_lr = True
    optimizer = "AdamW"
    save_period = 10
    cache = False
    seed = 36

    project_name = "1y33/test"
    experiment_name = run_name
    tags = ['Ultralytics', 'Training']
