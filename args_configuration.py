from dataclasses import dataclass

@dataclass
class LogDatasetConfig:
    PATH_TO_DATA = "data/data.yaml"
    PATH_TO_MODEL =None
    run_name = "testing"
    name =f"{run_name}_"
    epochs = 30
    batch_size = 10
    dropout = 0
    lr0 =1e-3
    lrf=1e-4
    imgsz = 640
    cos_lr = True
    optimizer = "AdamW"
    save_period = 10
    cache = False

    project_name = "1y33/test"
    experiment_name = run_name
    tags = ['Ultralytics', 'Training']
