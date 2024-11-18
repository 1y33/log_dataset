from args_configuration import COCOConfing,VisDroneConfig
from ultralytics import YOLO
import callback


def train():
    VisDrone = VisDroneConfig

    model = YOLO("yolov5n-p6.yaml")
    callbacks = callback.create_callbacks(VisDrone.project_name, VisDrone.experiment_name, VisDrone.tags)
    for name, func in callbacks.items():
        model.add_callback(name, func)

    _ = model.train(
                data="VisDrone.yaml",
                batch=VisDrone.batch_size,
                epochs=VisDrone.epochs,
                dropout=VisDrone.dropout,
                lr0=VisDrone.lr0,
                lrf=VisDrone.lrf,
                name=VisDrone.name,
                imgsz=VisDrone.imgsz,
                cos_lr=VisDrone.cos_lr,
                seed=VisDrone.seed,
                optimizer=VisDrone.optimizer,
                save_period=VisDrone.save_period,
                cache=VisDrone.cache,
                workers = VisDrone.workers
    )

train()
