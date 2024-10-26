import os.path

import ultralytics
import torch
from PIL import Image
import cv2
import callback


class Model:
    def __init__(self,config=None):
        self.device = None
        self.model = None
        self._device_dtype()

        self.config = config
        if config is not None:
            self.config = config
        else:
            self.config = "yolov5nu"

        self.dataset_yaml = None
        self._get_model()

    def add_callbacks(self, project_name, experiment_name, tags):
        callbacks = callback.create_callbacks(project_name,experiment_name,tags)
        for name, func in callbacks.items():
            self.model.add_callback(name, func)

    def _device_dtype(self):
        if torch.cuda.is_available():
             self.device = "cuda"
        else:
             self.device = "cpu"

    def _get_model(self):
        self.model = ultralytics.YOLO(self.config)
        ## this includes all cases no need to hard recode it
        return self.model.to(self.device)

    def get_dataset(self,path):
        # path -> path to dataset.yaml
        self.dataset_yaml = path

    def train_model(self,run_name,epochs,batch_size,dropout=0.2,cos_lr=False,name=None,lr0=0.01,lrf=0.005,workers=8,seed=36,imgsz=320,optimizer="AdamW",save_period=10,cache=True):
        if name is None:
            name = f"{run_name}--EP:{epochs}-BS:{batch_size}+{lrf}-cos_lr-{cos_lr}-drp-{dropout}+{optimizer}"

        if not os.path.isdir(name):
            results = self.model.train(
                    data = self.dataset_yaml,
                    batch= batch_size,
                    epochs = epochs,
                    dropout = dropout,
                    lr0 = lr0,
                    lrf = lrf,
                    name = name,
                    imgsz = imgsz,
                    cos_lr=cos_lr,
                    seed = 36,
                    optimizer=optimizer,
                    save_period = save_period,
                    cache = cache
                    )

    def load_model(self,path):
        # load the best path for the model saved :
        self.model = ultralytics.YOLO(path,config=self.config)

    def detect_image(self,path,labels=False):
        results = self.model(path)
        print(self.count_image(results))
        for result in results:
            result.show(labels=labels)
        return results

    def count_image(self,results):
        return len(results[0])

    def save_model(self,path):
        self.model.save(path)
