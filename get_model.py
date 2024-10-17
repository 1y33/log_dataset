import os.path
from contextlib import nullcontext

import ultralytics
import torch
from PIL import Image
import cv2
from sympy.codegen.ast import continue_



class Model:
    def __init__(self,weights_path=None):
        self.device = None
        self._device_dtype()

        self.weights_path = weights_path
        self.model_name = "yolov5nu"
        self.dataset_yaml = None
        self._get_model()

    def _device_dtype(self):
        if torch.cuda.is_available():
             self.device = "cuda"
        else:
             self.device = "cpu"

    def _get_model(self):
        if self.weights_path is None:
            self.model = ultralytics.YOLO(self.model_name)
        else:
            self.load_model(self.weights_path)

        return self.model.to(self.device)

    def get_dataset(self,path):
        # path -> path to dataset.yaml
        self.dataset_yaml = path

    def train_model(self,run_name,epochs,batch_size,dropout=0.2,cos_lr=False,name=None,lr0=0.01,lrf=0.005,workers=8,seed=36,imgsz=320,optimizer="AdamW"):
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
                    )

    def load_model(self,path):
        # load the best path for the model saved :
        self.model = ultralytics.YOLO(path)

    def detect_image(self,path,labels=False):
        results = self.model(path)
        print(self.count_image(results))
        for result in results:
            result.show(labels=labels)

    def count_image(self,results):
        return nullcontext

    def save_model(self,path):
        self.model.save(path)
