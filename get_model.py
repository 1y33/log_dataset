import os.path

import ultralytics
import torch
import callback


def count_image(results):
    return len(results[0])


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

    def train_model(self,args):
        if args.name is None:
            name = f"{args.run_name}--EP:{args.epochs}-BS:{args.batch_size}+{args.lrf}-cos_lr-{args.cos_lr}-drp-{args.dropout}+{args.optimizer}"
        if args.cache is None:
            args.cache = False
        if args.save_period is None:
            args.save_period = 10
        if not os.path.isdir(name):
            results = self.model.train(
                    data = self.dataset_yaml,
                    batch= args.batch_size,
                    epochs = args.epochs,
                    dropout = args.dropout,
                    lr0 = args.lr0,
                    lrf = args.lrf,
                    name = args.name,
                    imgsz = args.imgsz,
                    cos_lr=args.cos_lr,
                    seed = args.seed,
                    optimizer=args.optimizer,
                    save_period = args.save_period,
                    cache = args.cache
                    )

    def load_model(self,path):
        # load the best path for the model saved :
        self.model = ultralytics.YOLO(path)

    def detect_image(self,path,labels=False):
        results = self.model(path)
        print(count_image(results))
        for result in results:
            result.show(labels=labels)
        return results

    def save_model(self,path):
        self.model.save(path)
