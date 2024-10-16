import get_model
import torch
import hyper_params_search as h

# def gmodel(path):
#     m = get_model.Model(path)
#     # path = "data/data.yaml"
#     # m.get_dataset(path)
#     return m
#
# def main():
#     m = gmodel("models/best.pt")
#
#     # _ = m._train_model(20,64)
#     # m.model.save("models/best.pt")
#
#     path_image = "test_image.jpg"
#     m.detect_image(path_image,labels=False)


def main():
    # h.hyper_params_training("data/data.yaml")
    # m = get_model.Model("runs/detect/yoloV8---EP:100-BS:16+0.0001/weights/best.pt")
    m = get_model.Model("yolov8n.pt")
    m.get_dataset("data/data.yaml")
    m.train_model("yoloV8-",
                   epochs=100,
                   batch_size=16,
                   dropout=0.2,
                   lr0=1e-3,
                   lrf=1e-4,
                   imgsz=640,
                   cos_lr =True,
                   optimizer="SGD",
                   )

    # m = get_model.Model("runs/detect/yoloV8---EP:100-BS:16+0.005-cos_lr-True-drp-0/weights/best.pt")
    m.detect_image("test_image.jpeg")
    results = m.model("test_image.jpeg")
    print("Number of woods: ", len(results[0]))

main()
