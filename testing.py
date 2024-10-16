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
    # m = get_model.Model()
    # m.get_dataset("data/data.yaml")
    # m.train_model("max_training",
    #               epochs=50,
    #               batch_size=64,
    #               dropout=0.3,
    #               )
    # m.save_model("models/big_run.pt")

    m = get_model.Model("runs/detect/max_training--EP:50-BS:64+0.0052/weights/best.pt")
    m.detect_image("test_image.jpg")

main()