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
    path = "runs/detect/yoloV5-refine--EP:20-BS:16+1e-05-cos_lr-False-drp-0.4+AdamW/weights/best.pt"
    # m = get_model.Model(path)
    # m.get_dataset("data/data.yaml")
    # m.train_model("yoloV5-refine",
    #                epochs=20,
    #                batch_size=16,
    #                dropout=0.4,
    #                lr0=1e-4,
    #                lrf=1e-5,
    #                imgsz=640,
    #                cos_lr =False,
    #                optimizer="AdamW",
    #                )

    m = get_model.Model(path)
    m.detect_image("test_image.jpeg")
    results = m.model("test_image.jpeg")
    print("Number of woods: ", len(results[0]))

main()
