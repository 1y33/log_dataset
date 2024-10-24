import get_model
import torch
import hyper_params_search as h
import argparse
import inference

def gmodel(model_path):
    '''
    Function to get the model :D
    :param model_path: path to model weights
    :return: returns the model class
    '''
    return get_model.Model(model_path)

def train_model(m, args):
    '''
    Function to train the model :D
    :param m: model to train
    :param args: params for training
    '''
    m.get_dataset(args.data_path)
    m.train_model(
        "new_data/yolo5-refine",
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        lr0=args.lr0,
        lrf=args.lrf,
        imgsz=args.imgsz,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
    )

def infernce_model(m,args,slicing_mode=False):
    '''
    :param m: model used for inference
    :param args: args from CLI
    :param slicing_mode: slicing_mode. If false full image load.If true, image is sliced
    :return: results = number of detections
    '''
    if slicing_mode is False:
        results = m.detect_image(args.image_path)
        results = len(results[0])
    else:
        results = inference.inference_slicing(args.image_path,args.imgsz,iou_trashold=args.iou_trashold)

    return results

def main():
    parser = argparse.ArgumentParser(description="Simple CLI for this model: Training + Inference")

    # Required arguments
    parser.add_argument("-p", "--path", type=str, required=False, help="Path to dataset YAML file")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model weights")

    # Flags for training and inference
    parser.add_argument("-t", "--train", action='store_true', help="Train the model")
    parser.add_argument("-i", "--inference", action='store_true', help="Perform inference on an image")

    # Inference specific argument
    parser.add_argument("-i_img", "--inference_image", type=str, help="Path to the image for inference")

    # Training arguments
    parser.add_argument("-ep", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("-dr", "--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("-lr0", "--lr0", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("-lrf", "--lrf", type=float, default=1e-3, help="Final learning rate")
    parser.add_argument("-imgsz", "--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("-cos_lr", "--cos_lr", action='store_true',default=True, help="Use cosine learning rate")
    parser.add_argument("-opt", "--optimizer", type=str, default="AdamW", help="Optimizer type")

    # Slicing arguments (if applicable)
    parser.add_argument("-iou", "--iou_trashold", type=float, default=0.5, help="IOU threshold for inference slicing")

    args = parser.parse_args()
    m = gmodel(args.model)

    if args.train:
        m.train(m,args)

    if args.inference:
        if not args.inference_image:
            print("No inference image to load")
        else:
            results = infernce_model(m,args,slicing_mode=False)
            print("Number of objects in image: ", results)


main()
