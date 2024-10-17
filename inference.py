from pycparser.ply.yacc import restart

import get_model
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt

model_path = "runs/detect/yoloV5-refine--EP:20-BS:16+1e-05-cos_lr-False-drp-0.4+AdamW/weights/best.pt"
image_path = "images/test_image.jpeg"
#
# inference_slicing(image_path,model_path,imgsz=256,iou_trashold=0.5,overlap_wh=(0.3,0.3),overlap_filter="NMS")


def inference_slicing(image_path,model_path,imgsz,overlap_wh,iou_trashold,confidence=0.3,overlap_filter="NMS",show=True,return_count=False):
    if overlap_filter == "NMS":
        overlap_filter = sv.OverlapFilter.NON_MAX_SUPPRESSION
    else:
        if overlap_filter == "NMN":
            overlap_filter = sv.OverlapFilter.NON_MAX_MERGE
        else:
            overlap_filter = sv.OverlapFilter.NONE

    m = get_model.Model(model_path)
    image = cv2.imread(image_path)



    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = m.model(image_slice, imgsz=imgsz)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(callback=callback,
                                slice_wh=(imgsz, imgsz),
                                overlap_ratio_wh=None,
                                overlap_wh=overlap_wh,
                                iou_threshold=iou_trashold,
                                overlap_filter=overlap_filter,
                                )

    detections = slicer(image)
    detections = detections[detections.confidence > confidence]

    bbox_annotator = sv.BoxAnnotator()
    annotated_image = bbox_annotator.annotate(scene=image.copy(), detections=detections)

    if show:
        plt.figure(figsize=(20, 20))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    print(f"Number of detections: {len(detections.confidence)}")

    if return_count:
        return int(len(detections.confidence))

def best_inference_params(model_path,image_path,target_count):
    best_count = 0

    imgsz_list = [256,320]
    iou_trashold_list = [0.45,0.5]
    confidence_list = [0.5,0.6]
    overlap_wh_list = [0.4,0.5]
    overlap_filter_list =["NMS","NMG"]

    values = []
    for imgsz in imgsz_list:
        for iou_trashold in iou_trashold_list:
            for confidence in confidence_list:
                for overlap_filter in overlap_filter_list:
                    for overlap_wh in overlap_wh_list:
                        value = inference_slicing(
                            model_path=model_path,
                            image_path=image_path,
                            imgsz=imgsz,
                            overlap_wh=(overlap_wh,overlap_wh),
                            iou_trashold=iou_trashold,
                            confidence=confidence,
                            overlap_filter=overlap_filter,
                            show=False,
                            return_count=True
                        )
                        value_dict = {
                            "value": value,
                            "imgsz": imgsz,
                            "iou_trashold": iou_trashold,
                            "confidence": confidence,
                            "overlap_filter": overlap_filter,
                            "overlap_wh": (overlap_wh,overlap_wh),
                        }
                        values.append(value_dict)

    closest_value = min(values, key=lambda x: abs(x["value"] - target_count))["value"]
    result = [value_dict for value_dict in values if value_dict["value"] == closest_value]

    print("Value: " ,result["value"])
    print("Imgsz: " ,result["imgsz"])
    print("Iou Trashold: ",result["iou_trashold"])
    print("Confidence: ",result["confidence"])
    print("Overlap Filter: ",result["overlap_filter"])
    print("Overlap_Wh: ",result["overlap_wh"])

    return result


model_path = "runs/detect/yoloV5-refine--EP:20-BS:16+1e-05-cos_lr-False-drp-0.4+AdamW/weights/best.pt"
image_path = "images/test_image.jpeg"

result = best_inference_params(model_path,image_path,104)



