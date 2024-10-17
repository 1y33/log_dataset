import get_model
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt

def inference_slicing(image_path,model_path,imgsz,overlap_wh,iou_trashold,overlap_filter="NMS",show=True):
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
    detections = detections[detections.confidence > 0.3]

    bbox_annotator = sv.BoxAnnotator()
    annotated_image = bbox_annotator.annotate(scene=image.copy(), detections=detections)

    if show:
        plt.figure(figsize=(20, 20))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    print(f"Number of detections: {len(detections.confidence)}")


# model_path = "runs/detect/yoloV5-refine--EP:20-BS:16+1e-05-cos_lr-False-drp-0.4+AdamW/weights/best.pt"
# image_path = "images/test_image.jpeg"
#
# inference_slicing(image_path,model_path,imgsz=256,iou_trashold=0.5,overlap_wh=(0.3,0.3),overlap_filter="NMS")


