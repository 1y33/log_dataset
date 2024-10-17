model_path = "runs/detect/yoloV5-refine--EP:20-BS:16+1e-05-cos_lr-False-drp-0.4+AdamW/weights/best.pt"
image_path = "images/test_image.jpeg"

import get_model
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt

m = get_model.Model(model_path)
image = cv2.imread(image_path)

def callback(image_slice:np.ndarray) -> sv.Detections:
    result = m.model(image_slice,imgsz=320)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(callback=callback,
                            slice_wh=(320, 320),
                            overlap_ratio_wh=None,
                            overlap_wh=(0.5, 0.5),
                            iou_threshold=0.4,
                            overlap_filter=sv.OverlapFilter.NON_MAX_MERGE ,
                            )
detections = slicer(image)
detections = detections[detections.confidence > 0.3]

bbox_annotator = sv.BoxAnnotator()
annotated_image = bbox_annotator.annotate(scene=image.copy(), detections=detections)

plt.figure(figsize=(20, 20))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print the number of detections
print(f"Number of detections: {len(detections.confidence)}")
