from ultralytics import YOLO

def main():
    path = "runs/detect/imgsze-640-training--EP:100-BS:16+0.0001/weights/best.pt"
    # m.detect_image("images/test_image.jpeg")
    model = YOLO(path)
    model.export(format="edgetpu", imgsz=640);

main()

#requirements: Ultralytics requirements ['sng4onnx>=1.0.1', 'onnx_graphsurgeon>=0.3.26', 'onnx>=1.12.0', 'onnx2tf>1.17.5,<=1.22.3', 'onnxslim>=0.1.31', 'tflite_support', 'onnxruntime-gpu'] not found, attempting AutoUpdate...
