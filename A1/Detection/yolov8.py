from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    results = model.val(data="coco.yaml", imgsz=640, workers=0)