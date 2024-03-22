from ultralytics import YOLO

model = YOLO("YOLOv8m-seg.pt")

if __name__ == "__main__":
    model.train(data="config.yaml", epochs=100, hsv_h=0.0, hsv_s=0.0, scale=0.0, hsv_v=0.0, max_det=2000, mosaic=0.0)

