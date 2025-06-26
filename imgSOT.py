from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")  

# Load image
image_path = "/Users/aryan/Desktop/plane.png" 
image = cv2.imread(image_path)


# Run detection
results = model(image)[0]

# Draw boxes and labels
for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = box.conf[0].item()
    class_id = int(box.cls[0].item())
    class_name = model.names[class_id]

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = f"{class_name} {conf:.2f}"
    cv2.putText(image, label, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show result
cv2.imshow("YOLOv8 Image Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()