# %%
#import os
#from dotenv import load_dotenv
#load_dotenv()

# %%
# Check TensorFlow version and GPU availability
#import tensorflow as tf
#print("TensorFlow version:", tf.__version__)
#print("GPU Available:", tf.config.list_physical_devices('GPU'))

# %%
#os.environ["ULTRALYTICS_API_KEY"]=os.getenv("ULTRALYTICS_API_KEY")

# %%
##!pip install 'faster-coco-eval'

# %%
from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2 as cv

# Load YOLO model
model = YOLO("yolov8n.pt")

# %%
cap=cv.VideoCapture("/Users/aryan/Desktop/DRDO/3055764-uhd_3840_2160_24fps.mp4")


# %%
# Train the model
#from ultralytics import YOLO, checks, hub
#checks()

#hub.login('80b326ea5b905795387a78867cccc95aff5721149d')

#model = YOLO('https://hub.ultralytics.com/models/arSDJ5FiB9tzQH7dOOri')
#results = model.train()

# %%
## This is to get the size of the screen
import tkinter as tk
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# %%
while(True):
    ret,frame=cap.read()

    if(not ret):
        break

    results = model(frame)[0]
    print(type(results))

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]

        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv.putText(frame, label, (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        resized_frame = cv.resize(frame, (screen_width, screen_height))

    cv.imshow("output",resized_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        
        break
cv.destroyAllWindows()

