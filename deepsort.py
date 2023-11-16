from ultralytics import YOLO

import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker


#define variable for line detection
linea_1 = (420,250)
linea_2 = (520,460)
lineb_1 = (600, 300 + 3 * 50)
lineb_2 = (720, 510 + 3 * 50)
total_count_good = 0
i = 0
counter, fps, elapsed = 0, 0, 0
counter_good = []
start_time = time.perf_counter()
# Define the video path
video_path = '/home/wetu/bmo/My_work/smart_sense/dataset/video/video_0.MP4'
cap = cv2.VideoCapture(video_path)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
frames = []
counter_good = []
counter_bad = []
unique_track_ids = set()
# Load a model
model = YOLO("/home/wetu/bmo/My_work/smart_sense/yolov8/best.pt")  # load a pretrained model (recommended for training)
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

def is_touching_line(line_start, line_end, object_center):
    if line_start[0] <= object_center[0] <= line_end[0]:
        expected_y = line_start[1] + ((line_end[1] - line_start[1]) / (line_end[0] - line_start[0])) * (object_center[0] - line_start[0])
        if abs(object_center[1] - expected_y) <= 10:  # Increased tolerance value for better detection
            return True
    return False  # 

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = og_frame.copy()

        model = YOLO("/home/wetu/bmo/My_work/smart_sense/yolov8/best.pt")  # load a pretrained model (recommended for training)

        results = model(frame, device=0, classes=0, conf=0.8)
        class_names = ['shoes']
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            probs = result.probs  # Class probabilities for classification outputs
            cls = boxes.cls.tolist()  # Convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh  # box with xywh format, (N, 4)
            for class_index in cls:
                class_name = class_names[int(class_index)]
                #print("Class:", class_name)

        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)
        
        tracks = tracker.update(bboxes_xywh, conf, og_frame)
        
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height
            color = (255, 0, 0)  # Box color
            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                    #get center of bbox
            x_center = int(((x1) + (x2)) / 2.0)
            y_center = int(((y1) + (y2)) / 2.0)
            '''
            bbox[0] = x1
            bbox[1] = y1
            bbox [2] = x1+w
            bbox [3] = y1+h
            '''
            center = (x_center, y_center)
            #center_y = int(((bbox[1])+(bbox[3]))/2)
            #center_x = int(bbox[0] + (bbox[2] - bbox[0]) / 2)
            #enter = (center_x,center_y)
            # Add the track_id to the set of unique track IDs
            unique_track_ids.add(track_id)
            count = is_touching_line(linea_1, linea_2,center)

        #display center of bbox
            cv2.circle(og_frame, (x_center, y_center), 5, color, -1)
            #creat line
            cv2.line(og_frame,linea_1,linea_2,(0,0,255),3)
            cv2.line(og_frame,lineb_1,lineb_2,(0,0,255),3)
            if is_touching_line(linea_1, linea_2, center):
                if class_name == 'shoes':
                    track_id = int(track.track_id)
                    if track_id not in counter_good:  # Check if track_id is not already in the list
                        counter_good.append(int(track.track_id))
            if is_touching_line(linea_1, linea_2, center):
                if class_name == 'shoes':
                    track_id = int(track.track_id)
                    if track_id not in counter_bad:  # Check if track_id is not already in the list
                        counter_bad.append(int(track.track_id))

            total_count_bad = len(set(counter_good))
            cv2.putText(frame, "Good shoes: " + str(total_count_good), (0, 80), 0, 1, (0, 0, 255), 2)
            total_count_good = len(set(counter_bad))
            cv2.putText(frame, "Bad shoes: " + str(total_count_bad), (0, 80), 0, 1, (0, 0, 255), 2)


        # Update FPS and place on frame
        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        # Append the frame to the list
        frames.append(og_frame)

        # Write the frame to the output video file

        
        cv2.putText(og_frame, "Good shoes: " + str(total_count_good), (0, 80), 0, 1, (0, 0, 255), 2)
        #cv2.putText(og_frame, "Bad shoes: " + str(total_count_bad), (0,130), 0, 1, (0,0,255), 2)
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
    #Show the frame
    
        #cv2.imshow("Video", og_frame)
        cv2.imshow("Video", cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
             break

cap.release()
out.release()
cv2.destroyAllWindows()