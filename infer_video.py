from transformers import (
    SegformerFeatureExtractor, 
    SegformerForSemanticSegmentation
)
from config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import (
    draw_segmentation_map, 
    image_overlay,
    predict
)

import argparse
import cv2
import time
import os

import pandas as pd
import numpy as np
import openpyxl

# from extract_point_locate import find_coordinates
from extract_point_locate import find_area_between_points
from extract_features import find_area_between_points_optimized

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input video',
    default='input/inference_data/videos/video_1.mov'
)
parser.add_argument(
    '--device',
    default='cpu:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='outputs/model_iou'
)
args = parser.parse_args()

out_dir = 'outputs/inference_results_video'
os.makedirs(out_dir, exist_ok=True)

extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

cap = cv2.VideoCapture(args.input)
if args.imgsz is None:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
else: 
    frame_width = args.imgsz[0]
    frame_height = args.imgsz[1]
vid_fps = int(cap.get(5))
save_name = args.input.split(os.path.sep)[-1].split('.')[0]
# Define codec and create VideoWriter object.
out = cv2.VideoWriter(f"{out_dir}/{save_name}.mp4", 
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                    (frame_width, frame_height))

frame_count = 0
total_fps = 0
while cap.isOpened:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        image = frame
        if args.imgsz is not None:
            image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # Get labels.
        start_time = time.time()
        labels = predict(model, extractor, image, args.device)
        find_area_between_points_optimized(labels)
        # toado = find_coordinates(labels)
        # print(toado)
        # Open the file for appending and write the labels to the file.
        # print(labels)
        # Mở file 'labels_data.txt' để ghi dữ liệu
        # Chuyển tensor thành numpy array
        # labels_np = labels.cpu().numpy()  # Chuyển tensor về CPU và chuyển thành numpy array

        # # Tạo DataFrame từ numpy array
        # df = pd.DataFrame(labels_np)

        # # Ghi DataFrame vào file Excel
        # df.to_excel('/labels_data.xlsx', index=False)


        end_time = time.time()

        fps = 1 / (end_time - start_time)
        total_fps += fps
        
        # Get segmentation map.
        seg_map = draw_segmentation_map(
            labels.cpu(), LABEL_COLORS_LIST
        )
        outputs = image_overlay(image, seg_map)
        cv2.putText(
            outputs,
            f"{fps:.1f} FPS",
            (15, 35),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        out.write(outputs)
        cv2.imshow('Image', outputs)
        # Press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")