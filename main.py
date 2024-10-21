# import torch
# import cvzone
# import cv2
# from torchvision import transforms
# import torchvision

# # Load the pre-trained Faster R-CNN model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()

# # Load class names from the 'classes.txt' file
# classnames = []
# with open('classes.txt', 'r') as f:
#     classnames = f.read().splitlines()

# # Define vehicle categories
# vehicle_classes = {
#     "car": ["car"],
#     "truck": ["truck"],
#     "bike": ["bicycle", "motorcycle"],
#     "ambulance": ["ambulance"],
#     "other_vehicles": ["bus", "train"]  # Add other vehicle-related classes here if necessary
# }

# # Initialize counts for each vehicle category
# vehicle_count = {
#     "car": 0,
#     "truck": 0,
#     "bike": 0,
#     "ambulance": 0,
#     "other_vehicles": 0
# }

# # Read and resize the image
# image = cv2.imread('C:\\Users\\Dhanvanth S\\Documents\\ML\\Faster-RCNN-PYTORCH-main\\cat.png')
# image = cv2.resize(image, (640, 480))

# # Transform the image to tensor
# image_transform = transforms.ToTensor()
# img = image_transform(image)

# with torch.no_grad():
#     # Get predictions from the model
#     pred = model([img])

#     bbox, scores, labels = pred[0]['boxes'], pred[0]['scores'], pred[0]['labels']

#     # Filter out predictions with scores above the confidence threshold (0.70)
#     conf = torch.argwhere(scores > 0.70).shape[0]
    
#     for i in range(conf):
#         x, y, w, h = bbox[i].numpy().astype('int')
#         classname = labels[i].numpy().astype('int')
#         class_detected = classnames[classname]

#         # Categorize and count the detected vehicle types
#         if class_detected in vehicle_classes["car"]:
#             vehicle_count["car"] += 1
#         elif class_detected in vehicle_classes["truck"]:
#             vehicle_count["truck"] += 1
#         elif class_detected in vehicle_classes["bike"]:
#             vehicle_count["bike"] += 1
#         elif class_detected in vehicle_classes["ambulance"]:
#             vehicle_count["ambulance"] += 1
#         elif class_detected in vehicle_classes["other_vehicles"]:
#             vehicle_count["other_vehicles"] += 1

#         # Draw bounding boxes and labels on the image
#         cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 4)
#         cvzone.putTextRect(image, class_detected, [x + 8, y - 12], scale=2, border=1)

#     # Save the annotated image
#     cv2.imwrite('data1.png', image)

#     # Print the counts of each vehicle category
#     print("Vehicle Counts:")
#     print(f"Cars: {vehicle_count['car']}")
#     print(f"Trucks: {vehicle_count['truck']}")
#     print(f"Bikes: {vehicle_count['bike']}")
#     print(f"Ambulances: {vehicle_count['ambulance']}")
#     print(f"Other Vehicles: {vehicle_count['other_vehicles']}")

# # Display the image
# cv2.imshow('frame', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # import torch
# # import cvzone
# # import cv2
# # from torchvision import transforms
# # import torchvision
# # import numpy as np

# # # Use GPU if available for faster processing
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # Load the pre-trained Faster R-CNN model
# # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# # model = model.to(device)
# # model.eval()

# # # Load class names from the 'classes.txt' file
# # classnames = []
# # with open('classes.txt', 'r') as f:
# #     classnames = f.read().splitlines()

# # # Define vehicle categories
# # vehicle_classes = {
# #     "car": ["car"],
# #     "truck": ["truck"],
# #     "bike": ["bicycle", "motorcycle"],
# #     "ambulance": ["ambulance"],
# #     "other_vehicles": ["bus", "train"]  # Add other vehicle-related classes here if necessary
# # }

# # # Initialize counts for each vehicle category (only counting forward-moving vehicles)
# # vehicle_count = {
# #     "car": 0,
# #     "truck": 0,
# #     "bike": 0,
# #     "ambulance": 0,
# #     "other_vehicles": 0
# # }

# # # Initialize video capture
# # video_path = 'C:\Users\Dhanvanth S\Documents\ML\Faster-RCNN-PYTORCH-mainvideo_path.mp4'  # Replace with your correct video path
# # cap = cv2.VideoCapture(video_path)

# # # Check if video was opened successfully
# # if not cap.isOpened():
# #     print(f"Error: Could not open video file {video_path}")
# #     exit()

# # # Initialize the first frame for optical flow
# # ret, frame1 = cap.read()

# # if not ret or frame1 is None:
# #     print("Error: Could not read the first frame from the video.")
# #     exit()

# # prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# # # Continue with the rest of your processing...


# # # Preprocessing function for converting frames to tensors
# # def preprocess_frame(frame):
# #     image_resized = cv2.resize(frame, (640, 480))
# #     img_tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(device)
# #     return img_tensor, image_resized

# # # Function to detect forward-moving vehicles using optical flow
# # def is_moving_forward(optical_flow, bbox):
# #     x, y, w, h = bbox
# #     # Extract flow vectors within the bounding box
# #     flow_x = optical_flow[y:h, x:w, 0]
# #     flow_y = optical_flow[y:h, x:w, 1]

# #     # Compute the average flow vector direction
# #     avg_flow_x = np.mean(flow_x)
# #     avg_flow_y = np.mean(flow_y)

# #     # Decide if the motion is forward (you can tweak the threshold)
# #     if avg_flow_y < 0:  # Negative y-flow means moving upwards in image (forward)
# #         return True
# #     return False

# # while cap.isOpened():
# #     ret, frame2 = cap.read()
# #     if not ret:
# #         break

# #     # Convert the current frame to grayscale for optical flow
# #     next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# #     # Calculate the optical flow between the previous and current frame
# #     flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# #     # Preprocess the current frame for detection
# #     img_tensor, image_resized = preprocess_frame(frame2)

# #     with torch.no_grad():
# #         # Get predictions from the model
# #         pred = model([img_tensor[0]])

# #         # Extract bounding boxes, scores, and labels
# #         bbox, scores, labels = pred[0]['boxes'], pred[0]['scores'], pred[0]['labels']

# #         # Filter out predictions with scores above the confidence threshold (0.70)
# #         high_conf_indices = torch.where(scores > 0.70)[0]

# #         for i in high_conf_indices:
# #             x, y, w, h = bbox[i].cpu().numpy().astype('int')
# #             classname_idx = labels[i].cpu().numpy().astype('int')
# #             class_detected = classnames[classname_idx]

# #             # Check if the detected object is a vehicle and is moving forward
# #             if is_moving_forward(flow, (x, y, w, h)):
# #                 # Categorize and count the detected vehicle types moving forward
# #                 if class_detected in vehicle_classes["car"]:
# #                     vehicle_count["car"] += 1
# #                 elif class_detected in vehicle_classes["truck"]:
# #                     vehicle_count["truck"] += 1
# #                 elif class_detected in vehicle_classes["bike"]:
# #                     vehicle_count["bike"] += 1
# #                 elif class_detected in vehicle_classes["ambulance"]:
# #                     vehicle_count["ambulance"] += 1
# #                 elif class_detected in vehicle_classes["other_vehicles"]:
# #                     vehicle_count["other_vehicles"] += 1

# #                 # Draw bounding boxes and labels for forward-moving vehicles
# #                 cv2.rectangle(image_resized, (x, y), (w, h), (0, 255, 0), 2)
# #                 cvzone.putTextRect(image_resized, class_detected, [x + 8, y - 12], scale=2, border=2)

# #         # Display the frame with annotated forward-moving vehicles
# #         cv2.imshow('frame', image_resized)

# #         # Break the loop with 'q' key
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# #     # Update the previous frame to the current one for the next iteration
# #     prev_gray = next_gray

# # # Release video capture and close windows
# # cap.release()
# # cv2.destroyAllWindows()

# # # Print the counts of forward-moving vehicles
# # print("Vehicle Counts (Moving Forward):")
# # print(f"Cars: {vehicle_count['car']}")
# # print(f"Trucks: {vehicle_count['truck']}")
# # print(f"Bikes: {vehicle_count['bike']}")
# # print(f"Ambulances: {vehicle_count['ambulance']}")
# # print(f"Other Vehicles: {vehicle_count['other_vehicles']}")

# import torch
# import cv2
# from torchvision import transforms
# import torchvision
# from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# # Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the Faster R-CNN model with the most up-to-date weights
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# model = model.to(device)
# model.eval()

# # Load class names from the 'classes.txt' file
# classnames = []
# with open('classes.txt', 'r') as f:
#     classnames = f.read().splitlines()

# # Define vehicle categories
# vehicle_classes = {
#     "car": ["car"],
#     "truck": ["truck"],
#     "bike": ["bicycle", "motorcycle"],
#     "ambulance": ["ambulance"],
#     "other_vehicles": ["bus", "train"]
# }

# # Define the region of the right lane (you can adjust these values based on your image resolution)
# def is_in_right_lane(bbox, frame_width):
#     right_lane_start = int(frame_width * 0.7)  # 70% to 100% width is considered the right lane
#     x1, y1, x2, y2 = bbox
#     # Check if the center of the bounding box is within the right lane region
#     bbox_center_x = (x1 + x2) / 2
#     return bbox_center_x >= right_lane_start

# # Read and resize the image
# image_path = 'C:\\Users\\Dhanvanth S\\Documents\\ML\\Faster-RCNN-PYTORCH-main\\cat.png'  # Replace with the actual image path
# image = cv2.imread(image_path)
# frame_height, frame_width, _ = image.shape

# # Preprocess the image (convert to tensor)
# img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

# # Get predictions from the model
# with torch.no_grad():
#     pred = model([img_tensor[0]])

#     # Extract bounding boxes, scores, and labels
#     bbox, scores, labels = pred[0]['boxes'], pred[0]['scores'], pred[0]['labels']

#     # Filter out predictions with scores above the confidence threshold (e.g., 0.7)
#     high_conf_indices = torch.where(scores > 0.7)[0]

#     right_lane_vehicle_count = 0

#     for i in high_conf_indices:
#         x1, y1, x2, y2 = bbox[i].cpu().numpy().astype('int')
#         classname_idx = labels[i].cpu().numpy().astype('int')
#         class_detected = classnames[classname_idx]

#         # Only consider vehicle classes
#         if class_detected in vehicle_classes["car"] + vehicle_classes["truck"] + vehicle_classes["bike"] + \
#                             vehicle_classes["ambulance"] + vehicle_classes["other_vehicles"]:
            
#             # Check if the vehicle is in the right lane
#             if is_in_right_lane((x1, y1, x2, y2), frame_width):
#                 right_lane_vehicle_count += 1
#                 # Draw bounding box and label for vehicles in the right lane
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(image, class_detected, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Print the number of vehicles in the right lane
#     print(f"Number of vehicles in the right lane: {right_lane_vehicle_count}")

#     # Save or display the annotated image
#     cv2.imshow('Right Lane Detection', image)
#     cv2.imwrite('output_image.jpg', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2
import numpy as np
import torch

# Load pre-trained YOLO model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can choose other models from the repo

# Load Faster R-CNN model
from torchvision.models.detection import fasterrcnn_resnet50_fpn
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# Define a function to crop the frame
def crop_frame(frame, crop_percent=0.2):
    height, width, _ = frame.shape
    cropped_frame = frame[:, :int(width * (1 - crop_percent))]  # Adjust this depending on where to crop
    return cropped_frame

# Function to detect and crop forward-moving vehicles
def detect_and_count_vehicles(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO to detect all vehicles in the frame
        results = yolo_model(frame)
        
        # Convert YOLO results to usable bounding boxes and labels
        detections = results.xyxy[0].cpu().numpy()  # bounding boxes and labels
        
        # Crop the frame based on a predefined percentage
        cropped_frame = crop_frame(frame, crop_percent=0.2)
        
        # Convert the frame to tensor and pass it to Faster R-CNN
        transformed_frame = [torch.from_numpy(cropped_frame).permute(2, 0, 1).float() / 255.0]
        outputs = faster_rcnn_model(transformed_frame)
        
        # Extract vehicle count from Faster R-CNN's output
        vehicle_count = len(outputs[0]['boxes'])  # Assuming each box is a vehicle
        
        # Print the number of forward-moving vehicles
        print(f"Forward moving vehicles: {vehicle_count}")
        
        # Show the cropped frame (optional)
        cv2.imshow('Cropped Frame', cropped_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function with your video path
video_path = 'videoplayback.mp4'
detect_and_count_vehicles(video_path)
