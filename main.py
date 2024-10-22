import torch
import cvzone
import cv2
from torchvision import transforms
import torchvision
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load class names from the 'classes.txt' file
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Define vehicle categories
vehicle_classes = {
    "car": ["car"],
    "truck": ["truck"],
    "bike": ["bicycle", "motorcycle"],
    "ambulance": ["ambulance"],
    "other_vehicles": ["bus", "train"]
}

# Initialize counts for each vehicle category
vehicle_count = {
    "car": 0,
    "truck": 0,
    "bike": 0,
    "ambulance": 0,
    "other_vehicles": 0
}

# Read and resize the image
image = cv2.imread('C:\\Users\\Dhanvanth S\\Documents\\ML\\Faster-RCNN-PYTORCH-main\\cat.png')
image = cv2.resize(image, (640, 480))

# Transform the image to tensor
image_transform = transforms.ToTensor()
img = image_transform(image)

with torch.no_grad():
    # Get predictions from the model
    pred = model([img])

    bbox, scores, labels = pred[0]['boxes'], pred[0]['scores'], pred[0]['labels']

    # Filter out predictions with scores above the confidence threshold (0.70)
    conf = torch.argwhere(scores > 0.70).shape[0]
    
    for i in range(conf):
        x, y, w, h = bbox[i].numpy().astype('int')
        classname = labels[i].numpy().astype('int')
        class_detected = classnames[classname]

        # Categorize and count the detected vehicle types
        if class_detected in vehicle_classes["car"]:
            vehicle_count["car"] += 1
        elif class_detected in vehicle_classes["truck"]:
            vehicle_count["truck"] += 1
        elif class_detected in vehicle_classes["bike"]:
            vehicle_count["bike"] += 1
        elif class_detected in vehicle_classes["ambulance"]:
            vehicle_count["ambulance"] += 1
        elif class_detected in vehicle_classes["other_vehicles"]:
            vehicle_count["other_vehicles"] += 1

        # Draw bounding boxes and labels on the image
        cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 4)
        cvzone.putTextRect(image, class_detected, [x + 8, y - 12], scale=2, border=1)

    # Save the annotated image
    cv2.imwrite('data1.png', image)

    # Print the counts of each vehicle category
    print("Vehicle Counts:")
    print(f"Cars: {vehicle_count['car']}")
    print(f"Trucks: {vehicle_count['truck']}")
    print(f"Bikes: {vehicle_count['bike']}")
    print(f"Ambulances: {vehicle_count['ambulance']}")
    print(f"Other Vehicles: {vehicle_count['other_vehicles']}")

# Load data from the CSV file (historical data)
# CSV file should have columns: 'car_count', 'truck_count', 'bike_count', 'ambulance_count', 'other_vehicles_count', 'time_to_pass'
data = pd.read_csv('C:\\Users\\Dhanvanth S\\Documents\\ML\\Faster-RCNN-PYTORCH-main\\vehicle_data.csv')

# Define the feature columns and the target column
X = data[['car_count', 'truck_count', 'bike_count', 'ambulance_count', 'other_vehicles_count']]
y = data['time_to_pass']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# Predict the time for current vehicle count
vehicle_features = np.array([[vehicle_count['car'], vehicle_count['truck'], vehicle_count['bike'], vehicle_count['ambulance'], vehicle_count['other_vehicles']]])
predicted_time = reg.predict(vehicle_features)[0]
print(f"Predicted Time for {sum(vehicle_count.values())} vehicles to pass the signal: {predicted_time:.2f} seconds")

# Display the image
cv2.imshow('frame', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
