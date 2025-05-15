import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time
import requests

#model_name = 'resnet18'
#model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)
model.eval()

# Load ImageNet labels
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(imagenet_labels_url)
imagenet_labels = response.json()

# Define a simpler preprocessing pipeline with a smaller resize
preprocess = transforms.Compose([
    transforms.Resize(224),  # Smaller resize
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Access camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize the frame to a smaller size for faster processing
    small_frame = cv2.resize(frame, (320, 240))  # Example smaller resolution

    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    top_prob, top_class_idx = torch.topk(probabilities, 5)  # Show fewer top predictions

    cv2.putText(small_frame, f"FPS: {int(1/(time.time() - prev_time))}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for i in range(top_prob.size(0)):
        predicted_class_idx = top_class_idx[i].item()  # Access the i-th element and then get the item
        predicted_class_name = imagenet_labels[predicted_class_idx]
        probability = top_prob[i].item()  # Access the i-th probability
        text = f"{predicted_class_name}: {probability:.2f}"
        cv2.putText(small_frame, text, (10, 40 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow('Camera Feed (Basic)', small_frame)
    prev_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()