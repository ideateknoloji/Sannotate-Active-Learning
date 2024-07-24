import cv2
from yolov5 import detect

# Path to your image
image_path = "C:/Users/ufuk.cefaker/Desktop/S-Annotate/Sannotate-Active-Learning-main/asdf.jpeg"

# Load the image
img = cv2.imread(image_path)

# Perform detection
results = detect(img, source='C:/Users/ufuk.cefaker/Desktop/S-Annotate/Sannotate-Active-Learning-main/best.pt')

# Print results (class labels, bounding boxes, and confidences)
for result in results.xyxy[0]:
    print(f"Class: {result[-1]}, Confidence: {result[-2]}, BBox: {result[:4]}")

# Display image with bounding boxes
for *xyxy, conf, cls in results.xyxy[0]:
    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    cv2.putText(img, f'{cls}', (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('YOLOv5 Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
