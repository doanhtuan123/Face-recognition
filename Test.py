import cv2
import numpy as np
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt  # Import thư viện matplotlib.pyplot

# Load mô hình SVM từ tệp đã lưu
clf = joblib.load('face_detection_model.pkl')

# Đường dẫn đến ảnh đầu vào
image_path = '3ng.jpg'  # Đường dẫn đến ảnh kiểm tra

# Đọc ảnh đầu vào
input_image = cv2.imread(image_path)
gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

def extract_hog_features(image):
    # Thay đổi kích thước ô vuông và số ô vuông trong một block để kiểm soát số lượng đặc trưng
    features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

# Sử dụng Cascade Classifier để tìm khuôn mặt trên ảnh
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_input_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Trích xuất đặc trưng HOG và dự đoán khuôn mặt cho từng khuôn mặt tìm thấy
for (x, y, w, h) in faces:
    face = gray_input_image[y:y+h, x:x+w]  # Cắt khuôn mặt
    test_features = extract_hog_features(face)

    prediction = clf.predict(test_features)

    # Vẽ khung bao quanh khuôn mặt
    if prediction[0] == 1:
        color = (0, 255, 0)  # Màu xanh lá cây cho khuôn mặt
    else:
        color = (0, 0, 255)  # Màu đỏ cho không phải khuôn mặt

    cv2.rectangle(input_image, (x, y), (x + w, y + h), color, 2)

# Hiển thị ảnh với khung bao quanh khuôn mặt bằng matplotlib
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))  # Chuyển đổi màu ảnh từ BGR sang RGB
plt.show()  # Hiển thị ảnh
