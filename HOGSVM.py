import cv2
import numpy as np
from sklearn import svm
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import joblib
# Đường dẫn đến thư mục chứa ảnh khuôn mặt và không phải khuôn mặt
positive_samples_folder = 'Data\\negative'
negative_samples_folder = 'Data\\positive'

# Chuyển đổi ảnh sang vectơ đặc trưng HOG
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thay đổi kích thước ô vuông và số ô vuông trong một block để kiểm soát số lượng đặc trưng
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

# Chuẩn bị dữ liệu và trích xuất đặc trưng
X = []
y = []
# Dữ liệu khuôn mặt
for filename in os.listdir(positive_samples_folder):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(positive_samples_folder, filename))
        features = extract_hog_features(img)
        X.append(features)
        y.append(1)  # Nhãn cho khuôn mặt

# Dữ liệu không phải khuôn mặt
for filename in os.listdir(negative_samples_folder):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(negative_samples_folder, filename))
        features = extract_hog_features(img)
        X.append(features)
        y.append(0)  # Nhãn cho không phải khuôn mặt

X = np.array(X)
y = np.array(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình SVM
clf = svm.SVC(kernel='linear')

# Huấn luyện mô hình
clf.fit(X_train, y_train)

# Đánh giá hiệu suất mô hình trên tập kiểm tra
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy}")
joblib.dump(clf, 'face_detection_model.pkl')

