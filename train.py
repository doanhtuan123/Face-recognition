import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import os
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


embedder = FaceNet()
# Đọc hình ảnh và chuyển đổi sang định dạng màu RGB
#img = cv2.imread("Dataset/omama/o.jpg")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tạo một đối tượng detector MTCNN
detector = MTCNN()

# Phát hiện khuôn mặt trong hình ảnh
#results = detector.detect_faces(img)
#print(results)

# Lấy thông tin vị trí khuôn mặt
#x, y, w, h = results[0]['box']

# Vẽ hộp xung quanh khuôn mặt được phát hiện
#img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 30)
detector = MTCNN()
# Hiển thị hình ảnh với hộp xung quanh khuôn mặt
class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y + h, x:x + w]
        face_arr = cv2.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory + '/' + sub_dir + '/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)

    def plot_images(self):
        plt.figure(figsize=(18, 16))
        for num, image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y) // ncols + 1
            plt.subplot(nrows, ncols, num + 1)
            plt.imshow(image)
            plt.axis('off')

faceloading = FACELOADING("Dataset")
X, Y = faceloading.load_classes()

plt.figure(figsize=(16,12))
for num,image in enumerate(X):
    ncols = 3
    nrows = len(Y)//ncols + 1
    plt.subplot(nrows,ncols,num+1)
    plt.imshow(image)
    plt.axis('off')


def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)


EMBEDDED_X = []

for img in X:
    EMBEDDED_X.append(get_embedding(img))

EMBEDDED_X = np.asarray(EMBEDDED_X)

encoder = LabelEncoder()
encoder.fit(Y)
encoder_filename = "label_encoder.pkl"
joblib.dump(encoder, encoder_filename)
Y = encoder.transform(Y)
print(Y)

plt.plot(EMBEDDED_X[0])
plt.ylabel(Y[0])

#plt.show()
X_train, X_test, y_train, y_test = train_test_split(EMBEDDED_X, Y, test_size=0.2, random_state=42)

# Tạo mô hình SVM
svm_model = SVC(kernel='linear', C=1)

# Huấn luyện mô hình
svm_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = svm_model.predict(X_test)
model_filename = "svm_model.pkl"
joblib.dump(svm_model, model_filename)

print(f"Mô hình SVM đã được lưu vào {model_filename}")
# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

plt.show()