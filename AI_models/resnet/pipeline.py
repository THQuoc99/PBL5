# pipeline.py (phần liên quan ResNet-50+)
from models.resnet50 import ResNet50Plus, map_labels
from face_emotion_detection.utils.data_utils_5emotion import load_fer2013

# Load dữ liệu
fer_images, fer_labels = load_fer2013("data/fer2013/train.csv")
new_labels, mask = map_labels(fer_labels)
fer_images, fer_labels = fer_images[mask], new_labels

# Khởi tạo và huấn luyện
model = ResNet50Plus()
model.train(fer_images, fer_labels, epochs=20, lr=0.001)

# Dự đoán
test_images, test_labels = load_fer2013("data/fer2013/test.csv", train=False)
new_test_labels, mask = map_labels(test_labels)
test_images, test_labels = test_images[mask], new_test_labels
preds = model.predict(test_images)
accuracy, conf_matrix = model.evaluate(test_images, test_labels)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)