
Hướng Dẫn Chạy Dự Án
Mở terminal của thư mục làm việc là Resnet
Bước 1: Cài Đặt Dependencies
        pip install -r face_emotion_detection/requirements.txt"


Bước 2: Tải và Tách Dữ Liệu sau đó gộp FER-2013 và Affecnet
Tải fer2013.csv từ Kaggle (https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
Tải dữ liệu từ AffectNet
Xử lý dữ liệu về kích thước 128x128 và lưu ở định dạng csv vào data


Bước 3: Xử Lý Dữ Liệu
Chạy script để xử lý dữ liệu:
        face_emotion_detection/scripts/process_data.py
Tạo file train_images.pt, train_labels.pt, test_images.pt, test_labels.pt .vv trong data/processed/.


Bước 4: Huấn Luyện Mô Hình
Chạy script huấn luyện:
        python face_emotion_detection/scripts/train_resnet.py
Trên GPU (nếu có): Huấn luyện 20 epoch mất 10-30 phút.
Trên CPU: Mất 1-2 giờ, tùy phần cứng.

Bước 5 : Đánh giá dựa trên tập test
        python face_emotion_detection/scripts/evaluate_resnet.py

#Bước 6: Kiểm Tra Kết Quả, vẽ đồ thị 
        python face_emotion_detection/scripts/plot_training_logs.py
Kiểm tra data/logs/training_loss.txt để xem loss.

#Bước 7: Dự Đoán và Trực Quan Hóa
Chạy script dự đoán:
        python face_emotion_detection/scripts/predict_resnet.py


Hiển thị accuracy, confusion matrix, 5 ảnh test, và ghi video từ webcam vào data/test_images/output_video.mp4.
Xem video data/test_images/output_video.mp4 và cập nhật README.md với nhận xét lỗi/giải pháp.