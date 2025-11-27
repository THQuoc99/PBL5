import torch
import cv2
import argparse
import time
import threading
import queue
import sys
from pathlib import Path
from torchvision.ops import nms

# Thêm đường dẫn của các thư mục vào sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from yolo.models.common import YOLOV5m  # Import model YOLOv5m đã định nghĩa

# --- Thiết lập tham số ---
parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, default="/weights/yolo/Yolo-Weight.pt", help="Path to checkpoint")
parser.add_argument("--source", type=str, default="Test-1.jpg", help="Image path or 'webcam'")
parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--iou_thresh", type=float, default=0.45, help="IOU threshold for NMS")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run model on")
args = parser.parse_args()

# Hàm tối ưu để tạo grid cho tất cả các layer
def make_grids_all_layers(anchors, predictions, strides, device):
    grids = []
    anchor_grids = []
    
    for i, pred in enumerate(predictions):
        bs, naxs, ny, nx, _ = pred.shape
        stride = torch.tensor(strides[i], device=device)
        
        # Tạo grid
        x_grid = torch.arange(nx, device=device)
        y_grid = torch.arange(ny, device=device)
        y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)
        
        # Tạo anchor grid
        anchor_grid = (anchors[i]*stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)
        
        grids.append(xy_grid)
        anchor_grids.append(anchor_grid)
    
    return grids, anchor_grids

# Hàm cells_to_bboxes tối ưu
def optimized_cells_to_bboxes(predictions, anchors, strides, device, to_list=True):
    # Tạo grid và anchor_grid cho tất cả các layer
    grids, anchor_grids = make_grids_all_layers(anchors, predictions, strides, device)
    
    all_bboxes = []
    for i, pred in enumerate(predictions):
        pred = pred.to(device, non_blocking=True)
        
        layer_prediction = pred.sigmoid()
        obj = layer_prediction[..., 4:5]
        xy = (2 * layer_prediction[..., 0:2] + grids[i] - 0.5) * strides[i]
        wh = ((2*layer_prediction[..., 2:4])**2) * anchor_grids[i]
        best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)
        
        # Kết hợp tất cả thông tin
        scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(pred.shape[0], -1, 6)
        all_bboxes.append(scale_bboxes)
    
    result = torch.cat(all_bboxes, dim=1)
    return result.tolist() if to_list else result

# Hàm non_max_suppression tối ưu
def optimized_non_max_suppression(batch_bboxes, iou_threshold, threshold, max_detections=100, tolist=True):
    # Lọc theo confidence score và các điều kiện khác
    confidence_mask = batch_bboxes[..., 1] > threshold
    size_mask = (batch_bboxes[..., 4] > 48) & (batch_bboxes[..., 5] > 48)
    aspect_ratio = batch_bboxes[..., 4] / batch_bboxes[..., 5]
    aspect_mask = (aspect_ratio > 0.5) & (aspect_ratio < 1.5)
    
    # Kết hợp các mask
    mask = confidence_mask & size_mask & aspect_mask
    boxes = batch_bboxes[mask]
    
    if boxes.shape[0] == 0:
        return [] if tolist else torch.empty((0, 6), device=batch_bboxes.device)
    
    # Vectorized coordinate conversion
    boxes[..., 2:4] -= boxes[..., 4:] / 2
    boxes[..., 4:] += boxes[..., 2:4]
    
    # Batch NMS
    indices = nms(boxes=boxes[..., 2:], scores=boxes[..., 1], iou_threshold=iou_threshold)
    boxes = boxes[indices]
    
    # Giới hạn số lượng detection
    if boxes.shape[0] > max_detections:
        boxes = boxes[:max_detections]
    
    return boxes.tolist() if tolist else boxes

# --- Load model ---
device = torch.device(args.device)
num_classes = 1  # Số class của dataset
anchors = [
    [[0.88, 1.13],  [1.38, 2.0],  [2.5, 2.75]],
    [[1.44, 2.31],  [2.56, 3.44],  [3.63, 4.88]],
    [[2.34, 3.44],  [3.75, 5.63],  [5.94, 9.06]]
]

model = YOLOV5m(first_out=64, nc=num_classes, anchors=anchors, ch=[256, 512, 1024]).to(device)
checkpoint = torch.load(args.weights, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


class VideoProcessor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = False
        self.process_thread = None

    def load_image(self, image_array):
        img = cv2.resize(image_array, (640, 640))
        img_colored = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_colored / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        return img, img_tensor

    def process_predictions(self, predictions):
        # Sử dụng hàm tối ưu mới
        bboxes = optimized_cells_to_bboxes(predictions, self.model.head.anchors, 
                                         self.model.head.stride, self.device, to_list=False)
        bboxes = optimized_non_max_suppression(bboxes, iou_threshold=args.iou_thresh, 
                                             threshold=args.conf_thresh, 
                                             max_detections=100, tolist=False)
        
        results = []
        for bbox in bboxes:
            x1 = max(bbox[2], 0)
            y1 = max(bbox[3], 0)
            x2 = min(bbox[4], 640)
            y2 = min(bbox[5], 640)
            conf = bbox[1]
            cls = int(bbox[0])
            results.append((x1, y1, x2, y2, conf, cls))
        return results

    def process_frame(self, frame):
        img, img_tensor = self.load_image(frame)
        with torch.no_grad():
            predictions = self.model(img_tensor)
        detections = self.process_predictions(predictions)
        return img, detections

    def process_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break
                img, detections = self.process_frame(frame)
                self.result_queue.put((img, detections))
            except queue.Empty:
                continue

    def start(self):
        self.running = True
        self.process_thread = threading.Thread(target=self.process_loop)
        self.process_thread.start()

    def stop(self):
        self.running = False
        if self.process_thread:
            self.process_thread.join()

# --- Vẽ bbox lên ảnh ---
def draw_boxes(img, detections, class_names):
    for (x1, y1, x2, y2, conf, cls) in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{class_names[cls]} {conf:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

# --- Chạy detection ---
def run_detection(source):
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        # Thiết lập FPS mong muốn
        fps = 30
        delay = int(1000/fps)  # Chuyển sang milliseconds
        
        processor = VideoProcessor(model, device)
        processor.start()
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    processor.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    continue
                
                try:
                    img, detections = processor.result_queue.get(timeout=1)
                    frame = draw_boxes(img, detections, class_names=["face"])
                    
                    # Hiển thị FPS
                    fps = 1.0 / (time.time() - start_time)
                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("YOLOv5 Detection", frame)
                except queue.Empty:
                    continue
                
                # Thêm delay để kiểm soát FPS
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break

        finally:
            processor.stop()
            cap.release()
            cv2.destroyAllWindows()


# --- Chạy script ---
"""
if __name__ == "__main__":
    print(torch.cuda.is_available())  # Phải ra True
    print(torch.cuda.get_device_name(0))  # Phải ra "GeForce GTX 1650"
    print(torch.version.cuda)
    run_detection(args.source)
"""
