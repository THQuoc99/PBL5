import torch
import cv2
import threading
import queue
from torchvision.ops import nms

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

class VideoProcessor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = False
        self.process_thread = None

    # def load_image(self, image_array):
    #     img = cv2.resize(image_array, (640, 640))
    #     img_colored = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img_tensor = torch.from_numpy(img_colored / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
    #     return img, img_tensor
    
    def load_image(self, image_array):
        img = cv2.resize(image_array, (640, 640))
        img_colored = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_colored / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image_array, img_tensor

    def process_predictions(self, predictions):
        # Sử dụng hàm tối ưu mới
        bboxes = optimized_cells_to_bboxes(predictions, self.model.head.anchors, 
                                         self.model.head.stride, self.device, to_list=False)
        bboxes = optimized_non_max_suppression(bboxes, iou_threshold=0.25, 
                                             threshold=0.45, 
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