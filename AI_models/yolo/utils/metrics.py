
import math
import torch
from torchvision.ops import nms



def wh_iou(wh1, wh2, eps=1e-7):
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def cells_to_bboxes(predictions, anchors, strides,device, is_pred=False, to_list=True):
    num_out_layers = len(predictions)
    grid = [torch.empty(0,device=device) for _ in range(num_out_layers)]  # initialize
    anchor_grid = [torch.empty(0,device=device) for _ in range(num_out_layers)]  # initialize
    predictions = [p.to(device, non_blocking=True) for p in predictions]
    all_bboxes = []
    for i in range(num_out_layers):
        bs, naxs, ny, nx, _ = predictions[i].shape
        stride = torch.tensor(strides[i], device=device)
        grid[i], anchor_grid[i] = make_grids(anchors, naxs,device=device, ny=ny, nx=nx, stride=stride, i=i)
        grid[i] = grid[i].to(device)  # Chuyển grid về cùng device
        anchor_grid[i] = anchor_grid[i].to(device)  # Chuyển anchor_grid về cùng device
        
        if is_pred:
            # formula here: https://github.com/ultralytics/yolov5/issues/471
            #xy, wh, conf = predictions[i].sigmoid().split((2, 2, 80 + 1), 4)
            layer_prediction = predictions[i].sigmoid()
            obj = layer_prediction[..., 4:5]
            xy = (2 * (layer_prediction[..., 0:2]) + grid[i] - 0.5) * stride
            wh = ((2*layer_prediction[..., 2:4])**2) * anchor_grid[i]
            best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)

        else:
            predictions[i] = predictions[i].to(device, non_blocking=True)
            obj = predictions[i][..., 4:5]
            xy = (predictions[i][..., 0:2] + grid[i]) * stride
            wh = predictions[i][..., 2:4] * stride
            best_class = predictions[i][..., 5:6]

        scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)

        all_bboxes.append(scale_bboxes)

    return torch.cat(all_bboxes, dim=1).tolist() if to_list else torch.cat(all_bboxes, dim=1)

def make_grids(anchors, naxs, stride,device, nx=20, ny=20, i=0):
    anchors = anchors.to(device)
    
    x_grid = torch.arange(nx)
    x_grid = x_grid.repeat(ny).reshape(ny, nx)

    y_grid = torch.arange(ny).unsqueeze(0)
    y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)

    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)
    stride = stride.to(device)
    anchor_grid = (anchors[i]*stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)

    return xy_grid, anchor_grid
    
def non_max_suppression(batch_bboxes, iou_threshold, threshold, max_detections=300, tolist=True):

    """new_bboxes = []
    for box in bboxes:
        if box[1] > threshold:
            box[3] = box[0] + box[3]
            box[2] = box[2] + box[4]
            new_bboxes.append(box)"""

    bboxes_after_nms = []
    for boxes in batch_bboxes:
        boxes = torch.masked_select(boxes, boxes[..., 1:2] > threshold).reshape(-1, 6)

        # from xywh to x1y1x2y2

        boxes[..., 2:3] = boxes[..., 2:3] - (boxes[..., 4:5] / 2)
        boxes[..., 3:4] = boxes[..., 3:4] - (boxes[..., 5:] / 2)
        boxes[..., 5:6] = boxes[..., 5:6] + boxes[..., 3:4]
        boxes[..., 4:5] = boxes[..., 4:5] + boxes[..., 2:3]

        indices = nms(boxes=boxes[..., 2:] + boxes[..., 0:1], scores=boxes[..., 1], iou_threshold=iou_threshold)
        boxes = boxes[indices]

        # sorts boxes by objectness score but it's already done internally by torch metrics's nms
        # _, si = torch.sort(boxes[:, 1], dim=0, descending=True)
        # boxes = boxes[si, :]
        
        if boxes.shape[0] > max_detections:
            boxes = boxes[:max_detections, :]

        bboxes_after_nms.append(
            boxes.tolist() if tolist else boxes
        )

    if tolist:
        return bboxes_after_nms
    else:
        # Lọc các tensor không rỗng
        non_empty_boxes = [b for b in bboxes_after_nms if b.shape[0] > 0]
        if len(non_empty_boxes) == 0:
            device = batch_bboxes[0].device if len(batch_bboxes) > 0 and isinstance(batch_bboxes[0], torch.Tensor) else "cpu"
            return torch.empty((0, 6), device=device)
        return torch.cat(non_empty_boxes, dim=0)

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
