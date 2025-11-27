import time
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.metrics import bbox_iou, wh_iou

def smooth_label_values(eps:float=0.05, n_classes:int=2) -> tuple[float,float]:
    return 1.0 - eps, eps / (n_classes - 1)

class YOLOv5Loss:

    def __init__(self, anchors:torch.Tensor, anchor_t:int=4, balance:list[int]=[4.0, 1.0, 0.4],
                    lambda_box=0.05, lambda_obj=0.7, lambda_cls=0.3, label_smoothing=0.0):
        
        self.anchors = anchors
        self.anchor_t = anchor_t
        self.balance = balance
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.label_smoothing = label_smoothing
        # Positive, Negative target values
        self.pt_value, self.nt_value = smooth_label_values(eps=label_smoothing) 

    def __call__(self, predictions:list[torch.Tensor], targets:torch.Tensor) -> 'tuple[torch.Tensor, torch.Tensor]':
      
        device = targets.device
        p_shape = predictions[0].shape
        
        n_batches = p_shape[0]
        n_classes = p_shape[4] - 5
        n_targets = targets.shape[0]
        
        # Initialize losses
        cls_loss = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)
        
        if n_targets > 0:
            target_classes, target_boxes, indices, target_anchors = self.build_targets(predictions, targets)

        # Calculate losses for each prediction scale
        for i, layer_pred in enumerate(predictions):
            target_obj = torch.zeros(layer_pred.shape[:4], dtype=layer_pred.dtype, device=device) # Target objectness

            if n_targets > 0 and target_classes[i].shape[0] > 0:
                n_built_targets = target_classes[i].shape[0]  # number of built targets
                img_indices, anchor_indices, cell_j, cell_i = indices[i]

                pred_xy, pred_wh, _, pred_cls = \
                    layer_pred[img_indices, anchor_indices, cell_j, cell_i].tensor_split((2, 4, 5), dim=1)
                
                # Bbox regression loss
                pred_xy = pred_xy.sigmoid() * 2 - 0.5
                pred_wh = (pred_wh.sigmoid() * 2) ** 2 * target_anchors[i]
                pred_box = torch.cat((pred_xy, pred_wh), 1)
                iou = bbox_iou(pred_box, target_boxes[i], CIoU=True).squeeze()
                box_loss += (1.0 - iou).mean()  # iou loss
                
                # Store Iou as target objectness
                iou = iou.detach().clamp(0).type(target_obj.dtype)     
                target_obj[img_indices, anchor_indices, cell_j, cell_i] = iou

                # Object type classification loss
                if n_classes > 1:  # cls loss (only if multiple classes)
                    target_cls = torch.full_like(pred_cls, self.nt_value, device=device)
                    target_cls[range(n_built_targets), target_classes[i]] = self.pt_value
                    cls_loss += F.binary_cross_entropy_with_logits(pred_cls, target_cls)

            # Objectness Loss
            pred_obj = layer_pred[..., 4]
            obj_loss += F.binary_cross_entropy_with_logits(pred_obj, target_obj) * self.balance[i]

        box_loss *= self.lambda_box
        obj_loss *= self.lambda_obj
        cls_loss *= self.lambda_cls
        
        total_loss = (box_loss + obj_loss + cls_loss) * n_batches
        loss_items = torch.cat((box_loss, obj_loss, cls_loss)).detach()
        
        return total_loss, loss_items
    
    
    def build_targets(self, predictions:list[torch.Tensor], targets:torch.Tensor) -> tuple[list]:
        device = targets.device
        p_shape = predictions[0].shape

        n_anchors = p_shape[1]
        n_targets = targets.shape[0]
        n_layers = len(predictions)

        if n_targets == 0:
            raise ValueError("No targets provided")
        if len(predictions) != self.anchors.shape[0]:
            raise ValueError("Number of layers and anchors do not match")
        if n_anchors != self.anchors.shape[1]:
            raise ValueError("Number of layer bounding boxes and provided anchors do not match")
        
        target_classes, target_boxes, indices, target_anchors = [], [], [], []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain

        anchor_indices = torch.arange(n_anchors, device=device).float()
        anchor_indices = anchor_indices.view(n_anchors, 1).repeat(1, n_targets)
        # Append anchor indices at the end of each target
        target_anchor_pairs = torch.cat((targets.repeat(n_anchors, 1, 1), anchor_indices[..., None]), dim=2)
        g = 0.5 # bias
        # Define offsets for each direction in the grid cell. Offsets will be substracted 
        # from grid cell center point to select adjacent cells.
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=device).float() * g
        # -> [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1] -> current, left, up, right, down
        
        assigned_targets = torch.zeros((n_targets,), device=device)
        for i in range(n_layers):
            layer_anchors, layer_shape = self.anchors[i], predictions[i].shape
            # Adjust scale of targets based on layer grid size
            gain[2:6] = torch.tensor(layer_shape)[[3, 2, 3, 2]]  # xyxy gain
            scaled_ta_pairs = target_anchor_pairs * gain

            # Match targets to anchors
            ratio = scaled_ta_pairs[..., 4:6] / layer_anchors[:, None]  # wh ratio
            rmax = torch.max(ratio, 1 / ratio).max(dim=2)[0]
            mask_ = rmax < self.anchor_t
            selected_ta_pairs = scaled_ta_pairs[mask_]  # filter
            assigned_targets += mask_.sum(dim=0)

            

            # Compute offsets
            gxy = selected_ta_pairs[:, 2:4]  # grid xy (x increases right, y increases down)
            gxy_inv = gain[[2, 3]] - gxy   # grid xy inverse (x increases left, y increases up)
            # -- Check if center point is in the left | up sectors of the grid cell
            left, up = ((gxy % 1 < g) & (gxy > 1)).T
            # -- Check if center point is in the right | down sectors of the grid cell
            right, down = ((gxy_inv % 1 < g) & (gxy_inv > 1)).T
            # -- Stack boolean masks and compute offsets
            current = torch.ones_like(left) # All true. To select the cells in which the center point is located
            stacked_conditions = torch.stack((current, left, up, right, down))
            built_targets = selected_ta_pairs.repeat((5, 1, 1))[stacked_conditions]
            built_target_offsets = (torch.zeros_like(gxy)[None] + off[:, None])[stacked_conditions]            

            # Prepare built-targets for loss computation 
            bc, gxy, gwh, a = built_targets.chunk(4, 1)
            anchor_indices = a.long().view(-1)
            img_indices, classes = bc.long().T
            # -- Compute grid cell indices
            gij = (gxy - built_target_offsets).long()
            gi, gj = gij.T
            gj = gj.clamp(0, layer_shape[2] - 1)
            gi = gi.clamp(0, layer_shape[3] - 1)
            # -- Append results
            indices.append((img_indices, anchor_indices, gj, gi))
            target_boxes.append(torch.cat((gxy - gij, gwh), dim=1))
            target_anchors.append(layer_anchors[anchor_indices])
            target_classes.append(classes) 
        
        #msg = ("Try to increase the 'anchor_t' value or use better anchors")
        #if not (assigned_targets > 0).all():
        #   print(msg)
            
        return target_classes, target_boxes, indices, target_anchors