import os
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms, box_iou

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "data_root": "./dataset_node21",       
    "images_dir": "cxr_images/proccessed_mha", 
    "metadata_file": "metadata.csv",       # USE REAL METADATA ONLY
    "batch_size": 4,                       
    "lr_backbone": 1e-5,                   # Low LR to protect DINO weights
    "lr_head": 1e-4,                       # Higher LR for new Head
    "epochs": 15,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 2,                      # Background + Nodule
    "model_name": "microsoft/rad-dino-maira-2",
    "target_size": (518, 518),             # Must be multiple of 14
    "iou_threshold": 0.1                   # IoU > 0.1 counts as a "Hit" in medical AI
}

print(f"Running on: {CONFIG['device']}")

# ==========================================
# 2. CUSTOM DATASET
# ==========================================
class Node21Dataset(Dataset):
    def __init__(self, root, img_dir, csv_file, processor=None):
        self.root = root
        self.img_path = os.path.join(root, img_dir)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.processor = processor
        self.image_ids = self.df['filename'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        records = self.df[self.df['filename'] == img_id]
        
        # Load .mha
        mha_path = os.path.join(self.img_path, img_id)
        if not os.path.exists(mha_path):
            raise FileNotFoundError(f"Image not found: {mha_path}")
            
        image_obj = sitk.ReadImage(mha_path)
        image_np = sitk.GetArrayFromImage(image_obj) 
        
        if image_np.ndim == 3: image_np = image_np[0]
        # Normalize to 0-255
        image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8) * 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
        # Parse Boxes
        boxes = []
        labels = []
        for _, row in records.iterrows():
            x, y, w, h = row['x'], row['y'], row['width'], row['height']
            boxes.append([x, y, x + w, y + h])
            labels.append(1) 

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Handle empty boxes (negative samples)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Resize Image & Boxes
        orig_h, orig_w = image_rgb.shape[:2]
        new_h, new_w = CONFIG['target_size']
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        
        if len(boxes) > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y

        # Process for DINO (Normalization)
        inputs = self.processor(images=image_resized, return_tensors="pt")
        image_tensor = inputs.pixel_values.squeeze(0) 

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class DINOFasterRCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CONFIG['model_name'])
        
        # Wrapper to make DINO output look like a CNN feature map
        class BackboneWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.out_channels = model.config.hidden_size
            def forward(self, x):
                outputs = self.model(x)
                patch_tokens = outputs.last_hidden_state[:, 1:, :] 
                B, N, C = patch_tokens.shape
                side = int(N ** 0.5) # 518/14 = 37
                # Reshape (Batch, Patches, Dim) -> (Batch, Dim, Height, Width)
                features = patch_tokens.transpose(1, 2).view(B, C, side, side)
                return {"0": features}

        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )

        self.detector = FasterRCNN(
            backbone=BackboneWrapper(self.backbone),
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool_roi_layers=roi_pooler,
            image_mean=[0,0,0], image_std=[1,1,1] # Skip internal normalization (done by processor)
        )

    def forward(self, images, targets=None):
        return self.detector(images, targets)

# ==========================================
# 4. METRICS: FROC & VALIDATION LOSS
# ==========================================
def calculate_froc_score(all_preds, all_gts, iou_thresh=0.1):
    """
    Calculates Sensitivity @ specific False Positives per Image (FROC analysis)
    """
    detected_hits = [] # List of (score, is_tp)
    total_gt_nodules = 0
    total_images = len(all_gts)

    for i in range(total_images):
        gt_boxes = all_gts[i]['boxes'].numpy()
        pred_boxes = all_preds[i]['boxes']
        pred_scores = all_preds[i]['scores']
        
        total_gt_nodules += len(gt_boxes)
        
        if len(pred_boxes) == 0: continue
            
        # Sort preds by confidence
        sorted_idxs = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_idxs]
        pred_scores = pred_scores[sorted_idxs]

        matched_gt = set()
        
        for j, p_box in enumerate(pred_boxes):
            # If we have no GT boxes, everything is False Positive
            if len(gt_boxes) == 0:
                detected_hits.append((pred_scores[j], 0))
                continue
            
            # Calculate IoU with all GT boxes
            # simple manual IoU or use torchvision
            p_tensor = torch.tensor([p_box], dtype=torch.float32)
            g_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
            ious = box_iou(p_tensor, g_tensor).numpy()[0] # (num_gt,)
            
            best_iou = -1
            best_gt_idx = -1
            
            if len(ious) > 0:
                best_iou = np.max(ious)
                best_gt_idx = np.argmax(ious)
            
            if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
                detected_hits.append((pred_scores[j], 1)) # TP
                matched_gt.add(best_gt_idx)
            else:
                detected_hits.append((pred_scores[j], 0)) # FP

    # Sort all hits globally by score
    detected_hits.sort(key=lambda x: x[0], reverse=True)

    tps = 0
    fps = 0
    froc_map = {}
    checkpoints = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    
    for score, is_tp in detected_hits:
        if is_tp: tps += 1
        else: fps += 1
        
        avg_fps = fps / total_images
        sensitivity = tps / total_gt_nodules if total_gt_nodules > 0 else 0
        
        for c in checkpoints:
            if avg_fps >= c and c not in froc_map:
                froc_map[c] = sensitivity

    # Fill missing keys (if model never reached that many FPs)
    for c in checkpoints:
        if c not in froc_map: froc_map[c] = froc_map.get(c/2, 0.0) # simple fallback

    print(f"\n--- FROC Analysis (Total Nodules: {total_gt_nodules}) ---")
    for c in checkpoints:
        print(f"  Sens @ {c:5} FPs/Img: {froc_map.get(c, 0.0):.4f}")
        
    return froc_map.get(1.0, 0.0) # Return Sens @ 1.0 FP as primary metric

def evaluate_model(model, val_loader, device):
    # 1. Calculate Validation Loss (Hack: Train mode + No Grad)
    model.train()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            val_loss += sum(loss for loss in loss_dict.values()).item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    # 2. Calculate Metrics (Eval mode)
    model.eval()
    all_preds = []
    all_gts = []
    
    print("Generating predictions for FROC...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                all_preds.append({
                    'boxes': output['boxes'].cpu().numpy(),
                    'scores': output['scores'].cpu().numpy()
                })
                all_gts.append({
                    'boxes': targets[i]['boxes']
                })
                
    froc_score = calculate_froc_score(all_preds, all_gts, iou_thresh=CONFIG['iou_threshold'])
    
    return avg_val_loss, froc_score

# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    # Setup
    processor = AutoImageProcessor.from_pretrained(CONFIG['model_name'])
    dataset = Node21Dataset(CONFIG['data_root'], CONFIG['images_dir'], CONFIG['metadata_file'], processor)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Model Init
    print(f"Loading {CONFIG['model_name']}...")
    model = DINOFasterRCNN(num_classes=CONFIG['num_classes'])
    model.to(CONFIG['device'])

    # Differential Learning Rates
    params = [
        {"params": model.backbone.parameters(), "lr": CONFIG['lr_backbone']},
        {"params": model.detector.parameters(), "lr": CONFIG['lr_head']}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_froc = 0.0
    print("Starting Training...")

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(image.to(CONFIG['device']) for image in images)
            targets = [{k: v.to(CONFIG['device']) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Stabilize training
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if batch_idx % 10 == 0:
                print(f"Ep {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {losses.item():.4f}")

        # End of Epoch Evaluation
        val_loss, val_froc = evaluate_model(model, val_loader, CONFIG['device'])
        lr_scheduler.step()
        
        print(f"\n>>> EPOCH {epoch+1} RESULT: Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | FROC (1.0 FP): {val_froc:.4f}")
        
        # Save Best Model
        if val_froc > best_froc:
            best_froc = val_froc
            torch.save(model.state_dict(), "best_node21_dino.pth")
            print(">>> New Best Model Saved! <<<")
        
        # Save Latest Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, "latest_checkpoint.pth")
        print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()