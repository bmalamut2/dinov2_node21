import os
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
import cv2
import torchvision
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoImageProcessor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou, nms

CONFIG = {
    "data_root": "./dataset_node21",
    "images_dir": "cxr_images/proccessed_mha",
    "metadata_file": "metadata.csv",
    "checkpoint_path": "best_node21_dino.pth", 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_name": "microsoft/rad-dino-maira-2",
    "target_size": (518, 518),
    "iou_threshold": 0.1,       # Hit criteria
    "score_threshold": 0.05,    # Minimum confidence to draw a box
    "num_classes": 2
}

print(f"Running Inference on: {CONFIG['device']}")

class DINOFasterRCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CONFIG['model_name'])
        
        class BackboneWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.out_channels = model.config.hidden_size
            def forward(self, x):
                outputs = self.model(x)
                patch_tokens = outputs.last_hidden_state[:, 1:, :] 
                B, N, C = patch_tokens.shape
                side = int(N ** 0.5) 
                features = patch_tokens.transpose(1, 2).view(B, C, side, side)
                return {"0": features}

        anchor_generator = AnchorGenerator(
            sizes=((8, 16, 32, 64),),
            aspect_ratios=((1.0),) * 1
        )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )

        self.detector = FasterRCNN(
            backbone=BackboneWrapper(self.backbone),
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            image_mean=[0,0,0], image_std=[1,1,1],
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_fg_iou_thresh=0.5,
            rpn_bg_iou_thresh=0.3
        )

    def forward(self, images, targets=None):
        return self.detector(images, targets)

class Node21Dataset(torch.utils.data.Dataset):
    def __init__(self, root, img_dir, csv_file, processor):
        self.root = root
        self.img_path = os.path.join(root, img_dir)
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.processor = processor
        self.image_ids = self.df['img_name'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        records = self.df[self.df['img_name'] == img_id]
        
        mha_path = os.path.join(self.img_path, img_id)
        image_obj = sitk.ReadImage(mha_path)
        image_np = sitk.GetArrayFromImage(image_obj) 
        if image_np.ndim == 3: image_np = image_np[0]
        
        # Preserve original for visualization
        original_img = image_np.copy()
        
        image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8) * 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
        # Parse Boxes
        boxes = []
        for _, row in records.iterrows():
            x, y, w, h = row['x'], row['y'], row['width'], row['height']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Resize logic for model input
        orig_h, orig_w = image_rgb.shape[:2]
        new_h, new_w = CONFIG['target_size']
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Process for Model
        inputs = self.processor(images=image_resized, return_tensors="pt")
        image_tensor = inputs.pixel_values.squeeze(0) 
        
        # Scale boxes for metric calculation
        boxes_scaled = boxes.clone()
        if len(boxes) > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes_scaled[:, 0] *= scale_x
            boxes_scaled[:, 2] *= scale_x
            boxes_scaled[:, 1] *= scale_y
            boxes_scaled[:, 3] *= scale_y
        
        if len(boxes_scaled) == 0:
            boxes_scaled = torch.zeros((0, 4), dtype=torch.float32)

        return image_tensor, boxes_scaled, original_img, img_id

def run_inference():
    # Load Model
    try:
        processor = AutoImageProcessor.from_pretrained(CONFIG['model_name'])
    except:
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(CONFIG['model_name'])

    model = DINOFasterRCNN(num_classes=CONFIG['num_classes'])
    model.load_state_dict(torch.load(CONFIG['checkpoint_path']))
    model.to(CONFIG['device'])
    model.eval()

    full_dataset = Node21Dataset(CONFIG['data_root'], CONFIG['images_dir'], CONFIG['metadata_file'], processor)
    total_size = len(full_dataset)
    indices = list(range(total_size))
    np.random.seed(42) 
    np.random.shuffle(indices)
    val_indices = indices[:int(0.2 * total_size)]
    
    val_ds = torch.utils.data.Subset(full_dataset, val_indices)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    results = []

    print(f"Evaluating {len(val_ds)} images...")

    with torch.no_grad():
        for i, (img, gt_boxes, orig_img, img_id) in enumerate(loader):
            img = img.to(CONFIG['device'])
            gt_boxes = gt_boxes[0] # Remove batch dim
            
            output = model(img)[0]
            
            # Filter by confidence score
            keep = output['scores'] > CONFIG['score_threshold']
            pred_boxes = output['boxes'][keep].cpu()
            scores = output['scores'][keep].cpu()
            
            tp = 0
            fp = 0
            matched_gt = set()
            
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes)
                for p_idx, row in enumerate(ious):
                    best_iou, gt_idx = row.max(0)
                    if best_iou > CONFIG['iou_threshold'] and gt_idx.item() not in matched_gt:
                        tp += 1
                        matched_gt.add(gt_idx.item())
                    else:
                        fp += 1
            elif len(pred_boxes) > 0:
                fp = len(pred_boxes)
            
            fn = len(gt_boxes) - tp
            
            # Perfect score: (TP * 1.0)
            # Bad score: Negative
            score = (tp * 1.0) - (fp * 0.5) - (fn * 2.0)
            
            results.append({
                'img_id': img_id[0],
                'orig_img': orig_img[0].numpy(),
                'gt_boxes': gt_boxes,
                'pred_boxes': pred_boxes,
                'scores': scores,
                'stats': (tp, fp, fn),
                'score': score
            })
            
            if i % 50 == 0: print(f"Processed {i}/{len(val_ds)}")

    # Sort Results
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results[:3], results[-3:]

def save_visual(result_item, title, filename):
    img = result_item['orig_img']
    gt = result_item['gt_boxes']
    pred = result_item['pred_boxes']
    stats = result_item['stats'] # TP, FP, FN
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    ax = plt.gca()
    
    h_orig, w_orig = img.shape
    h_model, w_model = CONFIG['target_size']
    scale_x = w_orig / w_model
    scale_y = h_orig / h_model
    
    # Draw Ground Truth (Green)
    for box in gt:
        box = box * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        x, y, x2, y2 = box
        rect = plt.Rectangle((x, y), x2-x, y2-y, fill=False, color='#00FF00', linewidth=3, label='GT')
        ax.add_patch(rect)

    # Draw Predictions (Red)
    for box in pred:
        box = box * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        x, y, x2, y2 = box
        rect = plt.Rectangle((x, y), x2-x, y2-y, fill=False, color='red', linewidth=2, linestyle='--', label='Pred')
        ax.add_patch(rect)
    
    # Add Legend handles manually to avoid duplicates
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#00FF00', lw=3),
                    Line2D([0], [0], color='red', lw=2, linestyle='--')]
    ax.legend(custom_lines, ['Ground Truth', 'Prediction'])

    plt.title(f"{title}\nID: {result_item['img_id']}\nTP: {stats[0]} | FP: {stats[1]} | FN: {stats[2]}")
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    top3, bot3 = run_inference()
    
    print("\n--- SAVING RESULTS ---")
    for i, res in enumerate(top3):
        save_visual(res, f"TOP #{i+1} RESULT", f"result_top_{i+1}.png")
        print(f"Saved Top {i+1}: Score {res['score']}")
        
    for i, res in enumerate(bot3):
        save_visual(res, f"BOTTOM #{i+1} RESULT", f"result_bot_{i+1}.png")
        print(f"Saved Bottom {i+1}: Score {res['score']}")