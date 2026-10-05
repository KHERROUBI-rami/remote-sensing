import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import tifffile as tiff
from torch.utils.data import Dataset

# ==================== المسارات وإعدادات النظام ====================
MODEL_PATH = 'project/weights/model_weights.pth'
DATA_PATH = "project/data/eurosat_ms_data"  # مسار مجلد الصور
SPLIT_PATH = 'project/data/eurosat_split_indices.json'
test_path = "random" 
SELECTED_BANDS = [1, 2, 3, 7, 10, 11]

# تحميل مؤشرات التقسيم من ملف JSON
with open(SPLIT_PATH, 'r') as f:
    split_indices = json.load(f)

# جلب مؤشرات عينة الاختبار (test) أو التقييم (val)
test_indices = split_indices.get('test', split_indices.get('val', []))

# ==================== بناء Dataset ====================
class EuroSat6BandDataset(Dataset):
    def __init__(self, root_dir, bands_idx):
        self.root_dir = root_dir
        self.bands_idx = bands_idx
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths, self.labels = [], []

        for idx, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            if os.path.isdir(cls_path):
                for img_name in sorted(os.listdir(cls_path)):
                    self.image_paths.append(os.path.join(cls_path, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = tiff.imread(self.image_paths[idx]).astype(np.float32)
        img = img[:, :, self.bands_idx]
        img = torch.from_numpy(img).permute(2, 0, 1) / 10000.0
        return img, int(self.labels[idx])

dataset = EuroSat6BandDataset(DATA_PATH, SELECTED_BANDS)

# ==================== تحميل النموذج ====================
try:
    model_int8 = torch.jit.load(MODEL_PATH)
except Exception:
    model_int8 = torch.load(MODEL_PATH)

model_int8.eval()
print(f"Loaded: {MODEL_PATH}")

# ==================== دالة الاختبار والعرض ====================
def run_int8_test(model, path, dataset, test_idxs):
    model.eval()

    if path == "random":
        # اختيار صورة عشوائية حصراً من عينة الاختبار المحددة في json
        idx = np.random.choice(test_idxs) if len(test_idxs) > 0 else np.random.randint(len(dataset))
        img_tensor, label = dataset[idx]
        real_name = dataset.classes[label]
        display_img = img_tensor[:3].permute(1, 2, 0).cpu().numpy()
        input_tensor = img_tensor.clone()
    else:
        img = tiff.imread(path).astype(np.float32)

        if img.shape[0] != 64 or img.shape[1] != 64:
            from skimage.transform import resize
            img = resize(img, (64, 64, img.shape[2]), anti_aliasing=True).astype(np.float32)

        img_bands = img[:, :, SELECTED_BANDS]
        input_tensor = torch.from_numpy(img_bands).permute(2, 0, 1) / 10000.0
        display_img = input_tensor[[2, 1, 0]].permute(1, 2, 0).numpy()

        filename = os.path.basename(path).lower()
        real_name = next((c for c in dataset.classes if c.lower() in filename), "Unknown")

    display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-5)

    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0).cpu())
        if output.dim() > 2:
            output = output.mean(dim=[2, 3])
        pred_name = dataset.classes[output.argmax(1).item()]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1e1e1e')

    ax[0].imshow(np.clip(display_img, 0, 1))
    ax[0].set_title(f"Input\nReal: {real_name}", color='white')
    ax[0].axis('off')

    color = 'lime' if pred_name.lower() == real_name.lower() else 'red'
    ax[1].imshow(np.clip(display_img, 0, 1))
    ax[1].set_title(f"AI Prediction\n{pred_name}", color=color, fontweight='bold', fontsize=12)
    ax[1].axis('off')

    plt.suptitle(f"INT8 Test — {os.path.basename(MODEL_PATH)}", color='white')
    plt.tight_layout()
    plt.show()

run_int8_test(model_int8, test_path, dataset, test_indices)
