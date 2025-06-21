import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

def save_image_safe(img_array, filename):
    """Ensure proper format for OpenCV imwrite (uint8 BGR)"""
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    elif np.issubdtype(img_array.dtype, np.integer):
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img_array)

model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')  # Change to 'cuda' if GPU is available

test_img_folder = 'LR/*'
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# Load ESRGAN model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

print(f'Model path {model_path}. \nTesting...')

for idx, path in enumerate(glob.glob(test_img_folder), start=1):
    base = osp.splitext(osp.basename(path))[0]
    print(f'{idx}: {base}')
    
    # Load image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: Unable to read image {path}, skipping.")
        continue
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

    save_image_safe(output, osp.join(results_folder, f'{base}_rlt.png'))
