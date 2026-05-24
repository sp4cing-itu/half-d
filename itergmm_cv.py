import glob
import imagesize
import numpy as np
import os
from matplotlib import pyplot as plt
import string
import random
import cv2  # OpenCV library imported
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# --- Helper Functions ---

def rand_str(S=15):
    """Generates a random string of the specified length."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))

def iouf(box1, box2):
    """
    Calculates how much a bounding box (box1) overlaps with a crop area (box2)
    and returns the cropped coordinates.
    """
    # Calculate intersection area
    inter_x1 = np.maximum(box1[0], box2[0])
    inter_y1 = np.maximum(box1[1], box2[1])
    inter_x2 = np.minimum(box1[2], box2[2])
    inter_y2 = np.minimum(box1[3], box2[3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    
    inter = inter_w * inter_h
    
    # Area of box1
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    union = w1 * h1
    
    if union == 0:
        return -1

    # Calculating IoA (Intersection over Area), not IoU (Intersection over Union)
    # Meaning how much of the box remains inside the crop area
    iou = inter / union
    
    if iou >= 0.600088: # Threshold value
        if box1[0] >= box1[2] or box1[1] >= box1[3]:
            print('Error: Invalid box dimensions')
        
        a, b, c, d, e, f = box1
        
        # Calculate new coordinates relative to the cropped image
        boxret = [
            np.clip(a, box2[0], box2[2]) - box2[0],
            np.clip(b, box2[1], box2[3]) - box2[1],
            np.clip(c, box2[0], box2[2]) - box2[0],
            np.clip(d, box2[1], box2[3]) - box2[1],
            e, f
        ]
        
        if boxret[2] < 0 or boxret[3] < 0:
            print('Error: Negative box dimension')
            
        return boxret
    else:
        return -1

def imshow_bboxes(img, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes on an image using OpenCV.
    Used for visualization. (Instead of mmcv.imshow_bboxes)
    """
    img_copy = img.copy()
    for bbox in bboxes:
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    # Matplotlib is used to show the image (with BGR -> RGB conversion)
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.show()

# --- Main GMM Function ---

def gmm_cluster_and_crop(path, size=False, grid=False, save_path='', save=True, group=False):
    """
    Clusters objects in the image using Gaussian Mixture Model (GMM),
    crops, and saves them.
    """
    labelpath = path.replace('images', 'annotations').replace('.jpg', '.txt')
    
    # Read the image (cv2.imread instead of mmcv.imread)
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load image -> {path}")
        return None, None
        
    height, width, _ = img.shape
    
    # Read the annotation file (instead of mmcv.list_from_file)
    try:
        with open(labelpath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Annotation file not found -> {labelpath}")
        return None, None

    content = np.asarray([line.strip().split(',') for line in lines])
    
    if content.size == 0:
        # print(f"Warning: Empty annotation file -> {labelpath}")
        return img, np.array([])

    # Create grid
    gx = np.linspace(0, width, 2 * 6)
    gy = np.linspace(0, height, 2 * 4)
    mesh = np.meshgrid(gx, gy)
    mesh = np.reshape(mesh, [2, -1]).T
    
    bboxes = []
    features = []
    for i in content:
        # Ignored classes
        if int(i[5]) != 0 and int(i[5]) != 11:
            bbox_data = [float(val) for val in i[0:6]]
            if grid:
                # Calculate grid features based on object center
                center_x = bbox_data[0] + bbox_data[2] / 2
                center_y = bbox_data[1] + bbox_data[3] / 2
                temp0 = np.sqrt(np.sum(np.square(mesh - np.asarray([[center_x, center_y]])), axis=1))
                if np.max(temp0) > 0:
                    temp0 = temp0 / np.max(temp0)
                features.append(temp0)
                bboxes.append(bbox_data)
            else:
                bboxes.append(bbox_data)

    if not bboxes:
        return img, np.array([])
        
    bboxes = np.asarray(bboxes)
    features = np.asarray(features)
    
    # Convert bounding boxes to (x1, y1, x2, y2) format
    bboxes_xyxy = bboxes.copy()
    bboxes_xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes_xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3]

    # Number of clusters (n_components)
    nc = int(np.log2(len(bboxes))) + 1 if len(bboxes) > 1 else 1

    if len(bboxes) > nc:
        groups = {str(o): [] for o in range(nc)}
        
        # Gaussian Mixture Model
        if grid and len(features) > nc:
            gm = GaussianMixture(n_components=nc, random_state=0).fit(features)
            labels = gm.predict(features)
            for j, label in zip(bboxes_xyxy, labels):
                groups[str(label)].append(j)
        elif not grid and len(bboxes_xyxy) > nc:
            gm = GaussianMixture(n_components=nc, max_iter=500, random_state=0).fit(bboxes_xyxy)
            labels = gm.predict(bboxes_xyxy)
            for j, label in zip(bboxes_xyxy, labels):
                groups[str(label)].append(j)
        else: # If clustering cannot be done, put all in a single group
             groups['0'] = bboxes_xyxy.tolist()

        # Create crop boxes (cbboxes) from clusters
        cbboxes = []
        for k in groups.keys():
            temp = groups[k]
            if len(temp) > 0:
                temp = np.asarray(temp).reshape([-1, 6])
                padding = 15 if len(temp) > 3 else 40
                
                x1 = np.clip(np.min(temp[:, 0]) - padding, 0, width)
                y1 = np.clip(np.min(temp[:, 1]) - padding, 0, height)
                x2 = np.clip(np.max(temp[:, 2]) + padding, 0, width)
                y2 = np.clip(np.max(temp[:, 3]) + padding, 0, height)
                cbboxes.append([x1, y1, x2, y2])
        
        # Cropping with NumPy slicing instead of mmcv.imcrop
        patches = []
        for coord in cbboxes:
            x1, y1, x2, y2 = map(int, coord)
            patch = img[y1:y2, x1:x2]
            patches.append(patch)

        # Visualization (optional)
        # imshow_bboxes(img, np.asarray(cbboxes), color=(255, 0, 0)) # Red: Crop areas
        # imshow_bboxes(img, np.asarray(bboxes_xyxy)) # Green: Original boxes

        if save:
            if group:
                name = rand_str()
                # Save the grouped full image and merged annotations
                # cv2.imwrite instead of mmcv.imwrite
                cv2.imwrite(os.path.join(save_path, 'group', 'images', name + '.jpg'), img)
                
                anno = []
                # Add crop areas to annotations
                for ik in cbboxes:
                    anno.append([str(int(ik[0])), str(int(ik[1])), str(int(ik[2] - ik[0])), str(int(ik[3] - ik[1])), '1', '0', '0', '0'])
                
                anno = np.asarray(anno).reshape([-1, 8])
                # Also add original annotations (in bbox format)
                original_annos = np.asarray(content)[:, :8].reshape([-1, 8])
                anno = np.concatenate((anno, original_annos), 0)
                
                np.savetxt(os.path.join(save_path, 'group', 'annotations', name + '.txt'), anno, delimiter=',', fmt='%s')
            else:
                # Save each cropped patch separately
                for pt, coord in zip(patches, cbboxes):
                    if pt.size == 0: continue # Skip empty patches

                    cropboxes = []
                    for box in bboxes_xyxy:
                        ret = iouf(box, coord)
                        if ret != -1:
                            cropboxes.append(ret)
                    
                    if not cropboxes: continue # Skip patches without boxes
                    
                    # Visualization (optional)
                    # imshow_bboxes(pt, np.asarray(cropboxes))
                    
                    name = rand_str()
                    anno = []
                    for ik in cropboxes:
                        # Convert (x1,y1,x2,y2) -> (x,y,w,h) format
                        w = ik[2] - ik[0]
                        h = ik[3] - ik[1]
                        anno.append([str(int(ik[0])), str(int(ik[1])), str(int(w)), str(int(h)), '1', str(int(ik[5])), '0', '0'])
                    
                    # cv2.imwrite instead of mmcv.imwrite
                    cv2.imwrite(os.path.join(save_path, 'crop', 'images', name + '.jpg'), pt)
                    
                    anno = np.asarray(anno)
                    np.savetxt(os.path.join(save_path, 'crop', 'annotations', name + '.txt'), anno, delimiter=',', fmt='%s')

    return img, bboxes_xyxy


# --- Main Execution Block ---

if __name__ == '__main__':
    path = '/home/any1/fd/visdrone/VisDrone2019-DET-val'
    save_path = '/home/any1/fd/gmm_data'

    # Ensure output paths exist
    os.makedirs(os.path.join(save_path, 'group', 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'group', 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'crop', 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'crop', 'annotations'), exist_ok=True)

    paths = glob.glob(os.path.join(path, 'images', '*.jpg'))
    
    # paths = paths[:10] # Run only the first 10 images for testing

    for p in tqdm(paths, desc="Processing images"):
        gmm_cluster_and_crop(p, save=True, group=True, save_path=save_path) # group=False -> Saves cropped patches
        # gmm_cluster_and_crop(p, save=True, group=True, save_path=save_path) # group=True -> Saves the full image

    print("Process completed.")
