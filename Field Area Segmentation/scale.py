#Load all dependencies we need
import numpy as np
import os
import json
from shapely.geometry import Polygon
import pandas as pd
import itertools
import tifffile as tiff
from skimage import measure
from tqdm import tqdm
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


#initialize SAM
sam = sam_model_registry[vit_h](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(model=sam,
    points_per_side=50,
    pred_iou_thresh=0.93,
    stability_score_thresh=0.94,
    crop_n_layers=2,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100)  # Requires open-cv to run post-processing)


#Polygon Extraction
def extract_polygons(segmentation, num_points_threshold, min_area):
    contours = measure.find_contours(segmentation, 0.5)
    contours = [contour for contour in contours if len(contour) >= num_points_threshold]
    contours = [contour for contour in contours if Polygon(contour).area >= min_area]
    polygons = []
    for contour in contours:
        polygon = []
        for point in contour:
            polygon.append(float(point[1]))  # X coordinate
            polygon.append(float(point[0]))  # Y coordinate
        polygons.append(polygon)
    return polygons


#Save Results to JSON
def save_results_to_json(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


#Process a Single Image
def process_images(image_dir, num_points_threshold, min_area):
    images = []
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith('.tif'):
            image_path = os.path.join(image_dir, filename)
            image = tiff.imread(image_path)
            band_red = image[:, :, 1]  
            band_green = image[:, :, 3] 
            band_blue = image[:, :, 6]  

            rgb_image = np.dstack((band_red, band_green, band_blue))
            image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
            segmentations = mask_generator.generate(np.array(image))  # Replace with actual function call to your segmentation model
            
            annotations = []
            for segmentation in segmentations:
                polygons = extract_polygons(segmentation['segmentation'], num_points_threshold, min_area)
                for polygon in polygons:
                    annotations.append({
                        "class": "field",
                        "segmentation": polygon
                    })
            images.append({
                "file_name": filename,
                "annotations": annotations
            })
    return {"images": images}

#Process Multiple Images
def process_images(image_dir, output_dir):
    channel_indices = list(range(12))
    combinations = list(itertools.combinations(channel_indices, 3))
    
    for combo in combinations:
        results = {"images": []}
        for filename in tqdm(os.listdir(image_dir)):
            if filename.endswith('.tif'):
                image_path = os.path.join(image_dir, filename)
                image_result = process_image(image_path, combo)
                results["images"].append(image_result)
        
        output_file = os.path.join(output_dir, f'segmentation_results_combo_{combo[0]}_{combo[1]}_{combo[2]}.json')
        save_results_to_json(results, output_file)
        print(f"Results saved to {output_file}")


#Set Directories and Process Images
image_dir = 'competition\solafune\Field Area Segmentation\data\train_images'
output_file = 'results.json'
Number_of_point = 3
size = 10

# Process images and save results
results = process_images(image_dir, Number_of_point, size)
save_results_to_json(results, output_file)
print(f"Results saved to {output_file}")


#Evaluation Model Performance for all combinations
## Code from https://solafune.com/competitions/d91572d9-1680-4b9e-b372-25e71093f81a?menu=discussion&tab=&modal=%22%22&sort_by=total_vote&page=1
def getIOU(polygon1: Polygon, polygon2: Polygon):
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    if union == 0:
        return 0
    return intersection / union


def compute_pq(gt_polygons: list, pred_polygons: list, iou_threshold=0.5):
    matched_instances = {}
    gt_matched = np.zeros(len(gt_polygons))
    pred_matched = np.zeros(len(pred_polygons))

    gt_matched = np.zeros(len(gt_polygons))
    pred_matched = np.zeros(len(pred_polygons))
    for gt_idx, gt_polygon in enumerate(gt_polygons):
        best_iou = iou_threshold
        best_pred_idx = None
        for pred_idx, pred_polygon in enumerate(pred_polygons):
            # if gt_matched[gt_idx] == 1 or pred_matched[pred_idx] == 1:
            #     continue
            try:
                iou = getIOU(gt_polygon, pred_polygon)
            except:
                iou = 0
                print('Error Polygon -> iou is 0')
                
            if iou == 0:
                continue
            
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        if best_pred_idx is not None:
            matched_instances[(gt_idx, best_pred_idx)] = best_iou
            gt_matched[gt_idx] = 1
            pred_matched[best_pred_idx] = 1

    
    sq_sum = sum(matched_instances.values())
    num_matches = len(matched_instances)
    sq = sq_sum / num_matches if num_matches else 0
    rq = num_matches / ((len(gt_polygons) + len(pred_polygons))/2.0) if (gt_polygons or pred_polygons) else 0
    pq = sq * rq

    return pq, sq, rq

## Code adapted from https://solafune.com/competitions/d91572d9-1680-4b9e-b372-25e71093f81a?menu=discussion&tab=&modal=%22%22&sort_by=total_vote&page=1
with open('../data/train_annotation.json') as f:
    gts = json.load(f)

with open('results.json') as f:
    val_json = json.load(f)

scores, files = [], []

for i ,(_image_pred) in enumerate(val_json['images']):
    
    fname = _image_pred['file_name']
    annos_pred = _image_pred['annotations']
    print(fname)
    for j ,(_image_gt) in enumerate(gts['images']):
        if _image_gt['file_name'] == fname:
            fname_gt = _image_gt['file_name']
            break
    
    annos_gt = _image_gt['annotations']
    
    print(f'File:{fname} - {fname_gt} Num GT: {len(annos_gt)}, Num Pred: {len(annos_pred)}')
    
    polygons_gt, polygons_pred = [], []
    for anno in annos_gt:
        _polys = []
        for ii, (x, y) in enumerate(zip(anno['segmentation'][::2], anno['segmentation'][1::2])):
            _polys.append((x, y))
        polygons_gt.append(Polygon(_polys))
        
    for anno in annos_pred:
        _polys = []
        for ii, (x, y) in enumerate(zip(anno['segmentation'][::2], anno['segmentation'][1::2])):
            _polys.append((x, y))
        polygons_pred.append(Polygon(_polys))
    
    
    pq, sq, rq = compute_pq(polygons_gt, polygons_pred)
    print(f'File:{fname} PQ: {pq:.4f}, SQ: {sq:.4f}, RQ: {rq:.4f} Num: {len(polygons_gt)}')
    
    scores.append([pq, sq, rq])
    files.append(fname)
    
df = pd.DataFrame(scores, columns=['PQ', 'SQ', 'RQ'], index=files)
df['file'] = files

metrisc = df['PQ'].mean()
print(f'Mean PQ: {metrisc:.4f}')