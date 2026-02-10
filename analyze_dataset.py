
import json
import os

file_path = '/Users/jamesyang/Projects/LossZero/www.acmeai.tech ODataset 1 - Motorcycle Night Ride Dataset/COCO_motorcycle (pixel).json'

def analyze_coco(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Analyzing {path}...")
    
    # We load only the keys to avoid memory issues if possible, 
    # but standard json.load is often fine for 320MB if memory is sufficient.
    with open(path, 'r') as f:
        data = json.load(f)
    
    print("\n--- Project Info ---")
    if 'info' in data:
        print(json.dumps(data['info'], indent=2))
        
    print("\n--- Categories ---")
    categories = data.get('categories', [])
    for cat in categories:
        print(f"ID: {cat['id']}, Name: {cat['name']}")
        
    print("\n--- Statistics ---")
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")
    
    cat_counts = {}
    for ann in annotations:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
        
    print("\n--- Annotation Counts per Category ---")
    for cat in categories:
        count = cat_counts.get(cat['id'], 0)
        print(f"Category '{cat['name']}' (ID {cat['id']}): {count} objects")

if __name__ == "__main__":
    analyze_coco(file_path)
