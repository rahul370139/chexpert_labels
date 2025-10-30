"""
Create symbolic links for 1000 images from radiology_report to chexagent_chexpert_eval/data/images/
"""

import os
from pathlib import Path

def create_image_links():
    """Create symbolic links for all images in the 1000-image list."""
    
    # Paths
    image_list_path = Path("data/image_list_1000.txt")
    radiology_report_path = Path("/Users/bilbouser/radiology_report")
    target_dir = Path("data/images_1000")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Read image list
    with open(image_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"📁 Creating symbolic links for {len(image_paths)} images...")
    
    created_count = 0
    skipped_count = 0
    
    for i, rel_path in enumerate(image_paths):
        source_path = radiology_report_path / rel_path
        target_path = target_dir / Path(rel_path).name
        
        if source_path.exists():
            if not target_path.exists():
                try:
                    os.symlink(source_path, target_path)
                    created_count += 1
                except OSError as e:
                    print(f"⚠️  Failed to create link for {rel_path}: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
        else:
            print(f"⚠️  Source image not found: {source_path}")
            skipped_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)} images...")
    
    print(f"✅ Created {created_count} symbolic links")
    print(f"⚠️  Skipped {skipped_count} images")
    
    # Create new image list with correct paths
    new_image_list_path = Path("data/image_list_1000_corrected.txt")
    with open(new_image_list_path, 'w') as f:
        for rel_path in image_paths:
            target_path = target_dir / Path(rel_path).name
            if target_path.exists():
                f.write(f"{target_path}\n")
    
    print(f"📝 Created corrected image list: {new_image_list_path}")

if __name__ == "__main__":
    create_image_links()
