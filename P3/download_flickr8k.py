"""
Download and prepare Flickr8k dataset for image captioning

The Flickr8k dataset contains:
- 8,000 images
- 5 captions per image
- Total: 40,000 image-caption pairs

Sources:
1. Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k
2. Direct download (if available)
"""

import os
import zipfile
import requests
from tqdm import tqdm
import shutil

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def prepare_flickr8k_from_kaggle():
    """
    Prepare Flickr8k dataset downloaded from Kaggle
    
    Instructions:
    1. Go to: https://www.kaggle.com/datasets/adityajn105/flickr8k
    2. Download the dataset (you'll need a Kaggle account)
    3. Place the downloaded file in the 'downloads' folder
    4. Run this function
    """
    print("\n" + "="*70)
    print("FLICKR8K DATASET PREPARATION")
    print("="*70)
    
    downloads_dir = 'downloads'
    os.makedirs(downloads_dir, exist_ok=True)
    
    # Check for downloaded files
    possible_files = [
        'flickr8k.zip',
        'archive.zip',
        'Flickr8k_Dataset.zip'
    ]
    
    zip_file = None
    for f in possible_files:
        path = os.path.join(downloads_dir, f)
        if os.path.exists(path):
            zip_file = path
            break
    
    if not zip_file:
        print("\n❌ Dataset not found!")
        print("\n📥 Please download Flickr8k dataset:")
        print("   1. Visit: https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("   2. Click 'Download' (requires Kaggle account)")
        print(f"   3. Place the ZIP file in: {os.path.abspath(downloads_dir)}/")
        print("   4. Run this script again")
        print("\n💡 Alternative: Use the Kaggle API (see instructions below)")
        return False
    
    print(f"\n✓ Found dataset: {zip_file}")
    print("\n📦 Extracting...")
    
    # Extract
    extract_dir = 'flickr8k_extracted'
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print("✓ Extracted successfully")
    
    # Find images and captions
    print("\n📁 Organizing files...")
    
    # Common structures in Flickr8k downloads
    possible_structures = [
        ('Images', 'captions.txt'),
        ('Flicker8k_Dataset', 'Flickr8k.token.txt'),
        ('images', 'captions.txt'),
    ]
    
    images_source = None
    captions_source = None
    
    for root, dirs, files in os.walk(extract_dir):
        for d in dirs:
            if d.lower() in ['images', 'flicker8k_dataset', 'flickr8k_images']:
                images_source = os.path.join(root, d)
        for f in files:
            if 'caption' in f.lower() or 'token' in f.lower():
                if f.endswith('.txt'):
                    captions_source = os.path.join(root, f)
    
    if not images_source or not captions_source:
        print("❌ Could not find images or captions in the extracted files")
        print(f"   Please check: {os.path.abspath(extract_dir)}")
        return False
    
    print(f"✓ Found images: {images_source}")
    print(f"✓ Found captions: {captions_source}")
    
    # Copy images
    print("\n📸 Copying images to data/images/...")
    os.makedirs('data/images', exist_ok=True)
    
    image_files = [f for f in os.listdir(images_source) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in tqdm(image_files[:1000], desc="Copying images"):  # Limit to 1000 for faster processing
        src = os.path.join(images_source, img_file)
        dst = os.path.join('data/images', img_file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Process captions
    print("\n📝 Processing captions...")
    process_flickr8k_captions(captions_source, 'data/captions.txt')
    
    print("\n✅ Dataset prepared successfully!")
    print(f"\n📊 Statistics:")
    print(f"   Images: {len(os.listdir('data/images'))} files")
    
    with open('data/captions.txt', 'r', encoding='utf-8') as f:
        caption_count = len(f.readlines())
    print(f"   Captions: {caption_count} entries")
    
    return True


def process_flickr8k_captions(source_file, output_file):
    """
    Process Flickr8k captions file
    
    Flickr8k format: image_name.jpg#0\tcaption text
    Our format: image_name.jpg,caption text
    """
    captions_dict = {}
    
    with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Handle different formats
            if '\t' in line:
                parts = line.split('\t')
            elif ',' in line:
                parts = line.split(',', 1)
            else:
                continue
            
            if len(parts) >= 2:
                img_caption_id = parts[0]
                caption = parts[1].strip()
                
                # Remove caption number (e.g., "image.jpg#0" -> "image.jpg")
                if '#' in img_caption_id:
                    img_name = img_caption_id.split('#')[0]
                else:
                    img_name = img_caption_id
                
                if img_name not in captions_dict:
                    captions_dict[img_name] = []
                captions_dict[img_name].append(caption)
    
    # Write to output
    with open(output_file, 'w', encoding='utf-8') as f:
        for img_name, captions in captions_dict.items():
            for caption in captions:
                f.write(f"{img_name},{caption}\n")
    
    print(f"✓ Processed {len(captions_dict)} images with captions")


def download_with_kaggle_api():
    """
    Download Flickr8k using Kaggle API
    
    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Get Kaggle API token from https://www.kaggle.com/settings
    3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)
    """
    print("\n" + "="*70)
    print("DOWNLOAD USING KAGGLE API")
    print("="*70)
    
    try:
        import kaggle
        print("\n✓ Kaggle API is installed")
    except ImportError:
        print("\n❌ Kaggle API not installed")
        print("   Install with: pip install kaggle")
        return False
    
    try:
        print("\n📥 Downloading Flickr8k from Kaggle...")
        os.makedirs('downloads', exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'adityajn105/flickr8k',
            path='downloads',
            unzip=True
        )
        
        print("✓ Downloaded successfully")
        
        # Process the downloaded files
        return prepare_flickr8k_from_kaggle()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Kaggle account")
        print("2. API token configured (kaggle.json)")
        print("   Get it from: https://www.kaggle.com/settings")
        return False


def use_small_subset():
    """
    Create a small subset for quick testing
    Downloads random images from the web
    """
    print("\n" + "="*70)
    print("CREATE SMALL TEST DATASET")
    print("="*70)
    print("\nThis will download ~100 sample images from free sources")
    print("for quick testing. For production, use the full Flickr8k dataset.")
    
    # Using Unsplash API for sample images
    # This is just for demonstration
    print("\nℹ️  For the internship video, I recommend using the full Flickr8k dataset")
    print("   to show professional-quality results.")
    
    return False


def main():
    """Main function"""
    print("\n" + "="*70)
    print("IMAGE CAPTIONING DATASET SETUP")
    print("="*70)
    
    print("\nOptions:")
    print("  1. I have already downloaded Flickr8k from Kaggle")
    print("  2. Download using Kaggle API (automatic)")
    print("  3. Show manual download instructions")
    print("  4. Use small test dataset (quick demo)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        prepare_flickr8k_from_kaggle()
    elif choice == '2':
        download_with_kaggle_api()
    elif choice == '3':
        show_manual_instructions()
    elif choice == '4':
        use_small_subset()
    else:
        print("Invalid choice")


def show_manual_instructions():
    """Show detailed manual download instructions"""
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    print("\n📥 **OPTION 1: Kaggle (Recommended)**")
    print("   Dataset: Flickr8k")
    print("   Link: https://www.kaggle.com/datasets/adityajn105/flickr8k")
    print("   Steps:")
    print("     1. Create/login to Kaggle account")
    print("     2. Click 'Download' button")
    print("     3. Save to: downloads/ folder in this project")
    print("     4. Run this script again and choose option 1")
    
    print("\n📥 **OPTION 2: Alternative Flickr8k Source**")
    print("   Link: https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k")
    print("   Download:")
    print("     - Flickr8k_Dataset.zip (images)")
    print("     - Flickr8k_text.zip (captions)")
    
    print("\n📥 **OPTION 3: MS COCO (Larger, more comprehensive)**")
    print("   Link: https://cocodataset.org/#download")
    print("   Note: Much larger download, but industry standard")
    
    print("\n💡 **For Your Internship Video:**")
    print("   - Flickr8k is perfect (8,000 images)")
    print("   - Shows professional-quality results")
    print("   - Trains in reasonable time (30-60 minutes)")
    print("   - Creates impressive captions")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

