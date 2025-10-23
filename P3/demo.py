"""
Interactive demo for Image Captioning
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import config
from inference import CaptionGenerator


def create_demo_images():
    """Create some demo images for testing"""
    os.makedirs('demo_images', exist_ok=True)
    
    # Create sample images with different colors
    colors_and_descriptions = [
        ('red', 'demo_red.jpg'),
        ('blue', 'demo_blue.jpg'),
        ('green', 'demo_green.jpg'),
        ('yellow', 'demo_yellow.jpg'),
        ('purple', 'demo_purple.jpg'),
    ]
    
    for color, filename in colors_and_descriptions:
        filepath = os.path.join('demo_images', filename)
        if not os.path.exists(filepath):
            img = Image.new('RGB', (224, 224), color=color)
            img.save(filepath)
    
    print(f"Demo images created in 'demo_images' directory")
    return 'demo_images'


def visualize_multiple_captions(generator, image_paths):
    """
    Visualize multiple images with their captions
    
    Args:
        generator: CaptionGenerator instance
        image_paths: List of image paths
    """
    n_images = len(image_paths)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, img_path in enumerate(image_paths):
        if idx >= len(axes):
            break
        
        try:
            # Generate caption
            caption = generator.generate_caption(img_path, show_image=False)
            
            # Display image
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(caption, fontsize=12, wrap=True, pad=10)
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error: {str(e)}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('captions_demo.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'captions_demo.png'")
    plt.show()


def interactive_demo():
    """
    Interactive demo for caption generation
    """
    print("=" * 70)
    print("Image Captioning - Interactive Demo")
    print("=" * 70)
    
    # Check if model exists
    model_path = os.path.join(config.MODEL_DIR, 'model_config.json')
    if not os.path.exists(model_path):
        print("\nModel not found! Training a model first...")
        print("This will use sample data for demonstration.\n")
        
        # Run training
        from train import main as train_main
        train_main()
        print("\n" + "=" * 70)
    
    # Load model
    try:
        generator = CaptionGenerator()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\n" + "=" * 70)
    print("Model loaded successfully!")
    print("=" * 70)
    
    # Check for demo images
    demo_dir = 'demo_images'
    if not os.path.exists(demo_dir):
        print("\nCreating demo images...")
        create_demo_images()
    
    # Get list of available images
    available_images = []
    
    # Check demo images
    if os.path.exists(demo_dir):
        demo_images = [os.path.join(demo_dir, f) for f in os.listdir(demo_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        available_images.extend(demo_images)
    
    # Check training images
    if os.path.exists(config.IMAGES_DIR):
        train_images = [os.path.join(config.IMAGES_DIR, f) 
                       for f in os.listdir(config.IMAGES_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        available_images.extend(train_images[:5])  # Limit to 5 training images
    
    if not available_images:
        print("\nNo images found for demo.")
        print("Please add images to 'demo_images' directory or train with your own data.")
        return
    
    print(f"\nFound {len(available_images)} images for demo")
    
    # Menu
    while True:
        print("\n" + "=" * 70)
        print("Options:")
        print("  1. Generate caption for a specific image")
        print("  2. Generate captions for all available images")
        print("  3. Use beam search (slower but more accurate)")
        print("  4. Visualize multiple images with captions")
        print("  5. Exit")
        print("=" * 70)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\nAvailable images:")
            for i, img_path in enumerate(available_images, 1):
                print(f"  {i}. {os.path.basename(img_path)}")
            
            try:
                idx = int(input("\nEnter image number: ").strip()) - 1
                if 0 <= idx < len(available_images):
                    img_path = available_images[idx]
                    print(f"\nProcessing: {os.path.basename(img_path)}")
                    caption = generator.generate_caption(img_path, show_image=True)
                    print(f"Caption: {caption}")
                else:
                    print("Invalid image number")
            except ValueError:
                print("Invalid input")
        
        elif choice == '2':
            print(f"\nGenerating captions for {len(available_images)} images...")
            for img_path in available_images:
                try:
                    caption = generator.generate_caption(img_path, show_image=False)
                    print(f"\n{os.path.basename(img_path)}:")
                    print(f"  {caption}")
                except Exception as e:
                    print(f"\nError with {os.path.basename(img_path)}: {e}")
        
        elif choice == '3':
            print("\nAvailable images:")
            for i, img_path in enumerate(available_images, 1):
                print(f"  {i}. {os.path.basename(img_path)}")
            
            try:
                idx = int(input("\nEnter image number: ").strip()) - 1
                if 0 <= idx < len(available_images):
                    img_path = available_images[idx]
                    print(f"\nProcessing with beam search: {os.path.basename(img_path)}")
                    print("(This may take a moment...)")
                    caption = generator.generate_with_beam_search(
                        img_path, beam_width=3, show_image=True
                    )
                    print(f"Caption: {caption}")
                else:
                    print("Invalid image number")
            except ValueError:
                print("Invalid input")
        
        elif choice == '4':
            n_images = min(6, len(available_images))
            print(f"\nVisualizing {n_images} images...")
            visualize_multiple_captions(generator, available_images[:n_images])
        
        elif choice == '5':
            print("\nThank you for using Image Captioning demo!")
            break
        
        else:
            print("Invalid choice. Please try again.")


def main():
    """Main demo function"""
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

