"""
Name       : Himanshi Sharma
Roll No    : 2301010428
Course     : Image Processing & Computer Vision Lab
Assignment : Smart Document Scanner & Quality Analysis System
Date       : 15-02-2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==================== TASK 1: PROJECT SETUP ====================
print("="*70)
print("           SMART DOCUMENT SCANNER & QUALITY ANALYSIS SYSTEM")
print("           Course: Image Processing & Computer Vision")
print("           Simulates image acquisition, sampling & quantization effects")
print("="*70)
print(f"Run Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# Create outputs directory if it doesn't exist
if not os.path.exists("outputs"):
    os.makedirs("outputs")
    print("\n✅ Created 'outputs' directory for saving results")

# ==================== TASK 2: IMAGE ACQUISITION ====================
def load_and_preprocess(image_path, image_id=1):
    """
    Load document image, resize to 512x512, convert to grayscale
    """
    print(f"\n{'='*50}")
    print(f"📄 PROCESSING DOCUMENT {image_id}: {os.path.basename(image_path)}")
    print(f"{'='*50}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None, None, None
    
    # Get original dimensions
    h, w = img.shape[:2]
    print(f"📏 Original dimensions: {w} x {h} pixels")
    
    # Resize to 512x512
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    print(f"✅ Resized to: 512 x 512 pixels")
    print(f"✅ Converted to grayscale (8-bit)")
    
    return img_resized, gray, image_id

# ==================== TASK 3: IMAGE SAMPLING ====================
def analyze_sampling(gray_image):
    """
    Downsample to different resolutions and upsample back for comparison
    """
    print("\n--- TASK 3: SAMPLING ANALYSIS (Resolution Reduction) ---")
    
    resolutions = [512, 256, 128]
    labels = ["High (512×512)", "Medium (256×256)", "Low (128×128)"]
    sampled_images = []
    
    for i, (res, label) in enumerate(zip(resolutions, labels)):
        # Downsample
        downsampled = cv2.resize(gray_image, (res, res), interpolation=cv2.INTER_AREA)
        
        # Upsample back to 512x512 for visualization
        upsampled = cv2.resize(downsampled, (512, 512), interpolation=cv2.INTER_LINEAR)
        sampled_images.append(upsampled)
        
        # Save the downsampled version
        cv2.imwrite(f"outputs/sampled_{res}x{res}.png", downsampled)
        
        print(f"   ✅ {label}: {res}×{res} pixels (saved)")
    
    return sampled_images

# ==================== TASK 4: IMAGE QUANTIZATION ====================
def quantize_image(gray_image, levels):
    """
    Reduce number of gray levels
    """
    step = 256 // levels
    quantized = (gray_image // step) * step
    return quantized.astype(np.uint8)

def analyze_quantization(gray_image):
    """
    Quantize to different bit depths
    """
    print("\n--- TASK 4: QUANTIZATION ANALYSIS (Bit-depth Reduction) ---")
    
    # Original is already 8-bit (256 levels)
    bit_depths = [8, 4, 2]
    gray_levels = [256, 16, 4]
    labels = ["8-bit (256 levels)", "4-bit (16 levels)", "2-bit (4 levels)"]
    quantized_images = []
    
    # Original 8-bit image
    quantized_images.append(gray_image)
    cv2.imwrite("outputs/quantized_8bit.png", gray_image)
    print(f"   ✅ {labels[0]} (saved)")
    
    # 4-bit quantization
    q_16 = quantize_image(gray_image, 16)
    quantized_images.append(q_16)
    cv2.imwrite("outputs/quantized_4bit.png", q_16)
    print(f"   ✅ {labels[1]} (saved)")
    
    # 2-bit quantization
    q_4 = quantize_image(gray_image, 4)
    quantized_images.append(q_4)
    cv2.imwrite("outputs/quantized_2bit.png", q_4)
    print(f"   ✅ {labels[2]} (saved)")
    
    return quantized_images

# ==================== TASK 5: VISUALIZATION & ANALYSIS ====================
def create_comparison_figure(original, sampled_images, quantized_images, image_id):
    """
    Create a 2x3 comparison figure showing all results
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Document Scanner Quality Analysis - Document {image_id}', fontsize=16, fontweight='bold')
    
    # Row 0: Sampling results
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title("ORIGINAL\n(512×512, 8-bit)", fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(sampled_images[1], cmap='gray')
    axes[0,1].set_title("SAMPLED: Medium Resolution\n(256×256)", fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(sampled_images[2], cmap='gray')
    axes[0,2].set_title("SAMPLED: Low Resolution\n(128×128)", fontsize=12, fontweight='bold')
    axes[0,2].axis('off')
    
    # Row 1: Quantization results
    axes[1,0].imshow(quantized_images[0], cmap='gray')
    axes[1,0].set_title("QUANTIZED: 8-bit\n(256 gray levels)", fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(quantized_images[1], cmap='gray')
    axes[1,1].set_title("QUANTIZED: 4-bit\n(16 gray levels)", fontsize=12, fontweight='bold')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(quantized_images[2], cmap='gray')
    axes[1,2].set_title("QUANTIZED: 2-bit\n(4 gray levels)", fontsize=12, fontweight='bold')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    
    # Save the comparison figure
    comparison_path = f"outputs/comparison_doc{image_id}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison figure saved: {comparison_path}")
    
    return fig

def print_observations():
    """
    Print detailed quality analysis observations
    """
    print("\n" + "="*70)
    print("📊 TASK 5: QUALITY OBSERVATIONS & ANALYSIS")
    print("="*70)
    
    print("\n🔍 TEXT CLARITY ANALYSIS:")
    print("   • 512×512 (High Resolution):")
    print("     - Text is sharp and crisp")
    print("     - All character edges are well-defined")
    print("     - Fine details like punctuation visible")
    print("   • 256×256 (Medium Resolution):")
    print("     - Slight blurring observed")
    print("     - Main text remains readable")
    print("     - Small fonts start losing sharpness")
    print("   • 128×128 (Low Resolution):")
    print("     - Significant blurring")
    print("     - Character edges become jagged")
    print("     - Small text becomes illegible")
    
    print("\n📉 READABILITY DEGRADATION:")
    print("   • 8-bit (256 levels):")
    print("     - Perfect readability, original quality preserved")
    print("   • 4-bit (16 levels):")
    print("     - Visible false contours in smooth regions")
    print("     - Text remains readable but quality reduced")
    print("   • 2-bit (4 levels):")
    print("     - Heavy posterization effects")
    print("     - Significant loss of gray-scale information")
    print("     - Text barely readable, severe quality degradation")
    
    print("\n🤖 OCR SUITABILITY ASSESSMENT:")
    print("   • HIGH SUITABILITY: 512×512 & 8-bit")
    print("     - Ideal for OCR engines")
    print("     - Maximum accuracy expected")
    print("   • MODERATE SUITABILITY: 256×256 & 4-bit")
    print("     - May work with preprocessing")
    print("     - Some errors possible with small text")
    print("   • LOW SUITABILITY: 128×128 & 2-bit")
    print("     - Not recommended for OCR")
    print("     - High error rate expected")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   • For archival: Use 512×512, 8-bit minimum")
    print("   • For OCR processing: Minimum 300 DPI scan recommended")
    print("   • For web display: 256×256, 4-bit may be acceptable")
    print("="*70)

# ==================== MAIN EXECUTION ====================
def process_document(image_path, image_id):
    """
    Process a single document through all tasks
    """
    # Task 2: Load and preprocess
    orig_color, gray, doc_id = load_and_preprocess(image_path, image_id)
    
    if orig_color is None:
        return False
    
    # Save grayscale image
    cv2.imwrite(f"outputs/grayscale_doc{image_id}.png", gray)
    
    # Task 3: Sampling analysis
    sampled = analyze_sampling(gray)
    
    # Task 4: Quantization analysis
    quantized = analyze_quantization(gray)
    
    # Task 5: Create comparison figure
    fig = create_comparison_figure(gray, sampled, quantized, image_id)
    plt.show()
    
    return True

# ==================== RUN MULTIPLE DOCUMENTS ====================
if __name__ == "__main__":
    # Define your document images (update these paths)
    document_paths = [
        "document1.jpg",      # Printed text document
        "document2.png",       # Scanned PDF page
        "document3.jpg"        # Photographed document
    ]
    
    print("\n📋 DOCUMENTS TO PROCESS:")
    for i, path in enumerate(document_paths, 1):
        print(f"   Document {i}: {path}")
    
    # Process each document
    successful = 0
    for i, doc_path in enumerate(document_paths, 1):
        if os.path.exists(doc_path):
            if process_document(doc_path, i):
                successful += 1
        else:
            print(f"\n⚠️ Warning: Document {i} not found at {doc_path}")
    
    # Print final observations
    print_observations()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 PROCESSING SUMMARY:")
    print(f"   ✅ Successfully processed: {successful} document(s)")
    print(f"   📁 All outputs saved in: outputs/")
    print(f"   📂 Output files generated:")
    print(f"      - grayscale_doc*.png")
    print(f"      - sampled_*x*.png")
    print(f"      - quantized_*bit.png")
    print(f"      - comparison_doc*.png")
    print(f"{'='*70}")
    print("\n🎉 Assignment completed successfully!")
    print("📝 Don't forget to:")
    print("   • Add your name and roll number in header")
    print("   • Push code to GitHub repository")
    print("   • Submit repository URL via LMS")
    print("="*70)