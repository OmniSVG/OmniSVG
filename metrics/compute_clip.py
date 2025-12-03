import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def calculate_clip_score(image_path, text_path):
    """
    Calculate CLIP similarity score between an image and text prompt
    
    Args:
        image_path: Path to the image file
        text_path: Path to the text file containing the prompt
    
    Returns:
        CLIP similarity score (cosine similarity)
    """
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("/mnt/jfs-test/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("/mnt/jfs-test/clip-vit-base-patch32")
    
    # Check for available GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Load text prompt
    with open(text_path, 'r', encoding='utf-8') as f:
        caption = f.read().strip()
    
    # Process image and text
    inputs = processor(
        text=[caption], 
        images=image, 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    # Calculate cosine similarity
    with torch.no_grad():
        # Get image and text feature vectors
        image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = model.get_text_features(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )
        
        # L2 normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        similarity = (image_features @ text_features.T).item()
    
    return similarity

if __name__ == "__main__":
    text_path = input("Enter path to text file: ")
    image_path = input("Enter path to image file: ")
    
    if not os.path.isfile(text_path):
        print(f"Error: Text file '{text_path}' does not exist.")
        exit(1)
    
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        exit(1)
    
    score = calculate_clip_score(image_path, text_path)
    print(f"CLIP Score: {score:.4f}")
```

If you prefer using command line arguments instead of input prompts:

```python
import os
import sys
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def calculate_clip_score(image_path, text_path):
    """
    Calculate CLIP similarity score between an image and text prompt
    """
    model = CLIPModel.from_pretrained("/mnt/jfs-test/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("/mnt/jfs-test/clip-vit-base-patch32")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    
    with open(text_path, 'r', encoding='utf-8') as f:
        caption = f.read().strip()
    
    inputs = processor(
        text=[caption], 
        images=image, 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = model.get_text_features(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).item()
    
    return similarity

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <text_path> <image_path>")
        exit(1)
    
    text_path = sys.argv[1]
    image_path = sys.argv[2]
    
    if not os.path.isfile(text_path):
        print(f"Error: Text file '{text_path}' does not exist.")
        exit(1)
    
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        exit(1)
    
    score = calculate_clip_score(image_path, text_path)
    print(f"{score:.4f}")

    