import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
import huggingface_hub
from hpsv2.utils import root_path, hps_version_map

warnings.filterwarnings("ignore", category=UserWarning)

model_dict = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val


def convert_rgba_to_rgb_with_white_bg(image):
    """Convert RGBA image to RGB format with white background"""
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste RGBA image onto white background using alpha channel as mask
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode != 'RGB':
        # If not RGB or RGBA, convert to RGB
        return image.convert('RGB')
    else:
        # Already in RGB format
        return image


def score(img_path: str, prompt: str, cp: str = None, hps_version: str = "v2.0") -> float:
    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    # Check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if cp is None:
        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Process the image
        pil_image = Image.open(img_path)
        pil_image = convert_rgba_to_rgb_with_white_bg(pil_image)
        image = preprocess_val(pil_image).unsqueeze(0).to(device=device, non_blocking=True)
        
        # Process the prompt
        text = tokenizer([prompt]).to(device=device, non_blocking=True)
        
        # Calculate the HPS
        with torch.cuda.amp.autocast():
            outputs = model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            hps_score = torch.diagonal(logits_per_image).cpu().numpy()
    
    return float(hps_score[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate HPS score for a single image and prompt')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt to evaluate against')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--hps-version', type=str, default='v2.0', help='HPS version to use')

    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist.")
        exit(1)
    
    if not os.path.isfile(args.image):
        print(f"Error: '{args.image}' is not a file.")
        exit(1)
    
    # Calculate and print the score
    hps_score = score(args.image, args.prompt, args.checkpoint, args.hps_version)
    print(hps_score)
    