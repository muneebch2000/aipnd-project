import torch
from torchvision import models
from PIL import Image
import numpy as np
import argparse
import json

# ------------------------
# Load checkpoint
# ------------------------
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = getattr(models, checkpoint['arch'])(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

# ------------------------
# Process image
# ------------------------
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # Resize
    if image.size[0] < image.size[1]:
        image.thumbnail((256, 256**10))
    else:
        image.thumbnail((256**10, 256))

    # Center crop
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Normalize
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Channel first
    np_image = np_image.transpose((2, 0, 1))

    return torch.tensor(np_image, dtype=torch.float32)

# ------------------------
# Predict
# ------------------------
def predict(image_path, model, topk=5, device='cpu'):
    model.to(device)
    model.eval()

    image = process_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)

    top_p, top_class = ps.topk(topk, dim=1)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i.item()] for i in top_class[0]]

    return top_p[0].tolist(), classes

# ------------------------
# Main CLI
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained network checkpoint.")
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('checkpoint', help='Path to model checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', default=None, help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.image_path, model, args.top_k, device)

    # Map classes to names if JSON provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(c, c) for c in classes]

    # Display results
    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob:.3f}")
