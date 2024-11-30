import argparse
import torch
from torchvision import transforms
from PIL import Image
import json
from model import create_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Prediction Script")
    parser.add_argument('--checkpoint', type=str, default='checkpoint_model.pth', help="Path to the saved model checkpoint")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image for prediction")
    parser.add_argument('--json_path', type=str, required=True, help="Path to JSON file with class labels")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top predictions to display")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")
    return parser.parse_args()

def preprocess_image(image_path):
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    return transformations(image)

def get_predictions(model, image_tensor, top_k):
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(image_tensor)
    probabilities = torch.exp(log_probs)
    top_probs, top_classes = probabilities.topk(top_k, dim=1)

    return top_probs.cpu().numpy(), top_classes.cpu().numpy()

def main():
    args = parse_arguments()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model = create_model()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # Process image and predict
    image_tensor = preprocess_image(args.image_path)
    top_probs, top_classes = get_predictions(model, image_tensor, args.top_k)

    # Load class labels
    with open(args.json_path, 'r') as f:
        class_names = json.load(f)
    
    print("Top Predictions:")
    for i in range(args.top_k):
        class_name = class_names.get(str(top_classes[0][i] + 1), "Unknown")
        print(f"{class_name}: {top_probs[0][i] * 100:.2f}%")

if __name__ == '__main__':
    main()
