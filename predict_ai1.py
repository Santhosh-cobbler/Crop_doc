import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ['corn', 'not_corn']  # ImageFolder sorts alphabetically

# ── Load AI_1 Model ───────────────────────────────────────────────────────────
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 2)
)
model.load_state_dict(torch.load("best_model_ai1.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    confidence = probs[pred_idx].item() * 100

    print(f"\n{'='*40}")
    print(f"  Image     : {image_path}")
    print(f"  Result    : {CLASS_NAMES[pred_idx].upper()}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"{'='*40}")
    print(f"\n  Corn      : {probs[0].item()*100:.1f}%")
    print(f"  Not Corn  : {probs[1].item()*100:.1f}%")

    return CLASS_NAMES[pred_idx], confidence

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_leaf.jpg"
    predict(image_path)