import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from math import ceil
import sys


#EfficientNEt-models
base_model =[
    # expand ratio, channels, repeats, tride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1 ,1, 3]
]

phi_values = {
    "b0": (0, 224, 0.2), # the formula
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False
        ) 

        #if the group=1 -> it is nrml convolutional
        # if group=in_channels then it is depthwise_convo

        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # applied sigmoid Linear Unit function element-wise

    def forward(self,x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        return x*self.se(x)
    
class InvertedResidualBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            stride, 
            padding, 
            kernel_size, 
            expand_ratio, 
            reduction=4, # squeeze excitation
            surival_prob=0.8 # for stochastic depth
            ):
        super(InvertedResidualBlock, self). __init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels/reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
            )
        
        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim #depth-wise convolution
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def stochastic_depth(self, x):
        if not self.training:
            return x
          
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob)*binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280*width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        channels=int(32*width_factor)
        features = [CNNBlock(3, channels,3,stride=2,padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels*width_factor)/4)

            layer_repeats = ceil(repeats*depth_factor)

            for layer in range(layer_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride= stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding= kernel_size//2 # if k=1: pad=0, k=3: pad=1 k=5: pad=2 (2:1)
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )
        return nn.Sequential(*features)

    def forward(self,x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DISEASE_CLASSES = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy'
]

DISPLAY_NAMES = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Gray Leaf Spot',
    'Corn_(maize)___Common_rust_'                       : 'Common Rust',
    'Corn_(maize)___Northern_Leaf_Blight'               : 'Northern Leaf Blight',
    'Corn_(maize)___healthy'                            : 'Healthy'
}

# ── Load AI_1 (Corn vs Not Corn) ──────────────────────────────────────────────
print("Loading AI_1 (Corn Detector)...")
ai1 = models.mobilenet_v2(weights=None)
ai1.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(ai1.last_channel, 2)
)
ai1.load_state_dict(torch.load("best_model_ai1.pth", map_location=DEVICE))
ai1 = ai1.to(DEVICE)
ai1.eval()
print("  ✓ AI_1 loaded!")

# ── Load AI_2 (Disease Classifier) ───────────────────────────────────────────
print("Loading AI_2 (Disease Classifier)...")
ai2 = EfficientNet(version="b0", num_classes=4).to(DEVICE)
ai2.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
ai2.eval()
print("  ✓ AI_2 loaded!")

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Pipeline ──────────────────────────────────────────────────────────────────
def predict(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image: {e}")
        return

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    print(f"\n{'='*50}")
    print(f"  Image : {image_path}")
    print(f"{'='*50}")

    # Stage 1: AI_1 
    with torch.no_grad():
        out1  = ai1(img_tensor)
        prob1 = torch.softmax(out1, dim=1)[0]
        corn_conf     = prob1[0].item()
        not_corn_conf = prob1[1].item()

    print(f"\n  [STAGE 1] Corn Leaf Detector (AI_1)")
    print(f"    Corn     : {corn_conf*100:.1f}%")
    print(f"    Not Corn : {not_corn_conf*100:.1f}%")

    # Reject if not confident it's corn
    if corn_conf < 0.85:
        print(f"\n RESULT: NOT A CORN LEAF")
        print(f"     This image does not appear to be a corn leaf.")
        print(f"     Please upload a corn leaf image.")
        print(f"{'='*50}")
        return

    print(f"Confirmed Corn Leaf! ({corn_conf*100:.1f}% confident)")

    # ── Stage 2: AI_2 ─────────────────────────────
    with torch.no_grad():
        out2  = ai2(img_tensor)
        prob2 = torch.softmax(out2, dim=1)[0]
        pred_idx   = prob2.argmax().item()
        pred_class = DISEASE_CLASSES[pred_idx]
        confidence = prob2[pred_idx].item() * 100

    print(f"\n  [STAGE 2] Disease Classifier (AI_2)")
    print(f"    Diagnosis  : {DISPLAY_NAMES[pred_class]}")
    print(f"    Confidence : {confidence:.1f}%")

    print(f"\n  All Disease Probabilities:")
    for i, name in enumerate(DISEASE_CLASSES):
        bar = '█' * int(prob2[i].item() * 30)
        print(f"    {DISPLAY_NAMES[name]:<28} {prob2[i].item()*100:5.1f}%  {bar}")

    #Final Verdict 
    print(f"\n{'='*50}")
    if DISPLAY_NAMES[pred_class] == 'Healthy':
        print(f"FINAL RESULT : Your corn leaf is HEALTHY!")
    else:
        print(f"FINAL RESULT : {DISPLAY_NAMES[pred_class]} detected!")
        print(f"      Confidence   : {confidence:.1f}%")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_leaf.jpg"
    predict(image_path)