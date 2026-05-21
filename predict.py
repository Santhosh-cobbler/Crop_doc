import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
from math import ceil

# ── Paste your full EfficientNet code here ────────────────────────────────────
import torch
import torch.nn as nn
from math import ceil

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

"""def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"

    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4,10
    x = torch.randn((num_examples, 3, res, res)).to(device)
    model = EfficientNet(
        version=version,
        num_classes = num_classes,
    ).to(device)

    print(model(x).shape)

test()
"""
# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy'
]

# Friendly display names
DISPLAY_NAMES = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Gray Leaf Spot',
    'Corn_(maize)___Common_rust_':                        'Common Rust',
    'Corn_(maize)___Northern_Leaf_Blight':                'Northern Leaf Blight',
    'Corn_(maize)___healthy':                             'Healthy'
}

# ── Load model ────────────────────────────────────────────────────────────────
model = EfficientNet(version="b0", num_classes=4).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Predict function ──────────────────────────────────────────────────────────
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    pred_class   = CLASS_NAMES[pred_idx]
    display_name = DISPLAY_NAMES[pred_class]
    confidence   = probs[pred_idx].item() * 100

    print(f"\n{'='*40}")
    print(f"  Image     : {image_path}")

    # ── OOD Check ────────────────────────────────
    CONFIDENCE_THRESHOLD = 85.0  # adjust this value

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"  Result    : ⚠️  NOT A CORN LEAF")
        print(f"  Reason    : Model confidence too low ({confidence:.1f}%)")
        print(f"              Please upload a Corn leaf image only.")
        print(f"{'='*40}")
        return

    # ── Normal prediction ─────────────────────────
    print(f"  Disease   : {display_name}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"{'='*40}")
    print("\nAll class probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        bar = '█' * int(probs[i].item() * 30)
        print(f"  {DISPLAY_NAMES[name]:<30} {probs[i].item()*100:5.1f}% {bar}")

if __name__ == '__main__':
    # Usage: python predict.py path/to/leaf_image.jpg
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_leaf.jpg"
    predict(image_path)