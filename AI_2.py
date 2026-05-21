import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
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

test()"""

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "./color"   # path to your color folder
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
NUM_CLASSES = 4           # 4 corn disease classes
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ─── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Dataset ──────────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

print(f"\nClasses found: {full_dataset.classes}")
print(f"Total images : {len(full_dataset)}")

# 80% train, 20% val
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# Apply val transform to val set
val_ds.dataset.transform = val_transform

# Change num_workers=2 to 0 for both loaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ─── Model ────────────────────────────────────────────────────────────────────
model = EfficientNet(version="b0", num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ─── Training Loop ────────────────────────────────────────────────────────────
def train_one_epoch(loader):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def validate(loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    best_val_acc = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(train_loader)
        val_loss,   val_acc   = validate(val_loader)
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Best model saved (val_acc: {val_acc:.4f})")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")