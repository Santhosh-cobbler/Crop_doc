import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from math import ceil

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "./corn_or_not"   # corn vs not_corn folder
BATCH_SIZE  = 32
EPOCHS      = 10                # binary task is simpler, 10 is enough
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ── Transforms ────────────────────────────────────────────────────────────────
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

# ── Dataset ───────────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

print(f"Classes found : {full_dataset.classes}")  # should be ['corn', 'not_corn']
print(f"Total images  : {len(full_dataset)}")

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Model (using pretrained MobileNetV2 — lightweight and fast) ───────────────
# AI_1 is a simple binary task — no need for full EfficientNet from scratch
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Replace final classifier with binary output (corn vs not_corn)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 2)  # 2 classes only
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ── Training Loop ─────────────────────────────────────────────────────────────
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

# ── Run ───────────────────────────────────────────────────────────────────────
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
            torch.save(model.state_dict(), "best_model_ai1.pth")
            print(f"  ✓ Best AI_1 model saved (val_acc: {val_acc:.4f})")

    print(f"\nAI_1 Training complete. Best Val Accuracy: {best_val_acc:.4f}")