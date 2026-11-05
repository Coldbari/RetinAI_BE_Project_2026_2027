import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image

# =========================================
# CONFIG
# =========================================

IMAGE_PATH = "test images/image copy.png"
MODEL_PATH = "best_rop_model.pth"

IMAGE_SIZE = 160

# =========================================
# DEVICE
# =========================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"\nUsing device: {DEVICE}")

# =========================================
# CLASS NAMES
# =========================================

class_names = {
    0: "Normal Retina (DG0)",
    1: "Abnormal Retina (DG1)"
}

# =========================================
# IMAGE TRANSFORM
# =========================================

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================
# LOAD MODEL
# =========================================

model = models.efficientnet_b0(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")

# =========================================
# LOAD IMAGE
# =========================================

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

# =========================================
# PREDICTION
# =========================================

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_class = torch.max(probabilities, 1)

predicted_class = predicted_class.item()
confidence = confidence.item()

# =========================================
# RESULT
# =========================================

print("\n=====================================")
print("PREDICTION RESULT")
print("=====================================")

print(f"\nPrediction : {class_names[predicted_class]}")
print(f"Confidence : {confidence * 100:.2f}%")

print("\nClass Probabilities:")

for idx, prob in enumerate(probabilities[0]):

    print(
        f"{class_names[idx]}: "
        f"{prob.item() * 100:.2f}%"
    )