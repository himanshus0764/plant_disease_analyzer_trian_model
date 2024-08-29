import warnings
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
import json
from PIL import Image
import os

# Suppress deprecated warnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning)

data_dir = '/home/laptop_asus/Music/Plant_Disease_Detector'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
class_names = dataset.classes
num_classes = len(class_names)

# Use the recommended weights parameter instead of pretrained
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model_path = '/home/laptop_asus/Documents/modelfiles/model.pth'
torch.save(model.state_dict(), model_path)

class_names_path = '/home/laptop_asus/Documents/modelfiles/class_names.json'
with open(class_names_path, 'w') as f:
    json.dump(class_names, f)

def predict(image_path, model, transform, device, class_names):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    predicted_class_index = predicted.item()
    predicted_class_name = class_names[predicted_class_index]
    
    full_class_name = os.path.basename(os.path.dirname(image_path))
    
    return full_class_name

image_path = '/home/laptop_asus/Music/Plant_Disease_Detector/model/dataset/Tomato_Leaf_Mold/0a9b3ff4-5343-4814-ac2c-fdb3613d4e4d___Crnl_L.Mold 6559.JPG'
predicted_class = predict(image_path, model, transform, device, class_names)
print(f'Predicted class: {predicted_class}')