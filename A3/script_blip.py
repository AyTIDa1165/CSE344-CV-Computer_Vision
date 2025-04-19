from models.blip_vqa import blip_vqa
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def load_image(image_path, image_size, device):
    raw_image = Image.open(image_path).convert('RGB')   
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ]) 
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 480
image = load_image(image_path="sample_image.jpg", image_size=image_size, device=device)     

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    
model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

question = 'Where is the man present in the image?'

with torch.no_grad():
    answer = model(image, question, train=False, inference='generate') 
    print('answer: ' + answer[0])