import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip.models.blip_vqa import blip_vqa

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

def blip_answer_question(image_path, question):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 480

    # Load image
    image = load_image(image_path=image_path, image_size=image_size, device=device)

    # Load BLIP VQA model
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    # Inference
    with torch.no_grad():
        answer = model(image, question, train=False, inference='generate')

    return answer[0]

def main():
    image_path = "assets/sample_image.jpg"
    questions = [
        "Where is the dog present in the image?",
        "Where is the man present in the image?"
    ]

    for i, question in enumerate(questions, start=1):
        answer = blip_answer_question(image_path, question)
        print(f"Q{i}: {question}")
        print(f"A{i}: {answer}\n")

if __name__ == "__main__":
    main()

