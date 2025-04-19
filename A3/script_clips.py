import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = create_model_from_pretrained('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
model = model.to(device)

image = preprocess(Image.open("sample_image.jpg")).unsqueeze(0).to(device)

descriptions = [
    "A fair bearded man standing in a room in white formal shirt, grey trousers and brown leather shoes holding a big grey dog in his arms.",
    "A tall man standing in a room holding a big dog in his arms.",
    "A man standing straight and holding a dog with both hands.",
    "A big animal carried by a human being in a closed room.",
    "Two living beings in front of a bookcase in a white room",
    "A dog in a lying position some distance above the ground.",
    "A book case kept in front of a white wall being blocked by two living entities.",
    "A human showing empathy towards a fellow living creature by displaying care for it.",
    "A snake trying to attack an innocent child playing with her toys.",
    "A black cat is playing with a young boy in a sunlit park, surrounded by green trees and a clear blue sky.",
]

text = tokenizer(descriptions, context_length=model.context_length).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

sorted_indices = np.argsort(probs[0].cpu().numpy())[::-1]
for rank, idx in enumerate(sorted_indices, start=1):
    description = descriptions[idx]
    probability = probs[0][idx].item()
    print(f"{rank:2d}. {description:40s} ({probability * 100:.2f}%)")
