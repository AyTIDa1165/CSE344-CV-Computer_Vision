import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import clip
from open_clip import create_model_from_pretrained, get_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def rank_with_clip(image_path, descriptions):
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(descriptions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarity_scores = (text_features @ image_features.T).squeeze(1)  # [N]

    sorted_indices = similarity_scores.argsort(descending=True)

    return [(descriptions[idx], similarity_scores[idx].item()) for idx in sorted_indices]


def rank_with_clips(image_path, descriptions):
    model, preprocess = create_model_from_pretrained('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
    tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
    model = model.to(device)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(descriptions, context_length=model.context_length).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarity_scores = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    sorted_indices = np.argsort(similarity_scores)[::-1]

    return [(descriptions[idx], similarity_scores[idx]) for idx in sorted_indices]


def main():
    image_path = "assets/sample_image.jpg"

    print("\n===== CLIP (OpenAI) Ranking =====")
    for rank, (desc, score) in enumerate(rank_with_clip(image_path, descriptions), start=1):
        print(f"{rank:2d}. Score: {score:.4f} | {desc}")

    print("\n===== CLIPS Ranking =====")
    for rank, (desc, score) in enumerate(rank_with_clips(image_path, descriptions), start=1):
        print(f"{rank:2d}. Score: {score:.4f} | {desc}")


if __name__ == "__main__":
    main()
