import os
import torch
import csv
from blip.models.blip import blip_decoder
from question1 import rank_with_clip, rank_with_clips
from question2 import load_image

def generate_caption(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 480

    # Load and preprocess image
    image = load_image(image_path=image_path, image_size=image_size, device=device)

    # Load BLIP captioning model
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    # Generate caption
    with torch.no_grad():
        caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)

    return caption[0]

def main():
    image_dir = "assets/samples"
    output_csv = "image_caption_scores.csv"
    image_files = [f for f in os.listdir(image_dir)]
    results = []

    for img_file in image_files:
        image_path = os.path.join(image_dir, img_file)
        try:
            caption = generate_caption(image_path)
            clip_score = rank_with_clip(image_path, [caption])[0][1]
            clips_score = rank_with_clips(image_path, [caption])[0][1]
            entry = {
                "image": img_file,
                "caption": caption,
                "clip_score": round(clip_score, 4),
                "clips_score": round(clips_score, 4)
            }
            print(
                f"Image: {entry['image']}\n"
                f"Caption: {entry['caption']}\n"
                f"CLIP Score: {entry['clip_score']}\n"
                f"CLIPS Score: {entry['clips_score']}\n"
            )
            results.append(entry)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption", "clip_score", "clips_score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    main()
