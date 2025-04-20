import torchvision.transforms as T
import numpy as np
import os
import torch
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from ris.bert.tokenization_bert import BertTokenizer
from ris.bert.modeling_bert import BertModel
from ris.lib import segmentation
from scipy.ndimage.morphology import binary_dilation

def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale
    im_overlay = image.copy()
    object_ids = np.unique(mask)
    for object_id in object_ids[1:]:
        foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id
        im_overlay[binary_mask] = foreground[binary_mask]
        countours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours, :] = 0
    return im_overlay.astype(image.dtype)

def load_reference(reference_path):
    image_descriptions = []
    with open(reference_path, 'r') as f:
        for line in f:
            if ':' in line:
                img, desc = line.strip().split(':', 1)
                image_descriptions.append((img.strip(), desc.strip().strip('"')))
    return image_descriptions

def plot_saved_images(reference, image_dir, segmented_dir, feature_map_dir):
    N = len(reference)
    fig, axes = plt.subplots(N, 3, figsize=(15, 5 * N))  # <-- even bigger figure size

    if N == 1:
        axes = axes.reshape(1, 3)

    for idx, (img, _) in enumerate(reference):
        name = img.split('.')[0]
        
        orig_path = os.path.join(image_dir, img)
        seg_path = os.path.join(segmented_dir, f'{name}_seg.jpg')
        feat_path = os.path.join(feature_map_dir, f'{name}_feat.jpg')

        orig = Image.open(orig_path)
        seg = Image.open(seg_path)
        feat = Image.open(feat_path)

        axes[idx, 0].imshow(orig)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(seg)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(feat)
        axes[idx, 2].axis('off')

    axes[0, 0].set_title('Original', fontsize=16)
    axes[0, 1].set_title('Segmented', fontsize=16)
    axes[0, 2].set_title('Feature Map', fontsize=16)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, hspace=0.4, wspace=0.05)
    plt.show()

def segment_image_with_text(image_path, sentence, visualize_features=False):
    weights = './ris/checkpoints/refcoco.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_ndarray = np.array(img)
    original_w, original_h = img.size

    transform = T.Compose([
        T.Resize(480),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Tokenize sentence
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentence_tokenized = tokenizer.encode(sentence, add_special_tokens=True)[:20]
    padded_sent_toks = [0] * 20
    attention_mask = [0] * 20
    padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
    attention_mask[:len(sentence_tokenized)] = [1] * len(sentence_tokenized)
    padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

    # Load model
    class args:
        swin_type = 'base'
        window12 = True
        mha = ''
        fusion_drop = 0.0

    model = segmentation.__dict__['lavt'](pretrained='', args=args).to(device)
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.pooler = None

    checkpoint = torch.load(weights, map_location='cpu')
    bert_model.load_state_dict(checkpoint['bert_model'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    bert_model.eval()

    with torch.no_grad():
        last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
        embedding = last_hidden_states.permute(0, 2, 1)
        output, x_c4 = model(img_tensor, embedding, l_mask=attention_mask.unsqueeze(-1))
        
        # Segmentation mask
        seg_mask = output.argmax(1, keepdim=True)
        seg_mask = F.interpolate(seg_mask.float(), (original_h, original_w)).squeeze().cpu().numpy().astype(np.uint8)
        
        # Feature visualization
        if visualize_features:
            feat = x_c4.mean(1).squeeze().cpu().numpy()
            feat = cv2.resize(feat, (original_w, original_h))
            feat = (255 * (feat - feat.min()) / (np.ptp(feat) + 1e-5)).astype(np.uint8)
            feat_color = cv2.applyColorMap(feat, cv2.COLORMAP_JET)
            return Image.fromarray(overlay_davis(img_ndarray, seg_mask)), Image.fromarray(feat_color)

    vis = overlay_davis(img_ndarray, seg_mask)
    return Image.fromarray(vis)


def main():
    image_dir = 'assets/samples'
    segmented_image_dir = 'ris_results'
    reference_path = 'reference.txt'
    reference = load_reference(reference_path)

    for img, ref in reference:
        image_path = os.path.join(image_dir, img)
        name = img.split('.')[0]
        segmented_image_path = os.path.join(segmented_image_dir, f'segmentation/{name}_seg.jpg')
        feature_map_path = os.path.join(segmented_image_dir, f'feature_map/{name}_feat.jpg')

        seg_img, feat_img = segment_image_with_text(image_path, ref, visualize_features=True)
        seg_img.save(segmented_image_path)
        feat_img.save(feature_map_path)

    plot_saved_images(reference, image_dir, f'{segmented_image_dir}/segmentation', f'{segmented_image_dir}/feature_map')
    
if __name__ == "__main__":
    main()