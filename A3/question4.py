import torchvision.transforms as T
import numpy as np
import os
import torch
import torch.nn.functional as F
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

def segment_image_with_text(image_path, sentence):
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

    # Inference
    with torch.no_grad():
        last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
        embedding = last_hidden_states.permute(0, 2, 1)
        output = model(img_tensor, embedding, l_mask=attention_mask.unsqueeze(-1))
        output = output.argmax(1, keepdim=True)
        output = F.interpolate(output.float(), (original_h, original_w)).squeeze().cpu().numpy().astype(np.uint8)

    # Overlay mask and return visualization
    vis = overlay_davis(img_ndarray, output)
    return Image.fromarray(vis)

def main():
    image_dir = 'assets/samples'
    segmented_image_dir = 'ris_results'
    reference_path = 'reference.txt'
    reference = load_reference(reference_path)

    for img, ref in reference:
        image_path = os.path.join(image_dir, img)
        segmented_image_path = os.path.join(segmented_image_dir, f'{img.split('.')[0]}_seg.jpg')
        segmented_image = segment_image_with_text(image_path, ref)
        segmented_image.save(segmented_image_path)
    
if __name__ == "__main__":
    main()