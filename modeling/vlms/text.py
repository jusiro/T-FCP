import clip
import torch

from modeling.vlms.configs.constants import *

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_text_prototypes(clip_model, classnames, templates, clip_id="CLIP"):
    clip_model.eval()

    if clip_id.split("-")[0] in ["MetaCLIP"]:
        from transformers import AutoProcessor
        # Load tokenizer
        tokenizer = AutoProcessor.from_pretrained(vlms_loadid_from_name[clip_id]).tokenizer

    print("Extracting text prototypes from class names...")
    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            if clip_id.split("-")[0] == "CLIP":
                tokens = clip.tokenize(
                    [template.format(text.replace('_', ' ')) for template in templates]).to(device)
                prototype = clip_model.encode_text(tokens.to(device))
            elif clip_id.split("-")[0] in ["MetaCLIP"]:
                # Prepare tokens
                tokens = tokenizer([template.format(text.replace('_', ' ')) for template in templates], return_tensors="pt", padding=True).to(device)
                # Get text prototype
                prototype = clip_model.get_text_features(input_ids=tokens["input_ids"].to(device), attention_mask=tokens["attention_mask"].to(device))

            # Add prototype
            text_embeddings.append(prototype)

    text_embeddings = torch.stack(text_embeddings)
    text_embeddings_avg = text_embeddings.mean(1)
    return text_embeddings_avg, text_embeddings