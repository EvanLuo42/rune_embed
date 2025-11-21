import argparse

import torch
import torch.nn.functional as f
from PIL import Image
from torchvision import datasets, transforms

from rune.model import RuneResNetEmbedding


@torch.no_grad()
def embed_image(model, img_path, device):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)
    emb = model(img)
    return emb / emb.norm(p=2, dim=1, keepdim=True)  # normalize


def build_class_embeddings(root, model, device):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    folder = datasets.ImageFolder(root=root, transform=transform)
    class_to_embs = {cls: [] for cls in folder.classes}

    for img, label in folder:
        img = img.unsqueeze(0).to(device)
        emb = model(img)
        emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        class_name = folder.classes[label]
        class_to_embs[class_name].append(emb.cpu())

    return class_to_embs


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = RuneResNetEmbedding(emb_dim=128).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print("Building class embeddings...")
    class_embs = build_class_embeddings(args.data, model, device)

    print(f"Embedding input image: {args.input}")
    query_emb = embed_image(model, args.input, device)

    dist_list = []
    for cls, embs in class_embs.items():
        if len(embs) == 0:
            continue

        embs = torch.cat(embs, dim=0)
        dists = f.pairwise_distance(query_emb.cpu(), embs)
        mean_dist = dists.mean().item()

        dist_list.append((cls, mean_dist))

    dist_list.sort(key=lambda x: x[1])

    top_k = args.topk
    result = dist_list[:top_k]

    print("=====================================")
    print(f" Input image: {args.input}")
    print(" Top-{} predictions:".format(top_k))
    for i, (cls, d) in enumerate(result):
        print(f" {i+1}. {cls}   (dist={d:.4f})")
    print("=====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to test image")
    parser.add_argument("--data", default="data/runes",
                        help="Training dataset (ImageFolder format)")
    parser.add_argument("--model", default="weights/rune_embed.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--topk", type=int, default=1,
                        help="Output top-k most similar classes (default=1)")
    args = parser.parse_args()

    infer(args)
