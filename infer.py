import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as f

from rune.model import RuneEmbedding

def build_embeddings(root="data/runes", model_path="rune_embed.pt", emb_dim=128):
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=root, transform=_transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    _model = RuneEmbedding(emb_dim=emb_dim).to(_device)
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(_device)
            emb = _model(imgs)
            all_embeddings.append(emb.cpu())
            all_labels.extend(labels.tolist())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    _class_names = dataset.classes
    return all_embeddings, all_labels, _class_names, _transform, _model, _device

def cosine_sim(a, b):
    a = a.unsqueeze(0)
    sim = f.cosine_similarity(a, b, dim=1)
    return sim

if __name__ == "__main__":
    embs, labels, class_names, transform, model, device = build_embeddings()

    idx = 0
    query_emb = embs[idx]

    sims = cosine_sim(query_emb, embs)
    topk = torch.topk(sims, k=5)

    print("Query belongs to class:", class_names[labels[idx]])
    print("Top-5 similar runes:")
    for score, i in zip(topk.values, topk.indices):
        print(f"  sim={score:.3f}, class={class_names[labels[i]]}, idx={i.item()}")
