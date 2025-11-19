import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader

from rune.dataset import TripletRuneDataset
from rune.model import RuneEmbedding


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin=1.0):
    pos_dist = f.pairwise_distance(anchor, positive)
    neg_dist = f.pairwise_distance(anchor, negative)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TripletRuneDataset("data/runes")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = RuneEmbedding(emb_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            anchor, positive, negative = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = triplet_loss(emb_a, emb_p, emb_n, margin=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * anchor.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "weights/rune_embed.pt")
    print("Saved model to rune_embed.pt")

if __name__ == "__main__":
    main()
