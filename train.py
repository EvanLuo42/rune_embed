import torch
from torch.utils.data import DataLoader
from torch import nn

from rune.dataset import TripletRuneDataset
from rune.model import RuneResNetEmbedding


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TripletRuneDataset("data/runes")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = RuneResNetEmbedding(emb_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        total = 0

        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * anchor.size(0)

        print(f"Epoch {epoch + 1}/{num_epochs}, loss = {total / len(dataset):.4f}")

    torch.save(model.state_dict(), "weights/rune_embed.pt")
    print("Saved model to rune_embed.pt")

if __name__ == "__main__":
    main()
