import torch
from rune.model import RuneResNetEmbedding

def main():
    emb_dim = 128
    model = RuneResNetEmbedding(emb_dim=emb_dim)
    model.load_state_dict(torch.load("weights/rune_embed.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        "weights/rune_embedding.onnx",
        input_names=["input"],
        output_names=["embedding"],
        opset_version=18,
    )

    print("Exported to rune_embedding.onnx")

if __name__ == "__main__":
    main()