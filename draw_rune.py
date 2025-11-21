import argparse
import re
from pathlib import Path

import pygame
import numpy as np
from PIL import Image
import sys

WIDTH, HEIGHT = 512, 512
BRUSH_SIZE = 10

def parse_args():
    parser = argparse.ArgumentParser(description="Rune drawing tool")
    parser.add_argument("--type", "-t", required=True,
                        help="Rune type, e.g. fire, water, earth")
    parser.add_argument("--root", "-r", default="data/runes",
                        help="Root of the outputs")
    return parser.parse_args()


def next_rune_path(root: Path, rune_type: str) -> Path:
    type_dir = root / rune_type
    type_dir.mkdir(parents=True, exist_ok=True)

    files = list(type_dir.glob("*.png"))

    pattern = re.compile(rf"{rune_type}-(\d+)\.png")

    max_id = 0

    for f in files:
        match = pattern.match(f.name)
        if match:
            num = int(match.group(1))
            max_id = max(max_id, num)

    new_id = max_id + 1
    new_name = f"{rune_type}-{new_id:02d}.png"

    return type_dir / new_name


def save_grayscale(surface, filename="output.png"):
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr, (1, 0, 2))
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    img.save(filename)
    print(f"Saved grayscale image to {filename}")


def main():
    args = parse_args()
    rune_type = args.type
    root = Path(args.root)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drawing Canvas — Press S to Save")

    screen.fill((255, 255, 255))
    drawing = False

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    path = next_rune_path(root, rune_type)
                    save_grayscale(screen, str(path))
                    screen.fill((255, 255, 255))
                    pygame.display.flip()

        if drawing:
            x, y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, (0, 0, 0), (x, y), BRUSH_SIZE)

        pygame.display.flip()
        clock.tick(120)


if __name__ == "__main__":
    main()
