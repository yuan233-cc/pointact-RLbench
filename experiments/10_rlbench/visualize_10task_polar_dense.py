"""Preview one dense polar keyframe from each task in the ten-task export."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import lmdb
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps_polar_incomplete9_v1"
RAW = ROOT.parent / "rlbench_custom_render/RLBench/output/ten_tasks_polar_train_20260921"
OUTPUT = ROOT / "RLBench_10tasks_polar_dense_preview_20260922"
SIZE = 256
GAP = 12
LABEL = 28
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def heatmap(values: np.ndarray, low: float, high: float, valid: np.ndarray,
            colormap: int) -> Image.Image:
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    bgr = cv2.applyColorMap(np.uint8(np.rint(scaled * 255)), colormap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def angle_image(cos2: np.ndarray, sin2: np.ndarray,
                valid: np.ndarray) -> Image.Image:
    # AoLP is axial: 0 and 180 degrees represent the same orientation.
    angle = np.mod(0.5 * np.arctan2(sin2, cos2), np.pi)
    hue = np.uint8(np.floor(angle / np.pi * 180))
    hsv = np.stack((hue, np.full_like(hue, 255), np.full_like(hue, 255)), axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def signed_image(values: np.ndarray, valid: np.ndarray) -> Image.Image:
    # Blue (-1), white (0), red (+1), with a fixed physical channel scale.
    positive = np.clip(values, 0.0, 1.0)
    negative = np.clip(-values, 0.0, 1.0)
    rgb = np.stack((1.0 - negative,
                    1.0 - positive - negative,
                    1.0 - positive), axis=-1)
    rgb = np.uint8(np.rint(np.clip(rgb, 0.0, 1.0) * 255))
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def tile(title: str, image: Image.Image) -> Image.Image:
    result = Image.new("RGB", (SIZE, SIZE + LABEL), "#f5f5f5")
    result.paste(image, (0, LABEL))
    ImageDraw.Draw(result).text((6, 4), title, fill="#18212c", font=font(17))
    return result


def legend(valid_fraction: float, median_dolp: float) -> Image.Image:
    panel = Image.new("RGB", (SIZE, SIZE + LABEL), "#f5f5f5")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), "AoLP hue / stats", fill="#18212c", font=font(17))
    for x in range(8, SIZE - 8):
        hue = np.uint8(np.floor((x - 8) / (SIZE - 16) * 179))
        rgb = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8),
                           cv2.COLOR_HSV2RGB)[0, 0]
        draw.line((x, 62, x, 84), fill=tuple(map(int, rgb)))
    draw.text((8, 91), "0       45       90      135     180 deg", fill="black", font=font(12))
    draw.text((8, 138), f"Valid pixels: {valid_fraction:.1%}", fill="black", font=font(17))
    draw.text((8, 169), f"Median DoLP: {median_dolp:.3f}", fill="black", font=font(17))
    draw.text((8, 211), "Black = invalid polar pixel", fill="black", font=font(13))
    return panel


def assemble(title: str, tiles: list[Image.Image], columns: int) -> Image.Image:
    rows = (len(tiles) + columns - 1) // columns
    width = columns * SIZE + (columns + 1) * GAP
    height = 42 + rows * (SIZE + LABEL) + (rows + 1) * GAP
    output = Image.new("RGB", (width, height), "#e7ebef")
    draw = ImageDraw.Draw(output)
    draw.text((GAP, 8), title, fill="#172334", font=font(21))
    for index, item in enumerate(tiles):
        row, column = divmod(index, columns)
        output.paste(item, (GAP + column * (SIZE + GAP),
                            42 + GAP + row * (SIZE + LABEL + GAP)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    meta = json.loads((args.dataset / "meta/polar_incomplete_features.json").read_text())
    episodes = [json.loads(line) for line in (args.dataset / "meta/episodes.jsonl").open()]
    task_names = meta["tasks"]
    episodes_per_task = meta["episodes_per_task"]
    if len(task_names) != 10 or len(episodes) != len(task_names) * episodes_per_task:
        raise ValueError("Expected the complete ten-task dataset")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing preview: {args.output}")
    args.output.mkdir(parents=True)

    overview_cards = []
    records = []
    env = lmdb.open(str(args.dataset / "polar_frontview_dense"), readonly=True,
                    lock=False, readahead=False, max_readers=16)
    try:
        with env.begin() as transaction:
            for task_index, task in enumerate(task_names):
                episode_index = task_index * episodes_per_task
                frame_index = episodes[episode_index]["length"] // 2
                key = f"{episode_index}-{frame_index}"
                payload = transaction.get(key.encode("ascii"))
                if payload is None:
                    raise KeyError(f"Missing polar frame {key}")
                with np.load(io.BytesIO(payload)) as frame:
                    dolp = np.asarray(frame["DoLP"], dtype=np.float32)
                    cos2 = np.asarray(frame["cos2AoLP"], dtype=np.float32)
                    sin2 = np.asarray(frame["sin2AoLP"], dtype=np.float32)
                    valid = np.asarray(frame["valid_mask"], dtype=bool)

                raw_frame = args.raw / task / "episode_000000" / "frames_spp512" / f"{frame_index:06d}.npz"
                with np.load(raw_frame) as source:
                    rgb = np.asarray(source["rgb"], dtype=np.uint8)
                    if not np.allclose(source["DoLP"], dolp, equal_nan=True):
                        raise ValueError(f"RGB reference and dense polar frame differ: {key}")
                if rgb.shape != (SIZE, SIZE, 3) or dolp.shape != (SIZE, SIZE):
                    raise ValueError(f"Unexpected frame size: {key}, {rgb.shape}, {dolp.shape}")
                valid &= np.isfinite(dolp) & np.isfinite(cos2) & np.isfinite(sin2)
                safe_dolp = np.nan_to_num(dolp, nan=0.0, posinf=0.0, neginf=0.0)
                safe_cos2 = np.nan_to_num(cos2, nan=0.0)
                safe_sin2 = np.nan_to_num(sin2, nan=0.0)

                rgb_tile = tile("RGB reference", Image.fromarray(rgb))
                dolp_absolute = tile("DoLP (0-1)", heatmap(safe_dolp, 0, 1, valid, cv2.COLORMAP_TURBO))
                dolp_detail = tile("DoLP detail (0-0.25)", heatmap(safe_dolp, 0, .25, valid, cv2.COLORMAP_TURBO))
                aolp = tile("AoLP (0-180 deg)", angle_image(safe_cos2, safe_sin2, valid))
                cos_image = tile("cos(2 AoLP) (-1 to 1)", signed_image(safe_cos2, valid))
                sin_image = tile("sin(2 AoLP) (-1 to 1)", signed_image(safe_sin2, valid))
                valid_image = tile("valid_mask", Image.fromarray(np.uint8(valid) * 255, "L").convert("RGB"))
                fraction = float(valid.mean())
                median = float(np.median(safe_dolp[valid]))
                filename = f"{task}_episode_{episode_index:06d}_frame_{frame_index:06d}.png"
                assemble(f"{task} | episode {episode_index} | frame {frame_index}",
                         [rgb_tile, dolp_absolute, dolp_detail, aolp,
                          cos_image, sin_image, valid_image, legend(fraction, median)],
                         4).save(args.output / filename)
                overview_cards.append(assemble(
                    f"{task_index + 1:02d} {task.replace('_', ' ')} | episode {episode_index} frame {frame_index}",
                    [rgb_tile, dolp_detail, aolp], 3))
                records.append({"task": task, "episode_index": episode_index,
                                "frame_index": frame_index, "lmdb_key": key,
                                "valid_fraction": fraction, "median_dolp": median,
                                "image": filename})
                print(f"{task}: {filename}")
    finally:
        env.close()

    card_width, card_height = overview_cards[0].size
    overview = Image.new("RGB", (2 * card_width + 3 * GAP,
                                 5 * card_height + 6 * GAP), "#d7dfe7")
    for index, card in enumerate(overview_cards):
        row, column = divmod(index, 2)
        overview.paste(card, (GAP + column * (card_width + GAP),
                              GAP + row * (card_height + GAP)))
    overview.save(args.output / "overview_10_tasks.png")
    (args.output / "manifest.json").write_text(json.dumps({
        "source_lmdb": str(args.dataset / "polar_frontview_dense"),
        "selection": "first episode and middle keyframe of each task",
        "dolp_detail_display_range": [0, 0.25],
        "angle_display_degrees": [0, 180],
        "invalid_pixels": "black",
        "frames": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
