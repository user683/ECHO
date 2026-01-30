# visulogic_to_json.py
import argparse
import json
import os
import re
from io import BytesIO

from PIL import Image


def _resolve_path(path, base_dirs):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    for base in base_dirs:
        if not base:
            continue
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return None


def _normalize_image(image, base_dirs):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        resolved = _resolve_path(image, base_dirs)
        if resolved is None:
            raise FileNotFoundError(f"Image path not found: {image}")
        return Image.open(resolved)
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(BytesIO(image["bytes"]))
        if image.get("path"):
            resolved = _resolve_path(image["path"], base_dirs)
            if resolved is None:
                raise FileNotFoundError(f"Image path not found: {image['path']}")
            return Image.open(resolved)
        if image.get("image") is not None:
            return Image.open(BytesIO(image["image"]))
    raise TypeError(f"Unsupported image type: {type(image)}")


def _save_image(image, out_dir, filename, rel_root, path_prefix, base_dirs):
    pil_image = _normalize_image(image, base_dirs)
    if pil_image is None:
        return None
    path = os.path.join(out_dir, filename)
    pil_image.convert("RGB").save(path, format="PNG")
    if path_prefix:
        rel_path = os.path.join(path_prefix, filename)
    else:
        rel_path = os.path.relpath(path, rel_root)
    return {"path": rel_path}


def _load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description="Convert VisuLogic jsonl to verl json format.")
    parser.add_argument("--input_jsonl", required=True, help="Path to data.jsonl")
    parser.add_argument("--image_root", required=True, help="Base directory for images/")
    parser.add_argument("--output_dir", required=True, help="Output directory for json and images")
    parser.add_argument(
        "--path_prefix",
        default=None,
        help="Path prefix recorded in json for each image (relative to project root).",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if args.path_prefix is None:
        path_prefix = f"data/{os.path.basename(data_dir)}/images"
    else:
        path_prefix = args.path_prefix

    rel_root = os.path.dirname(data_dir)
    base_dirs = [args.image_root]

    items = _load_jsonl(args.input_jsonl)
    rows = []
    for idx, ex in enumerate(items):
        raw_question = ex.get("question", "")
        question = raw_question
        if isinstance(question, str):
            stripped = re.sub(r"<image\d+>", "", question).strip()
            question = stripped if stripped else raw_question.strip()
        answer = ex.get("label")
        subject = ex.get("tag")
        image = ex.get("image_path")

        prompt = question
        if subject:
            prompt = f"[{subject}]\n{prompt}"

        out = {
            "prompt": prompt,
            "answer": answer,
        }
        if image is not None:
            saved = _save_image(image, images_dir, f"test_{idx}.png", rel_root, path_prefix, base_dirs)
            out["images"] = [saved] if saved is not None else []
        rows.append(out)

    output_path = os.path.join(data_dir, "test.json")
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
