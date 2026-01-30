# save_as_json.py
import os
import re
from io import BytesIO

from PIL import Image
from datasets import load_dataset


def pick_first(example, keys, label):
    for key in keys:
        if key in example:
            return key
    raise KeyError(f"Missing {label} key. Available keys: {list(example.keys())}")


def resolve_output_dirs(output_dir: str):
    data_dir = os.path.abspath(output_dir)
    images_dir = os.path.join(data_dir, "images")
    return data_dir, images_dir


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


def normalize_image(image, base_dirs):
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


def save_image(image, out_dir, filename, rel_root, path_prefix, base_dirs):
    pil_image = normalize_image(image, base_dirs)
    if pil_image is None:
        return None
    path = os.path.join(out_dir, filename)
    pil_image.convert("RGB").save(path, format="PNG")
    if path_prefix:
        rel_path = os.path.join(path_prefix, filename)
    else:
        rel_path = os.path.relpath(path, rel_root)
    return {"path": rel_path}


def _format_choices_with_map(choices, add_letters=True):
    if choices is None:
        return "", {}
    choice_map = {}
    if isinstance(choices, dict):
        lines = []
        for key, value in choices.items():
            label = str(key).strip().rstrip(".")
            value_str = str(value).strip()
            if add_letters:
                rendered = f"{label}.{value_str}"
                choice_map[value_str] = rendered
            else:
                rendered = value_str
            lines.append(rendered)
        return "\n".join(lines), choice_map
    if isinstance(choices, list):
        lines = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, choice in enumerate(choices):
            value_str = str(choice).strip()
            if add_letters:
                label = letters[i] if i < len(letters) else str(i)
                rendered = f"{label}.{value_str}"
                choice_map[value_str] = rendered
            else:
                rendered = value_str
            lines.append(rendered)
        return "\n".join(lines), choice_map
    return str(choices), {}


def _label_to_answer(label):
    if label is None:
        return None
    if isinstance(label, str):
        if label.strip().isdigit():
            label = int(label.strip())
        else:
            return label.strip().upper()
    if isinstance(label, (int, float)):
        label_int = int(label)
        letters = "ABCD"
        if 0 <= label_int < len(letters):
            return letters[label_int]
    return None


def _format_subject(subject):
    if subject is None:
        return None
    if isinstance(subject, list):
        return ", ".join([str(item) for item in subject])
    return str(subject)


def map_fn(
    ex,
    idx,
    split,
    image_out_dir,
    rel_root,
    path_prefix,
    base_dirs,
    question_key,
    answer_key,
    choices_key,
    subject_key,
    label_key,
    add_choice_letters,
    answer_from_choices,
):
    image_key = None
    for key in ["decoded_image", "image", "images"]:
        if key in ex:
            image_key = key
            break

    question = ex[question_key]
    if isinstance(question, str):
        question = re.sub(r"<image\d+>", "", question).strip()
    answer = ex[answer_key] if answer_key else None
    subject = _format_subject(ex.get(subject_key)) if subject_key else None
    label = ex.get(label_key) if label_key else None
    choices = ex.get(choices_key) if choices_key else None
    image = ex.get(image_key, None) if image_key else None

    label_answer = _label_to_answer(label) if label_key else None
    if label_answer is not None:
        answer = label_answer

    prompt = question
    if subject:
        prompt = f"[{subject}]\n{prompt}"
    if choices is not None:
        choices_text, choice_map = _format_choices_with_map(choices, add_letters=add_choice_letters)
        prompt = f"{prompt}\n\nChoices:\n{choices_text}"
        if answer_from_choices and answer is not None:
            answer_str = str(answer).strip()
            if answer_str in choice_map:
                answer = choice_map[answer_str]

    out = {
        "prompt": prompt,
        "answer": answer,
    }
    if image is not None:
        if isinstance(image, list):
            images = []
            for i, img in enumerate(image):
                saved = save_image(img, image_out_dir, f"{split}_{idx}_{i}.png", rel_root, path_prefix, base_dirs)
                if saved is not None:
                    images.append(saved)
        else:
            saved = save_image(image, image_out_dir, f"{split}_{idx}.png", rel_root, path_prefix, base_dirs)
            images = [saved] if saved is not None else []
        out["images"] = images
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download dataset and export to json + image paths.")
    parser.add_argument("--dataset_name", default="hiyouga/geometry3k", help="HuggingFace dataset name.")
    parser.add_argument(
        "--dataset_config",
        default=None,
        help="HuggingFace dataset config name. Use comma to pass multiple configs.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry3k"),
        help="Output directory to write json and images.",
    )
    parser.add_argument(
        "--path_prefix",
        default=None,
        help="Path prefix recorded in json for each image (relative to project root).",
    )
    parser.add_argument("--question_key", default=None, help="Question field name in dataset.")
    parser.add_argument("--answer_key", default=None, help="Answer field name in dataset.")
    parser.add_argument("--choices_key", default=None, help="Choices field name in dataset.")
    parser.add_argument(
        "--choices_add_letters",
        action="store_true",
        help="Prepend A/B/C/D... when formatting choices.",
    )
    parser.add_argument(
        "--answer_from_choices",
        action="store_true",
        help="Replace answer with the matching choice text (e.g., 8 -> A.8).",
    )
    parser.add_argument("--subject_key", default=None, help="Subject/category field name in dataset.")
    parser.add_argument("--label_key", default=None, help="Label field name in dataset (e.g. 0/1/2/3).")
    parser.add_argument(
        "--image_base_dir",
        action="append",
        default=[],
        help="Base directory to resolve relative image paths (can be passed multiple times).",
    )
    parser.add_argument("--train_split", default="train", help="Train split name.")
    parser.add_argument("--test_split", default="test", help="Test split name (optional).")
    parser.add_argument("--extra_test_split", default=None, help="Optional extra test split name.")
    parser.add_argument("--split_key", default=None, help="Column name storing split labels (e.g. original_split).")
    parser.add_argument("--only_test", action="store_true", help="Only export test split(s).")
    args = parser.parse_args()

    config_list = []
    if args.dataset_config:
        config_list = [cfg.strip() for cfg in args.dataset_config.split(",") if cfg.strip()]
    if not config_list:
        config_list = [None]

    for cfg_name in config_list:
        cfg_suffix = cfg_name.replace("/", "_") if cfg_name else None
        output_dir = args.output_dir
        if len(config_list) > 1 and cfg_suffix:
            output_dir = os.path.join(output_dir, cfg_suffix)

        data_dir, images_dir = resolve_output_dirs(output_dir)
        if args.path_prefix is None:
            path_prefix = f"data/{os.path.basename(data_dir)}/images"
        else:
            if len(config_list) > 1 and cfg_suffix:
                path_prefix = os.path.join(args.path_prefix, cfg_suffix)
            else:
                path_prefix = args.path_prefix

        ds = load_dataset(args.dataset_name, cfg_name) if cfg_name else load_dataset(args.dataset_name)
        first_split = next(iter(ds.keys()))
        print(ds[first_split].column_names)

        print(f"Writing images to: {images_dir}")
        os.makedirs(images_dir, exist_ok=True)

        train_split = args.train_split
        test_split = args.test_split
        extra_test_split = args.extra_test_split
        split_key = args.split_key
        if split_key:
            if train_split is None:
                raise ValueError("train_split must be set when using split_key.")
            sample = ds["train"][0]
        else:
            first_split = next(iter(ds.keys()))
            sample = ds[train_split][0] if train_split in ds else ds[first_split][0]
        question_key = args.question_key or pick_first(
            sample, ["question", "prompt", "problem", "instruction", "text", "formal_point"], "question"
        )
        answer_key = args.answer_key or pick_first(
            sample, ["answer", "solution", "target", "label", "response"], "answer"
        )
        label_key = args.label_key or ("label" if "label" in sample else None)
        add_choice_letters = args.choices_add_letters
        answer_from_choices = args.answer_from_choices

        rel_root = os.path.dirname(data_dir)
        base_dirs = list(args.image_base_dir)
        for split_name, split_ds in ds.items():
            for cache_item in split_ds.cache_files:
                cache_dir = os.path.dirname(cache_item.get("filename", ""))
                if cache_dir:
                    base_dirs.append(cache_dir)
                    base_dirs.append(os.path.dirname(cache_dir))
        if split_key:
            train_ds = ds["train"].filter(lambda x: x.get(split_key) == train_split)
            test_ds = None
            if test_split:
                test_ds = ds["train"].filter(lambda x: x.get(split_key) == test_split)
            extra_test_ds = None
            if extra_test_split:
                extra_test_ds = ds["train"].filter(lambda x: x.get(split_key) == extra_test_split)
        else:
            train_ds = ds[train_split] if train_split in ds else None
            test_ds = ds[test_split] if test_split and test_split in ds else None
            extra_test_ds = ds[extra_test_split] if extra_test_split and extra_test_split in ds else None

        if train_ds is not None and not args.only_test:
            train = train_ds.map(
                map_fn,
                with_indices=True,
                fn_kwargs={
                    "split": "train",
                    "image_out_dir": images_dir,
                    "rel_root": rel_root,
                    "path_prefix": path_prefix,
                    "base_dirs": base_dirs,
                    "question_key": question_key,
                    "answer_key": answer_key,
                    "choices_key": args.choices_key,
                    "subject_key": args.subject_key,
                    "label_key": label_key,
                    "add_choice_letters": add_choice_letters,
                    "answer_from_choices": answer_from_choices,
                },
                remove_columns=train_ds.column_names,
            )
            train.to_json(os.path.join(data_dir, "train.json"), force_ascii=False)

        if test_ds is not None and len(test_ds) > 0:
            test = test_ds.map(
                map_fn,
                with_indices=True,
                fn_kwargs={
                    "split": "test",
                    "image_out_dir": images_dir,
                    "rel_root": rel_root,
                    "path_prefix": path_prefix,
                    "base_dirs": base_dirs,
                    "question_key": question_key,
                    "answer_key": answer_key,
                    "choices_key": args.choices_key,
                    "subject_key": args.subject_key,
                    "label_key": label_key,
                    "add_choice_letters": add_choice_letters,
                    "answer_from_choices": answer_from_choices,
                },
                remove_columns=test_ds.column_names,
            )
            test.to_json(os.path.join(data_dir, "test.json"), force_ascii=False)
        else:
            print(f"Skip test split: '{test_split}' not found.")

        if extra_test_ds is not None and len(extra_test_ds) > 0:
            extra_test = extra_test_ds.map(
                map_fn,
                with_indices=True,
                fn_kwargs={
                    "split": "testmini",
                    "image_out_dir": images_dir,
                    "rel_root": rel_root,
                    "path_prefix": path_prefix,
                    "base_dirs": base_dirs,
                    "question_key": question_key,
                    "answer_key": answer_key,
                    "choices_key": args.choices_key,
                    "subject_key": args.subject_key,
                    "label_key": label_key,
                    "add_choice_letters": add_choice_letters,
                    "answer_from_choices": answer_from_choices,
                },
                remove_columns=extra_test_ds.column_names,
            )
            extra_test.to_json(os.path.join(data_dir, "testmini.json"), force_ascii=False)
        elif extra_test_split:
            print(f"Skip extra test split: '{extra_test_split}' not found.")


if __name__ == "__main__":
    main()
