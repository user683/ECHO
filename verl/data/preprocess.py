import os

import datasets


def make_map_fn(
    split: str,
    source: str | None = None,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    image_key: str = "images",
    video_key: str = "videos",
    auto_insert_mm_token: bool = True,
):
    def process_fn(example, idx):
        if source is None:
            data_source = example.pop("source")
        else:
            data_source = source

        question = example.pop(prompt_key)
        solution = example.pop(answer_key)

        images = example.pop(image_key, None)
        videos = example.pop(video_key, None)

        # RLHFDataset will parse "<image>/<video>" placeholders into multimodal segments
        # when corresponding {images|videos} columns exist.
        if auto_insert_mm_token:
            if images and "<image>" not in question:
                question = "<image>\n" + question
            if videos and "<video>" not in question:
                question = "<video>\n" + question

        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": f"{data_source}-{idx}",
            },
        }

        if images is not None:
            data[image_key] = images
        if videos is not None:
            data[video_key] = videos

        return data

    return process_fn

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Convert json data to verl parquet format (supports multimodal).")
    parser.add_argument("--data_source", default="geometry3k", help="Dataset directory name (default: AIME-TTT)")
    parser.add_argument("--train_json", default=None, help="Override train json path")
    parser.add_argument("--test_json", default=None, help="Override test json path")
    parser.add_argument("--train_parquet", default=None, help="Override output train parquet path")
    parser.add_argument("--test_parquet", default=None, help="Override output test parquet path")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--image_key", default="images")
    parser.add_argument("--video_key", default="videos")
    parser.add_argument(
        "--no_auto_insert_mm_token",
        action="store_true",
        help="Do not auto prepend <image>/<video> when multimodal columns exist.",
    )
    args = parser.parse_args()

    data_source = args.data_source
    train_json = args.train_json or os.path.join(data_source, "train.json")
    test_json = args.test_json or os.path.join(data_source, "test.json")
    train_parquet = args.train_parquet or os.path.join(data_source, "train.parquet")
    test_parquet = args.test_parquet or os.path.join(data_source, "test.parquet")

    train_dataset = datasets.load_dataset("json", data_files=train_json, split="train")
    test_dataset = datasets.load_dataset("json", data_files=test_json, split="train")

    map_kwargs = dict(
        source=data_source,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        image_key=args.image_key,
        video_key=args.video_key,
        auto_insert_mm_token=not args.no_auto_insert_mm_token,
    )

    train_dataset = train_dataset.map(function=make_map_fn("train", **map_kwargs), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", **map_kwargs), with_indices=True)

    train_dataset.to_parquet(train_parquet)
    test_dataset.to_parquet(test_parquet)
