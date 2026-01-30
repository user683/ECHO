# Dataset Format
## RLHF dataset
We combine all the data sources into a single parquet files. We directly organize the prompt into the chat format so that multi-turn chats can be easily incorporated. In the prompt, we may add instruction following texts to guide the model output the answers in a particular format so that we can extract the answers.

Math problems
```json
{
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": ["72"]
    },
}
```
