# Algorithm Baselines

Last updated: 06/18/2025.

## Math related datasets

### GSM8k

Assuming GSM8k/math dataset is preprocessed via:

```bash
python3 examples/data_preprocess/*.py
```

Refer to the table below to reproduce RL training from different pre-trained checkpoints. Below is the performance on the GSM8k dataset if not specified otherwise. More comprehensive benchmark results areavailable in the recipe folder.


|-------------|----------------------------------|-------------------|--------------|---------|

### DAPO math-17k


Note:

|-------------|----------------------------------|-------------------|--------------|---------|



## Coding related datasets

Below is the result on leetcode if not specified otherwise.

|-------------|----------------------------------|-------------------|--------------|---------|


### Notes

[1] During evaluation, we have only extracted answers following the format `"####"`. A more flexible answer extraction, longer response length, and better prompt engineering may lead to a higher score.

[2] The default value of `actor_rollout_ref.actor.entropy_coeff` is set to `0.0` since verl 0.3.x on 2025-05-30, which is different from previous versions.
