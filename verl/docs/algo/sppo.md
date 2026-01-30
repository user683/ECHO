# Recipe: Self-Play Preference Optimization (SPPO)

Last updated: 05/28/2025.





## Reproduce the Experiment


```
cd verl
python3 -m uv pip install -e ".[sglang]"


python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math

export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Note that the installation would occasionally fail to install flash-attn. If this happens, you can install it manually by running:

```bash
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

## Acknowledgement

We sincerely thank the contribution and guidance from:

