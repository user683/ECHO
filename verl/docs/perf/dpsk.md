# Training DeepSeek 671b

Last updated: 06/13/2025.


In the journey the community added the following features and optimizations that enable verl with larger models:
- per tensor weight resharding between rollout and training
- context parallelism and expert parallelism enabled via megatron
- dynamic batch size (sequence balance) for megatron
- reduced ray-related serialization overhead
- optimizer offloading, recomputation, and efficient kernels
- various debugging metrics and utils

and the megatron backend now has a wider list of models supported:
- DeepSeek-V3
- Moonlight
- Mixtral

## Getting Started

### DeepSeek 671b



- vllm rollout with TP=32, bfloat16
- megatron training with attention DP, MoE EP=32, PP=16, bfloat16

MTP is disabled during RL training.



## Upcoming Optimizations

The community continue to optimize large MoE models further, ongoing efforts include:
- further optimizing memory consumption, and provide recommended/tuned configurations with various machine types
- optimizing long context RL training performance
- performance improvement with SGLang x Megatron


## Acknowledgement
