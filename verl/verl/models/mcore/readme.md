# verl Megatron-Core Models

The migration has been successful with the help of the mcore team and the community. What we have done is:
1. update `Megatron` version to `0.11.0`
3. support sequence packing/thd format.
4. support `tensor parallel`, `pipeline parallel`, `sequence parallel`, `virtual pipeline parallel`, `context parallel`.

We are working on the following features:
- support `MixtralForCausalLM`
- support `DeepseekV3ForCausalLM`
- support `expert parallel`

Features we invite the community to contribute:
    - conversion of large models with multiple GPUs
    - conversion of large models with single GPU
- refactor the `megatron_checkpoint_manager.py` by `dist_checkpointing` format.
- support llama4


## How things work now
To engage the community in contributing, here are the key steps in our mcore integration process and features under development. 

main steps:
    - b. init the mcore `GPTModel` with the converted config
    - b. online resharding the mcore weights to rollout engine
        - this part is very complicated with multiple parallel strategies composition between mcore and rollout engine
3. support the mcore features in verl
    - a. support `tensor parallel`, `pipeline parallel`, `sequence parallel`, `virtual pipeline parallel`, `context parallel`
    - b. support recompute and other mcore speed up features

4. checkpointing
    - a. support recovering the verl training.


1. Runtime loading
    - speed is slow and memory consumption is high.
    - this way is deprecated and will not support new models.
2. Offline loading
    - online loading and sharding is automatically done by mcore `dist_checkpointing` format. The speed is fast and memory consumption is low.
    - the offline script is in `verl/scripts/converter_hf_to_mcore.py`.

See function `convert_megatron_model_to_transformers_model` in `verl/utils/megatron_utils.py` for the details.

It should be refatored for extensibility and better performance.

### support the mcore features in verl
Most of the features of `GPTModel` is out-of-the-box supported in verl through changing the `TransformerConfig`, except those about parallel strategies, such as `expert parallel`. 
Features about parallel strategies should be supported with changes about the online weights conversion(especially the resharding part) and verl work dispatching.

### checkpointing

The existing checkpoint format simply saves every rank's weights and optimizer states. It should be refactored by `dist_checkpointing` format.


## How to support new models
1. make sure the model is supported by vLLM
    - b. init the mcore `GPTModel` with the converted config
    - d. for VLM the interface might be different, it is ok to add a new model class with GPTModel as its module.
    - it is recommended to initialize a vLLM model with the converted mcore weights, and then test if the generating sequence is correct.


## How to scale up to larger models like deepseek-v3 or other 100B+ models
The greatest challenge for scaling up to larger models is the memory consumption.

The necessary features under development for scaling up are
1. Training engine part
    - expert parallel
2. Rollout engine part
    - pipeline parallel
    - expert parallel
    - more efficient and general weight resharding and loading
3. Offline weights conversion
    - support weights larger than single GPU memory
