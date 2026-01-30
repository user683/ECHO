verl x Ascend
===================================

Last updated: 06/17/2025.

We added support for Ascend devices in verl.

Supported hardware
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc


Installation
-----------------------------------

Base environment setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------+-------------+
| software  | version     |
+-----------+-------------+
| Python    | == 3.10     |
+-----------+-------------+
| CANN      | == 8.1.RC1  |
+-----------+-------------+
| torch     | == 2.5.1    |
+-----------+-------------+
| torch_npu | == 2.5.1.RC1|
+-----------+-------------+


vllm & vllm-ascend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use vllm in verl, build and install vllm and vllm-ascend with the following commands. Choose the installation steps based on your machine type.

.. code-block:: bash
    
    # vllm
    cd vllm
    pip install -r requirements-build.txt

    # for Atlas 200T A2 Box16
    
    # for Atlas 900 A2 PODc
    VLLM_TARGET_DEVICE=empty pip install -e .

.. code-block:: bash
    
    # vllm-ascend
    cd vllm-ascend
    export COMPILE_CUSTOM_KERNELS=1
    python setup.py install

Install verl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd verl
    pip install -r requirements-npu.txt
    pip install -e .

Notes on other third-party libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------+---------------+
| software     | description   |
+--------------+---------------+
| transformers | v4.52.4       |
+--------------+---------------+
| flash_attn   | not supported |
+--------------+---------------+
| liger-kernel | not supported |
+--------------+---------------+
| tensordict   | 0.8.3 (ARM)   |
+--------------+---------------+

1. You can enable `--flash_attention_2` via transformers; transformers must be >= 4.52.0.
2. flash_attn is not supported for flash attention acceleration.
3. liger-kernel is not supported.
4. On ARM servers, tensordict must be 0.8.3; install it manually after dependencies if needed.
5. On x86 servers, install the CPU version of torchvision.

.. code-block:: bash



Quickstart
-----------------------------------

1. Download the dataset and preprocess it into parquet to include required fields for RL reward computation.

.. code-block:: bash

    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

2. Run training

.. code-block:: bash

    set -x

    export VLLM_ATTENTION_BACKEND=XFORMERS

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$HOME/data/gsm8k/test.parquet \
        data.train_batch_size=128 \
        data.max_prompt_length=512 \
        data.max_response_length=128 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl_grpo_example_gsm8k' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.total_epochs=1 \
        trainer.device=npu $@


Current support status
-----------------------------------

+-----------+-------------------------+-------------+-------------------+----------------------+
| algorithm |         model           | rewards mae |  throughput ratio |        hardware      |
+-----------+-------------------------+-------------+-------------------+----------------------+
+-----------+-------------------------+-------------+-------------------+----------------------+
+-----------+-------------------------+-------------+-------------------+----------------------+
+-----------+-------------------------+-------------+-------------------+----------------------+
+-----------+-------------------------+-------------+-------------------+----------------------+
+-----------+-------------------------+-------------+-------------------+----------------------+
+-----------+-------------------------+-------------+-------------------+----------------------+
+-----------+-------------------------+-------------+-------------------+----------------------+

Accuracy comparison notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Based on experience, for RL algorithms such as GRPO, we expect the mean absolute reward error to be <= 4% between Ascend devices and A100 under the same configuration. The calculation follows the figure above.


Throughput comparison notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For Ascend NPU and A100, take the average of the first 4 steps' "perf/throughput" in the logs. Throughput ratio = average NPU / average A100.



Plan
-----------------------------------




Disclaimer
-----------------------------------
The Ascend support code provided in verl is for reference only. For commercial use, please contact the official channels.
