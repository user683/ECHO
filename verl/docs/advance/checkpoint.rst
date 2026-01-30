.. _checkpoint-page:

Using Checkpoints to Support Fault Tolerance Training
=====================================================

Last updated: 06/25/2025.

There could be training errors or machine failure during the whole RLHF training process, 
so it is recommended to enable checkpoints to minimize your loss.

The API Interface has already been listed in :ref:`config-explain-page`,
and we will not repeat them. But there are still some technique details
we hope to clarify.

.. note:: 

    Notice that the ``checkpoint.contents`` field has no effect to FSDP checkpoint except ``hf_model``, 

Checkpoint Saving Directory Structure
-------------------------------------

Commonly, we use the ``default_local_dir`` declared in ``ppo_trainer.yaml`` or ``ppo_megatron_trainer.yml``
to work as preffix when saving checkpoints, which is ``checkpoints/${trainer.project_name}/${trainer.experiment_name}``.

So the inner checkpoint structure of **FSDP** is like:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   └── fsdp_config.json # FSDP config file, including world_size and fsdp version
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   ├── critic
    │   │   └── fsdp_config.json
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    └── latest_checkpointed_iteration.txt


While **Megatron** current checkpoint structure is:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   └── dist_ckpt       # save sharded model/optimizer/rng_states, naming the same as Megatron
    │   └── critic
    │   │   └── dist_ckpt
    └── latest_checkpointed_iteration.txt

-----------------------------------------------------------------

The tool is located in ``verl/model_merger``. For older versions of verl that don't include fsdp_config.json in checkpoints, you can use the legacy model merger located at ``verl/scripts/legacy_model_merger.py``.

The script supports two main sub-commands: `merge` (to convert and save checkpoints) and `test` (to validate merged checkpoints against a reference model).
The arguments for the `merge` sub-command are as follows:

.. code:: bash

    usage: python -m verl.model_merger merge [-h] --backend {fsdp,megatron} [--local_dir LOCAL_DIR] [--tie-word-embedding] [--is-value-model] [--use_cpu_initialization] [--target_dir TARGET_DIR]
                         [--hf_upload_path HF_UPLOAD_PATH] [--private]

    options:
    -h, --help            show this help message and exit
    --backend {fsdp,megatron}
                            The backend of the model
    --local_dir LOCAL_DIR
                            Path to the saved model checkpoints
    --use_cpu_initialization
    --target_dir TARGET_DIR
    --hf_upload_path HF_UPLOAD_PATH
                            Hugging Face repository ID to upload the model

Example usage for merging Megatron checkpoints:

.. code:: bash

    python -m verl.model_merger merge \
        --backend megatron \
        --tie-word-embedding \
        --target_dir /path/to/merged_hf_model

Example usage for merging FSDP checkpoints:

.. code:: bash

    python -m verl.model_merger merge \
        --backend fsdp \
        --target_dir /path/to/merged_hf_model


Megatron Merger details
-----------------------

Current implement of decoder layers uses ``nn.ModuleList`` to store the layers, 
and thus the model layers on every PP rank and VPP rank starts their index from 0.

There are 3 ways to correct this behavior:

1. Modify the decoder layer's state_dict, add ``offset`` to each layer's index, thus rewrite ``nn.ModuleList`` implementation.
2. Modify the layer index when saving checkpoint and recover them when loading checkpoint.
3. The Checkpoint merger do this work, calculate the actual ``offset`` from ``state_dict`` only, a little complex.

Current implementation use solution 2.


----------------------------------------------

If your model is quite huge, we recommend you to use Megatron dist-checkpoint to load the model.
Megatron dist-checkpoint supports loading with different kinds of model parallelism,
and it is much faster than the original checkpoint loading.

you can use the ``scripts/converter_hf_to_mcore.py`` script. Large MoE models are temporarily supported with CPU initialization,
which is a little slower. While we are working on a better solution to support large models.

Example command to convert the model is as follows:

.. code:: bash

    python scripts/converter_hf_to_mcore.py \
        --use_cpu_initialization    # Only work for MoE models


Original Checkpoint Utils
-------------------------

Original Checkpoint Utils refer to original checkpoint implementation in ``verl/models/[model]/megatron/checkpoint_utils``.

We only need ``[model]_loader.py`` in original checkpoint utils now, since we get rid of storing ``hf_model`` every time (which is not recommended for large model training, try only saving sharded models if you can).

.. note:: 

    Note that ``[model]_loader`` only support environments where **storage clusters are able to connect with every calculation nodes**. 
    Because it utilizes **sharded load way to minimize the loading checkpoint overhead**. 
    Every rank loads its own data from ``state_dict`` which can be accessed by all of them.
    While there is also no need to broadcast among DP ranks, since the saved state_dict is only produced by DP rank 0.


    To use deprecated loader, change the import package of ``load_state_dict_to_megatron_llama``.
