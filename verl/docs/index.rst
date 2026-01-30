Welcome to verl's documentation!
================================================


verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM, vLLM and SGLang. Moreover, users can easily extend to other LLM training and inference frameworks.

- **Flexible device mapping and parallelism**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.



verl is fast with:

- **State-of-the-art throughput**: By seamlessly integrating existing SOTA LLM training and inference frameworks, verl achieves high generation and training throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

--------------------------------------------

.. _Contents:

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   start/install
   start/quickstart
   start/multinode
   start/ray_debug_tutorial
   start/more_resources

.. toctree::
   :maxdepth: 2
   :caption: Programming guide

   hybrid_flow
   single_controller

.. toctree::
   :maxdepth: 1
   :caption: Data Preparation

   preparation/prepare_data
   preparation/reward_function

.. toctree::
   :maxdepth: 2
   :caption: Configurations

   examples/config

.. toctree::
   :maxdepth: 1
   :caption: PPO Example

   examples/ppo_code_architecture
   examples/gsm8k_example
   examples/multi_modal_example

.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   algo/ppo.md
   algo/grpo.md
   algo/dapo.md
   algo/spin.md
   algo/sppo.md
   algo/entropy.md
   algo/opo.md
   algo/baseline.md

.. toctree::
   :maxdepth: 1
   :caption: PPO Trainer and Workers

   workers/ray_trainer
   workers/fsdp_workers
   workers/megatron_workers
   workers/sglang_worker

.. toctree::
   :maxdepth: 1
   :caption: Performance Tuning Guide

   perf/dpsk.md
   perf/perf_tuning
   README_vllm0.8.md
   perf/device_tuning
   perf/nsight_profiling.md

.. toctree::
   :maxdepth: 1
   :caption: Adding new models

   advance/fsdp_extension
   advance/megatron_extension

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advance/checkpoint
   advance/rope
   advance/ppo_lora.rst
   sglang_multiturn/multiturn.rst
   sglang_multiturn/interaction_system.rst
   advance/placement
   advance/dpo_extension
   examples/sandbox_fusion_example

.. toctree::
   :maxdepth: 1
   :caption: Hardware Support

   amd_tutorial/amd_build_dockerfile_page.rst
   amd_tutorial/amd_vllm_page.rst
   ascend_tutorial/ascend_quick_start.rst

.. toctree::
   :maxdepth: 1
   :caption: API References

   api/data
   api/single_controller.rst
   api/trainer.rst
   api/utils.rst


.. toctree::
   :maxdepth: 2
   :caption: FAQ

   faq/faq

.. toctree::
   :maxdepth: 1
   :caption: Development Notes

   sglang_multiturn/sandbox_fusion.rst

Contribution
-------------

verl is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.


Code Linting and Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: bash



.. code-block:: bash


Adding CI tests
^^^^^^^^^^^^^^^^^^^^^^^^

If possible, please add CI test(s) for your new feature:

1. Find the most relevant workflow yml file, which usually corresponds to a ``hydra`` default config (e.g. ``ppo_trainer``, ``ppo_megatron_trainer``, ``sft_trainer``, etc).
2. Add related path patterns to the ``paths`` section if not already included.
3. Minimize the workload of the test script(s) (see existing scripts for examples).

