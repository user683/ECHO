verl performance tuning for AMD (ROCm Kernel)
=====================================================

Last updated: 04/25/2025.


Patch vLLM to Enable Sleep Mode for AMD GPUs
--------------------------------------------------------------

By default, verl requires vLLM to enable sleep mode, which allows vLLM to offload GPU memory to CPU memory after rollout. However, this feature is still under review by the vLLM community.


1. Clone the vLLM repository and build it with the following commands:

.. code-block:: bash

    cd vllm
    sudo ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so
    VLLM_TARGET_DEVICE=rocm ROCM_PATH=/opt/rocm/ VLLM_GPU_LANG=HIP SETUPTOOLS_SCM_PRETEND_VERSION=0.8.4.dev python3 setup.py develop



.. code-block:: python

	import torch
	from vllm import LLM


	def run_inference(prompt):
		outputs = llm.generate(prompt)
		for output in outputs:
			prompt = output.prompt
			generated_text = output.outputs[0].text
			print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


	print("CUDA Memory Usage (after inference):")
	torch.cuda.empty_cache()
	print(f"{torch.cuda.memory_allocated()=}")

	run_inference("San Francisco is")
	llm.sleep()

	print("CUDA Memory Usage (after sleep):")
	torch.cuda.empty_cache()
	print(f"{torch.cuda.memory_allocated()=}")

	llm.wake_up()

	print("CUDA Memory Usage (after wakeup):")
	torch.cuda.empty_cache()
	print(f"{torch.cuda.memory_allocated()=}")


If sleep mode is enabled, you should see the memory usage reduce after sleep.



Enable CUDA Graph and Bypass ROCm-related issues
--------------------------------------------------------------

Due to potential issues with CUDA graph capture in ROCm, we’ve found that vLLM’s CUDA graph feature cannot be enabled on multiple nodes in verl on AMD platforms with vLLM V1 mode. This leads to significantly slower rollout performance.


.. code-block:: python
	
    self.inference_engine = LLM(
        model=model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype=config.dtype,
        enforce_eager=config.enforce_eager,
        gpu_memory_utilization=config.gpu_memory_utilization,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=True,
        skip_tokenizer_init=False,
        max_model_len=max_model_len,
        load_format=load_format,
        disable_log_stats=config.disable_log_stats,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        enable_prefix_caching=True,
        trust_remote_code=trust_remote_code,
        # enable compilation config to bypass oom on rocm
	# change depends on your GPU memory size
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]},
    )


.. code-block:: bash

	actor_rollout_ref.rollout.enforce_eager=False \
