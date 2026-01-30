#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.sampling_metadata import SamplingMetadata
else:
    try:
        from vllm.model_executor.sampling_metadata import SamplingMetadata
    except Exception:
        try:
            from vllm.model_executor.layers.sampling_metadata import SamplingMetadata  # type: ignore
        except Exception:
            class SamplingMetadata:  # type: ignore
                pass
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.deepconf_branching import DeepConfBranchController

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
# 预处理输入的token id序列，去除左侧的padding部分
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

# 将输入的张量或数组按指定次数在第0维方向重复
def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


@dataclass
class CompletionRecord:
    parent_idx: int
    tokens: List[int]
    logprob_steps: List[Dict[int, float]]
    logprob_values: List[float]
    pruned: bool = False
    prune_penalty: float = 0.0
    prune_reason: Optional[str] = None


@dataclass
class BranchRequest:
    parent_idx: int
    prefix_tokens: List[int]
    branch_token: int
    branch_logprob: float
    prompt_token_ids: List[int]
    branch_logprob_step: Dict[int, float]



class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)  # 控制是否允许执行远程代码
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format # 指定模型权重和加载格式

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )  # 从配置中提取vLLM引擎的额外参数

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}  # 过滤掉值为None的参数
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")} 
            # 允许模型处理包含图像的输入数据，限制每个prompt最多可以包含几张图片

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )
        # kwargs参数初始化

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.branch_controller: Optional[DeepConfBranchController] = None
        if getattr(config, "deepconf", None) and config.deepconf.get("enable", False):
            deepconf_cfg = OmegaConf.to_container(config.deepconf, resolve=True)
            self.branch_controller = DeepConfBranchController(deepconf_cfg)
            self.sampling_params.n = 1
            if not self.sampling_params.logprobs or self.sampling_params.logprobs < self.branch_controller.cfg.logprob_top_k:
                self.sampling_params.logprobs = self.branch_controller.cfg.logprob_top_k
            self.config.calculate_log_probs = True

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        try:
            yield
        finally:
            # roll back to previous sampling params even if the caller raises
            for key, value in old_sampling_params_args.items():
                setattr(self.sampling_params, key, value)

    def _build_lora_requests(self, count: int):
        if not self.lora_kwargs or count <= 0:
            return None
        lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
        if not lora_int_ids:
            return None
        lora_int_id = lora_int_ids[0]
        return [
            LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
        ] * count

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        if self.branch_controller is not None:
            return self._generate_sequences_deepconf(prompts, **kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        prompt_token_id_lists = non_tensor_batch.pop("raw_prompt_ids")
        multi_modal_list = non_tensor_batch.pop("multi_modal_data") if "multi_modal_data" in non_tensor_batch else None

        lora_requests = self._build_lora_requests(batch_size)
        vllm_inputs = []
        for idx_prompt, raw_prompt_ids in enumerate(prompt_token_id_lists):
            if isinstance(raw_prompt_ids, np.ndarray):
                prompt_ids = raw_prompt_ids.tolist()
            else:
                prompt_ids = list(raw_prompt_ids)
            entry = {"prompt_token_ids": prompt_ids}
            if multi_modal_list is not None:
                entry["multi_modal_data"] = multi_modal_list[idx_prompt]
            if lora_requests is not None:
                entry["lora_request"] = lora_requests[idx_prompt]
            vllm_inputs.append(entry)

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=None,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(
                        non_tensor_batch["tools_kwargs"], self.sampling_params.n
                    )
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(
                        non_tensor_batch["interaction_kwargs"], self.sampling_params.n
                    )
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt"], self.sampling_params.n
                    )

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _generate_sequences_deepconf(self, prompts: DataProto, **kwargs) -> DataProto:
        t_start = time.time()
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        prompt_token_id_lists = non_tensor_batch.pop("raw_prompt_ids")
        multi_modal_list = non_tensor_batch.pop("multi_modal_data") if "multi_modal_data" in non_tensor_batch else None

        lora_requests = self._build_lora_requests(batch_size)
        vllm_inputs = []
        prompt_token_ids_cache = []
        for idx_prompt, raw_prompt_ids in enumerate(prompt_token_id_lists):
            if isinstance(raw_prompt_ids, np.ndarray):
                prompt_ids = raw_prompt_ids.tolist()
            else:
                prompt_ids = list(raw_prompt_ids)
            entry = {"prompt_token_ids": prompt_ids}
            if multi_modal_list is not None:
                entry["multi_modal_data"] = multi_modal_list[idx_prompt]
            if lora_requests is not None:
                entry["lora_request"] = lora_requests[idx_prompt]
            vllm_inputs.append(entry)
            prompt_token_ids_cache.append(prompt_ids)

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            sampling_kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            sampling_kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        else:
            sampling_kwargs = {}

        requested_n = None
        user_kwargs = prompts.meta_info.get("kwargs")
        if user_kwargs:
            user_kwargs = dict(user_kwargs)
            requested_n = user_kwargs.pop("n", None)
            sampling_kwargs.update(user_kwargs)

        target_count = None
        if requested_n is not None:
            try:
                target_count = max(int(requested_n), 1)
            except Exception:
                target_count = 1
        else:
            if is_validate:
                default_n = getattr(self.config.val_kwargs, "n", None)
                if default_n is None:
                    default_n = getattr(self.config, "n", 1)
            elif not do_sample:
                default_n = 1
            else:
                default_n = getattr(self.config, "n", 1)
            try:
                target_count = max(int(default_n), 1)
            except Exception:
                target_count = 1

        sampling_kwargs["n"] = 1

        with self.update_sampling_params(**sampling_kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=None,
                use_tqdm=False,
            )

        completions_per_prompt: List[List[CompletionRecord]] = [[] for _ in range(batch_size)]
        max_response_len = self.config.response_length

        def _make_completion_from_sample(prompt_idx: int, sample):
            token_ids_full = list(sample.token_ids)
            token_ids = token_ids_full[:max_response_len]
            logprob_steps_full = self._extract_logprob_steps(sample.logprobs)
            logprob_values_full = self._gather_token_logprobs(token_ids_full, sample.logprobs)
            logprob_steps = logprob_steps_full[: len(token_ids)]
            logprob_values = logprob_values_full[: len(token_ids)]
            return CompletionRecord(
                parent_idx=prompt_idx,
                tokens=token_ids,
                logprob_steps=logprob_steps,
                logprob_values=logprob_values,
            )

        def _generate_additional_completions(deficits: List[Tuple[int, int]]):
            extra_inputs = []
            extra_mapping = []
            for prompt_idx, deficit in deficits:
                prompt_tokens = prompt_token_ids_cache[prompt_idx]
                mm_data = None
                if multi_modal_list is not None:
                    mm_data = multi_modal_list[prompt_idx]
                for _ in range(deficit):
                    entry = {"prompt_token_ids": list(prompt_tokens)}
                    if mm_data is not None:
                        entry["multi_modal_data"] = mm_data
                    if lora_requests is not None:
                        entry["lora_request"] = lora_requests[prompt_idx]
                    extra_inputs.append(entry)
                    extra_mapping.append(prompt_idx)
            if not extra_inputs:
                return []
            with self.update_sampling_params(**sampling_kwargs):
                extra_outputs = self.inference_engine.generate(
                    prompts=extra_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=None,
                    use_tqdm=False,
                )
            records = []
            for idx, output in enumerate(extra_outputs):
                prompt_idx = extra_mapping[idx]
                if len(output.outputs) == 0:
                    continue
                for sample in output.outputs:
                    records.append((prompt_idx, _make_completion_from_sample(prompt_idx, sample)))
            return records

        for prompt_idx, output in enumerate(outputs):
            if len(output.outputs) == 0:
                continue
            sample = output.outputs[0]
            completions_per_prompt[prompt_idx].append(_make_completion_from_sample(prompt_idx, sample))

        branch_requests: List[BranchRequest] = []
        for prompt_idx, completions in enumerate(completions_per_prompt):
            if not completions:
                continue
            base_completion = completions[0]
            branch_plan = self.branch_controller.plan_branch(base_completion.logprob_steps, base_completion.tokens)
            if branch_plan is None:
                continue
            if branch_plan.position >= len(base_completion.tokens):
                continue
            base_token = base_completion.tokens[branch_plan.position]
            candidates = []
            for token_id in branch_plan.candidate_token_ids:
                if token_id not in branch_plan.branch_logprob_step:
                    continue
                if token_id not in candidates:
                    candidates.append(token_id)
            if base_token not in candidates:
                candidates.insert(0, base_token)
            candidates = candidates[: branch_plan.width]
            prefix_tokens = base_completion.tokens[: branch_plan.position]
            for token_id in candidates:
                if token_id == base_token:
                    continue
                branch_requests.append(
                    BranchRequest(
                        parent_idx=prompt_idx,
                        prefix_tokens=list(prefix_tokens),
                        branch_token=token_id,
                        branch_logprob=branch_plan.branch_logprob_step.get(token_id, -1e4),
                        prompt_token_ids=prompt_token_ids_cache[prompt_idx] + list(prefix_tokens) + [token_id],
                        branch_logprob_step=branch_plan.branch_logprob_step,
                    )
                )

        if branch_requests:
            branch_inputs = []
            for req in branch_requests:
                entry = {"prompt_token_ids": req.prompt_token_ids}
                if lora_requests is not None:
                    entry["lora_request"] = lora_requests[req.parent_idx]
                branch_inputs.append(entry)
            with self.update_sampling_params(**sampling_kwargs):
                branch_outputs = self.inference_engine.generate(
                    prompts=branch_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=None,
                    use_tqdm=False,
                )
            for req, output in zip(branch_requests, branch_outputs):
                if len(output.outputs) == 0:
                    continue
                sample = output.outputs[0]
                base_tokens = list(req.prefix_tokens) + [req.branch_token] + list(sample.token_ids)
                tokens = base_tokens[:max_response_len]
                logprob_steps_branch = [req.branch_logprob_step] + self._extract_logprob_steps(sample.logprobs)
                logprob_values_branch = [req.branch_logprob] + self._gather_token_logprobs(sample.token_ids, sample.logprobs)
                logprob_steps = logprob_steps_branch[: len(tokens)]
                logprob_values = logprob_values_branch[: len(tokens)]
                completions_per_prompt[req.parent_idx].append(
                    CompletionRecord(
                        parent_idx=req.parent_idx,
                        tokens=tokens,
                        logprob_steps=logprob_steps,
                        logprob_values=logprob_values,
                    )
                )

        min_required = target_count if target_count is not None else 1
        max_allowed = target_count

        for prompt_idx, completions in enumerate(completions_per_prompt):
            if not completions:
                continue
            pruned_records = []
            for completion in completions:
                prune_decision = self.branch_controller.should_prune(completion.logprob_steps)
                if prune_decision.should_prune and prune_decision.prune_pos is not None:
                    cutoff = min(prune_decision.prune_pos + 1, len(completion.tokens))
                    completion.tokens = completion.tokens[:cutoff]
                    completion.logprob_steps = completion.logprob_steps[:cutoff]
                    completion.logprob_values = completion.logprob_values[:cutoff]
                    completion.pruned = True
                    completion.prune_penalty = prune_decision.penalty
                    completion.prune_reason = prune_decision.reason
                pruned_records.append(completion)
            if max_allowed is not None and len(pruned_records) > max_allowed:
                pruned_records = pruned_records[:max_allowed]
            completions_per_prompt[prompt_idx] = pruned_records

        branch_counts = np.array([len(records) for records in completions_per_prompt], dtype=np.int32)
        deficits = []
        for prompt_idx in range(batch_size):
            deficit = max(min_required - branch_counts[prompt_idx], 0)
            if deficit > 0:
                deficits.append((prompt_idx, deficit))
        if deficits:
            extra_records = _generate_additional_completions(deficits)
            for prompt_idx, record in extra_records:
                completions_per_prompt[prompt_idx].append(record)

        branch_counts = np.array([len(records) for records in completions_per_prompt], dtype=np.int32)
        flat_completions: List[CompletionRecord] = []
        for records in completions_per_prompt:
            flat_completions.extend(records)

        if len(flat_completions) == 0:
            return DataProto(batch=None, non_tensor_batch=non_tensor_batch)

        response_list = [comp.tokens for comp in flat_completions]
        response = pad_2d_list_to_length(response_list, self.pad_token_id, max_length=self.config.response_length).to(
            idx.device
        )

        rollout_log_probs = None
        if self.config.calculate_log_probs:
            logprob_list = [
                comp.logprob_values for comp in flat_completions
            ]
            rollout_log_probs = pad_2d_list_to_length(
                logprob_list, -1, max_length=self.config.response_length
            ).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

        counts_tensor = torch.tensor(branch_counts.tolist(), device=idx.device, dtype=torch.long)
        repeated_idx = torch.repeat_interleave(idx, counts_tensor, dim=0)
        repeated_attention = torch.repeat_interleave(attention_mask, counts_tensor, dim=0)
        repeated_position = torch.repeat_interleave(position_ids, counts_tensor, dim=0)

        seq = torch.cat([repeated_idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(seq.size(0), -1)
        if repeated_position.dim() == 3:
            delta_position_id = delta_position_id.view(seq.size(0), 1, -1).expand(seq.size(0), 3, -1)
        response_position_ids = repeated_position[..., -1:] + delta_position_id
        position_ids = torch.cat([repeated_position, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((repeated_attention, response_attention_mask), dim=-1)

        repeated_non_tensor = {}
        for key, val in non_tensor_batch.items():
            repeated_non_tensor[key] = np.repeat(val, branch_counts, axis=0)
        repeated_non_tensor["raw_prompt_ids"] = np.repeat(prompt_token_id_lists, branch_counts, axis=0)
        if multi_modal_list is not None:
            repeated_non_tensor["multi_modal_data"] = np.repeat(multi_modal_list, branch_counts, axis=0)
        repeated_non_tensor["deepconf_pruned"] = np.array([comp.pruned for comp in flat_completions], dtype=bool)
        repeated_non_tensor["deepconf_prune_penalty"] = np.array(
            [comp.prune_penalty for comp in flat_completions], dtype=np.float32
        )

        batch = TensorDict(
            {
                "prompts": repeated_idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=response.shape[0],
        )
        if rollout_log_probs is not None:
            batch["rollout_log_probs"] = rollout_log_probs

        meta_info = dict(prompts.meta_info)
        meta_info["branch_counts"] = branch_counts
        meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        return DataProto(batch=batch, non_tensor_batch=repeated_non_tensor, meta_info=meta_info)

    @staticmethod
    def _extract_logprob_steps(logprob_entries):
        steps = []
        for step in logprob_entries:
            steps.append({token_id: info.logprob for token_id, info in step.items()})
        return steps

    @staticmethod
    def _gather_token_logprobs(token_ids, logprob_entries):
        values = []
        for token_id, step in zip(token_ids, logprob_entries):
            info = step.get(token_id)
            if info is None:
                values.append(float("-inf"))
            else:
                values.append(info.logprob)
        return values


def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
