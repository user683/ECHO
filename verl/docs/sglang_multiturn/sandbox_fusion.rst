===============================
Sandbox Fusion Tool Integration
===============================

Last updated: 06/10/2025.

Motivations
===========

- As users of verl, we want to allow the model to call certain tools during Actor rollout, incorporating the results into the training process.
- We aim to support tool-calling capabilities of inference engines using `sandbox-fusion` as the code execution system, providing the community with a reimplementation of `retools`.

Reward Compute with Sandbox Fusion + FaaS Integration
=====================================================

- In current datasets and tasks, similar work already exists (e.g., Prime), which uses local processes as runners to execute model-generated code for reward computation.
- On this basis, #1429 has advanced the design by integrating FaaS as the runner for reward computation.

Goals
=====

- Adapt to the `sglang` tool-calling protocol and define tools for sandbox fusion.
- Integrate with the `async-rollout` process, ensuring sandbox fusion tools follow asyncIO conventions.

Non-Goals
=========

- Training effectiveness is out of scope.
- Observability metrics are not considered.
- Distributed failover and component fault tolerance are not addressed.

Design Details
==============

Tool Schema Definition
----------------------

- Currently, only code execution is considered, requiring a `code` field in the JSON from the model.
- Only Python code is supported for now, so no `language` parameter is defined.

.. code-block:: python

   OpenAIFunctionToolSchema(
       type="function",
       function=OpenAIFunctionSchema(
           name="code_interpreter",
           description="A tool for executing code.",
           parameters=OpenAIFunctionParametersSchema(
               type="object",
               properties={
                   "code": OpenAIFunctionPropertySchema(
                       type="string",
                       description="The code to execute.",
                       enum=None,
                   )
               },
               required=["code"],
           ),
           strict=False,
       )
   )

Configuration Parameters
--------------------------

+----------------------------+--------------------------------------------------------------+
| Parameter Name             | Description                                                  |
+============================+==============================================================+
| `num_workers`              | Number of worker threads/processes per DP to request runner. |
+----------------------------+--------------------------------------------------------------+
+----------------------------+--------------------------------------------------------------+
| `default_timeout`          | Timeout (in seconds) for each code execution. Default: 30    |
+----------------------------+--------------------------------------------------------------+
| `default_language`         | Default programming language. Default: "python"              |
+----------------------------+--------------------------------------------------------------+
+----------------------------+--------------------------------------------------------------+
| `sandbox_fusion_url`       | URL for the veFaas sandbox execution service                 |
+----------------------------+--------------------------------------------------------------+

-----------------------

Objective:


- Ensure ordered submission to code runners to avoid starvation due to backoff.

Design Highlights:

- Use Ray Global Actor as a singleton distributed counter at cluster level.
  
- Semaphore used for counting, with `acquire` and `release` in separate thread pools to preserve order.
  
- Use Ray’s cloud-pickle to serialize functions for decoupled `ExecutionWorker`.

.. code-block:: python

   class TokenBucketWorker:
           self.current_count = 0

       def acquire(self):
           self._semaphore.acquire()
           self.current_count += 1

       def release(self):
           self._semaphore.release()
           self.current_count -= 1

       def get_current_count(self):
           return self.current_count

   class ExecutionWorker:


       def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
           with ExitStack() as stack:
               try:
                   return fn(*fn_args, **fn_kwargs)
               except Exception as e:
                   logger.warning(f"Error when executing code: {e}")

       if mode == PoolMode.ThreadMode:
           return ray.remote(ExecutionWorker).options(max_concurrency=num_workers).remote(
           )
       else:
           raise NotImplementedError("Process mode is not implemented yet")

Tool Implementation
-------------------

- Use `instance_id` to identify requests across multiple dialogue rounds.
  
- Use `execution_pool` to implement async invocation.
  
- Cleanup state after rollout completion.

.. code-block:: python

   class SandboxFusionTool(BaseTool):
       def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
           ...
           self.execution_pool = init_execution_pool(...)
           ...

       async def create(self, instance_id: Optional[str] = None, ...):
           ...

        async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
            code = parameters.get("code", "")
            timeout = parameters.get("timeout", self.default_timeout)
            language = parameters.get("language", self.default_language)
            if not isinstance(code, str):
                code = str(code)

            result = await self.execution_pool.execute.remote(self.execute_code,instance_id,code,timeout,language)
            self._instance_dict[instance_id]["reward"].append(result.strip())

            return result, result, {}

        def execute_code(self,instance_id,code,timeout=30,language="python"):
            # we should always expect this since we don't have correct answer
                return actual_output
            else:
                return "no stdout here"

       async def calc_reward(self, instance_id: str, ...):
           ...

       async def release(self, instance_id: str, ...):
           ...

Test Plan
=========

Unit Tests
----------

- **test_tools_registration**: Test tool registration and initialization.
- **test_rollout_req_creation**: Validate that `AsyncRolloutReq` is built correctly.
- **test_over_size_case**: Ensure rollout terminates early when exceeding `max_seq_len`.
- **test_tool_call_basic_case**: Mock `sglang` output, validate tool call and result.
- **test_tool_call_batch_case**: Test batch processing of tool calls.
- **test_basic_multi_process_init**: Validate Ray global actor behaves as singleton.

e2e Tests
----------
we provide e2e test scripts in `tests/special_e2e` folder, named `tests/special_e2e/run_gsm8k_fsdp_sgl_multiturn_sf_tool.sh`

by setting 'trainer.rollout_data_dir' you can dump the rollout data to local disk. here is an sample taken from the rollout data:

.. code-block:: python

   {
     "input": "
     
     system\nYou are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the `calc_gsm8k_reward` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary. Put your final answer in the format of `#### <answer>`.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"code_interpreter\", \"description\": \"A tool for executing code.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\", \"description\": \"The code to execute.\", \"enum\": null}}, \"required\": [\"code\"]}, \"strict\": false}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n
     
     
     assistant\n",
     
     
     tool\n220000.0\n\n
     
     "score": 0,
     "step": 1
   }

here is the readable format version:

.. code-block:: python

   [system]
   
   You are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the `calc_gsm8k_reward` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary. Put your final answer in the format of `#### <answer>`.
   
   # Tools
   
   You may call one or more functions to assist with the user query.
   
   You are provided with function signatures within <tools></tools> XML tags:
   <tools>
   {"type": "function", "function": {"name": "code_interpreter", "description": "A tool for executing code.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to execute.", "enum": null}}, "required": ["code"]}, "strict": false}}
   </tools>
   
   For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
   <tool_call>
   {"name": <function-name>, "arguments": <args-json-object>}
   </tool_call>
   
   [user]
   
   
   [assistant]
   
   <think>
   
   
   
   
   <tool_call>
   </tool_call>
   
   [tool]
   
   220000.0
   
   [assistant]
   
   <think>
   
   </think>
   
   #### 220000.0
