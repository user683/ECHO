The Design of ``verl.single_controller``
==============================================

Last updated: 05/21/2025.


Preface
-------

We prepared this document for developers of ``verl``, particularly those
interested in understanding or contributing to the
``verl.single_controller`` module. It is not intended for end users, but
for contributors seeking to understand the architectural rationale and
internal mechanics.

--------------

Origin
------

The ``single_controller`` module originated from a request I received —
to adapt a toy single-process RLHF script into a distributed system with
minimal changes, while maintaining ease of debugging.

Common practice — such as using PyTorch’s Distributed Data Parallel
(DDP) — typically involves wrapping ``nn.Module`` and launching multiple
processes that execute the same function under different ranks. However,
distributed RLHF: - Difficulty representing multiple DAGs as required by
PPO; - Difficulty inspecting intermediate tensors during training.

To maintain debuggability, we opted for a different approach — breaking
the training loop into well-defined stages like ``generate_sequences``,
``compute_advantages``, and so on.

one RPC**, while training LLMs typically requires coordination across
multiple processes.

we introduced the following components:

-  ``WorkerGroup`` – manages a group of remote workers and provides
   a unified interface for multi-process distributed computation;
-  ``ResourcePool`` – binds computational resources to worker
   processes;
-  ``ClassWithArgs`` – enables delayed remote instantiation with
   specified initialization arguments.

--------------

A Running Example: ``generate_sequences``
-----------------------------------------

To illustrate the design, we walk through how the ``generate_sequences``
across distributed workers.

--------------

Step 1: Register with a Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to define the ``generate_sequences`` and decorate it

**Source:**

.. code:: python

   class ActorRolloutRefWorker(Worker):
       ...
       def generate_sequences(self, prompts: DataProto):
           prompts = prompts.to(torch.cuda.current_device())
           ...

attributes via a magic key (``MAGIC_ATTR``):

**Source:**

.. code:: python

   def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
       ...
       def decorator(func):
           def inner(*args, **kwargs):
               if materialize_futures:
                   args, kwargs = _materialize_futures(*args, **kwargs)
               return func(*args, **kwargs)

           attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
           setattr(inner, MAGIC_ATTR, attrs)
           return inner

       return decorator

As the code shows, values of ``dispatch_mode``, ``execute_mode`` and

--------------

Step 2: Binding During Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These attached attributes are extracted and utilized when
``ActorRolloutRefWorker``, wrapped in a ``RayClassWithArgs``, is passed
into a ``RayWorkerGroup``.

**Source:**

.. code:: python

   ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
   resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
   wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)

During the
of ``RayWorkerGroup``, two key steps occur:

1. Worker instances (Ray actors) are created:

   :alt: initialization_and_binding_of_worker_group

   initialization_and_binding_of_worker_group

The binding procedure is the heart of ``verl.single_controller``.

**Key function:**

.. code:: python

       ...
           try:
           except Exception:
               continue  # Skip properties
           <<<to be continue 1>>>


.. code:: python

           <<<continue 1>>>
               dispatch_mode = attribute["dispatch_mode"]
               execute_mode = attribute["execute_mode"]
               blocking = attribute["blocking"]

               <<<to be continue 2>>>

As show in the flow chart above, these attributes are fed into
``dispatch_fn``, ``collect_fn``, ``execute_fn``, ``blocking``. We need
to find the corresponding ``dispatch_fn`` and ``collect_fn`` associated
with the ``dispatch_mode`` (``DP_COMPUTE_PROTO``) from

.. code:: python3

   DISPATCH_MODE_FN_REGISTRY = {
       Dispatch.ONE_TO_ALL: {
           "dispatch_fn": dispatch_one_to_all,
           "collect_fn": collect_all_to_all,
       },
       ...
       Dispatch.DP_COMPUTE_PROTO: {
           "dispatch_fn": dispatch_dp_compute_data_proto,
           "collect_fn": collect_dp_compute_data_proto,
       },
       ...
   }

Similarly, the ``execute_fn`` is selected by ``execute_mode`` and
extracted by:

.. code:: python

               <<<continue 2>>>
               # get execute_fn_name
               execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
               wg_execute_fn_name = execute_mode["execute_fn_name"]

               # get execute_fn from string
               try:
                   execute_fn = getattr(self, wg_execute_fn_name)
                   assert callable(execute_fn), "execute_fn must be callable"
               except Exception:
                   print(f"execute_fn {wg_execute_fn_name} is invalid")
                   raise
               <<<to be continue 3>>>

In this ``generate_sequences`` cases: -
``dispatch_mode = Dispatch.DP_COMPUTE_PROTO`` -
``dispatch_fn = dispatch_dp_compute_data_proto`` -
``collect_fn = collect_dp_compute_data_proto`` -
``execute_fn = RayWorkerGroup.execute_all``

ONE_TO_ALL v.s. DP_COMPUTE_PROTO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``dispatch_mode`` is associated with a ``dispatch_fn`` and a
``collect_fn``. As the name implies, ``dispatch_fn`` processes the input
arguments in ``WorkerGroup`` and generate a batch (list) of input
arguments, each of which will be fed into a worker attached to the
``WorkerGroup``.

``dispatch_fn`` of ``ONE_TO_ALL`` is
which just duplicates all the input arguments into N replicas, where N
equals the number of Workers attached to the ``worker_group``:

.. code:: python

   def dispatch_one_to_all(worker_group, *args, **kwargs):
       args = tuple([arg] * worker_group.world_size for arg in args)
       kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
       return args, kwargs

``dispatch_fn`` of ``DP_COMPUTE_PROTO`` is
which uses ``DataProto.chunk`` to split a large ``DataProto`` into N
smaller ``DataProto``, where N equals the world_size (number of the
workers) of the ``worker_group``:

.. code:: python

   def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
       from verl.single_controller.base.worker_group import WorkerGroup

       assert isinstance(worker_group, WorkerGroup)
       # Note: enable auto padding for dp compute DatapProto
       splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
           worker_group.world_size,
           *args,
           **kwargs,
       )
       return splitted_args, splitted_kwargs

The ``collect_fn`` follows the same pattern and process a batch (list)
of returned value from all workers of a ``WorkerGroup`` and merge it
into a list as ``collect_all_to_all`` does or a large ``DataProto`` as
``collect_dp_compute_data_proto`` does.

and added to the ``WorkerGroup`` instance:

.. code:: python

               <<<continue 3>>>
               func = func_generator(
                   self,
                   dispatch_fn=dispatch_fn,
                   collect_fn=collect_fn,
                   execute_fn=execute_fn,
                   blocking=blocking,
               )

               try:
               except Exception as e:


--------------

Step 3: Call Chain
~~~~~~~~~~~~~~~~~~

All the machinery above ensures that distributed calls feel identical to
single-process ones. In the original single-process script, the code
looks like:

.. code:: python

   rollout = Rollout()
   rollout.generate_sequences(batch)

With ``verl``, the multiprocess program becomes:

.. code:: python

   rollout = RayWorkerGroup(resource_pool=[4], RayClassWithArgs(Rollout))
   rollout.generate_sequences(batch)

   :alt: call_chain_of_generate_sequences

   call_chain_of_generate_sequences

Behind this simple call: - ``dispatch_fn`` splits input across workers -
``execute_fn`` performs the actual remote invocation - ``collect_fn``
gathers the results

All of this is abstracted away, enabling developers to write distributed
code with minimal changes to their existing logic.

--------------

Beyond RL Post-Training: Generalizing ``verl.single_controller``
----------------------------------------------------------------

The ``verl.single_controller`` module generalizes well beyond
reinforcement learning. It provides a clean abstraction to batch-process

By minimizing the gap between single-process and multi-process scripts,
``verl.single_controller`` opens the door to distributed computing in

We hope this design inspires more examples and extensions from the
community.
