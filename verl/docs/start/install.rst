Installation
============

Requirements
------------

- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.1

verl supports various backends. Currently, the following configurations are available:

- **FSDP** and **Megatron-LM** (optional) for training.
- **SGLang**, **vLLM** and **TGI** for rollout generation.

Choices of Backend Engines
----------------------------

1. Training:

We recommend using **FSDP** backend to investigate, research and prototype different models, datasets and RL algorithms. The guide for using FSDP backend can be found in :doc:`FSDP Workers<../workers/fsdp_workers>`.



2. Inference:

For inference, vllm 0.8.3 and later versions have been tested for stability. We recommend turning on env var `VLLM_USE_V1=1` for optimal performance.



Install from docker image
-------------------------

We provide pre-built Docker images for quick setup.

For vLLM with Megatron or FSDP, please use the stable version of image ``whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3``, which supports DeepSeek-V3 671B post-training.

For latest vLLM with FSDP, please refer to ``hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0``.

For SGLang with FSDP, please use ``ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post5`` which is provided by SGLang RL Group.

See files under ``docker/`` for NGC-based image or if you want to build your own.

1. Launch the desired Docker image and attach into it:

.. code:: bash

    docker start verl
    docker exec -it verl bash


2.	Inside the container, install latest verl:

.. code:: bash

    # install the nightly version (recommended)
    # pick your choice of inference engine: vllm or sglang
    # pip3 install -e .[vllm]
    # pip3 install -e .[sglang]
    # or install from pypi instead of git via:
    # pip3 install verl[vllm]
    # pip3 install verl[sglang]

.. note::

    The Docker image ``whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3`` is built with the following configurations:

    - **PyTorch**: 2.6.0+cu124
    - **CUDA**: 12.4
    - **cuDNN**: 9.8.0
    - **Flash Attenttion**: 2.7.4.post1
    - **Flash Infer**: 0.2.5
    - **vLLM**: 0.8.5
    - **SGLang**: 0.4.6.post5
    - **Megatron-LM**: core_v0.12.1
    - **TransformerEngine**: 2.3
    - **Ray**: 2.44.1

.. note::


Install from custom environment
---------------------------------------------

We recommend to use docker images for convenience. However, if your environment is not compatible with the docker image, you can also install verl in a python environment.


Pre-requisites
::::::::::::::

For training and inference engines to utilize better and faster hardware support, CUDA/cuDNN and other dependencies are required,
and some of the dependencies are easy to be overridden when installing other packages,
so we put them in the :ref:`Post-installation` step.

We need to install the following pre-requisites:

- **CUDA**: Version >= 12.4
- **cuDNN**: Version >= 9.8.0
- **Apex**

CUDA above 12.4 is recommended to use as the docker image,

.. code:: bash

    # change directory to anywher you like, in verl source code directory is not recommended
    apt-get update
    apt-get -y install cuda-toolkit-12-4
    update-alternatives --set cuda /usr/local/cuda-12.4


cuDNN can be installed via the following command,

.. code:: bash

    # change directory to anywher you like, in verl source code directory is not recommended
    apt-get update
    apt-get -y install cudnn-cuda-12

You can install it via the following command, but notice that this steps can take a very long time.
It is recommended to set the ``MAX_JOBS`` environment variable to accelerate the installation process,
but do not set it too large, otherwise the memory will be overloaded and your machines may hang.

.. code:: bash

    # change directory to anywher you like, in verl source code directory is not recommended
    cd apex && \
    MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./


Install dependencies
::::::::::::::::::::

.. note::

    We recommend to use a fresh new conda environment to install verl and its dependencies.


    As a countermeasure, it is recommended to install inference frameworks first with the pytorch they needed. For vLLM, if you hope to use your existing pytorch,
    please follow their official instructions


1. First of all, to manage environment, we recommend using conda:

.. code:: bash

   conda create -n verl python==3.10
   conda activate verl


2. Then, execute the ``install.sh`` script that we provided in verl:

.. code:: bash

    # Make sure you have activated verl conda env
    # If you need to run with megatron
    bash scripts/install_vllm_sglang_mcore.sh
    # Or if you simply need to run with FSDP
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh


If you encounter errors in this step, please check the script and manually follow the steps in the script.


Install verl
::::::::::::

For installing the latest version of verl, the best way is to clone and
install it from source. Then you can modify our code to customize your
own post-training jobs.

.. code:: bash

   cd verl
   pip install --no-deps -e .


Post-installation
:::::::::::::::::

Please make sure that the installed packages are not overridden during the installation of other packages.

The packages worth checking are:

- **torch** and torch series
- **vLLM**
- **SGLang**
- **pyarrow**
- **tensordict**

If you encounter issues about package versions during running verl, please update the outdated ones.


Install with AMD GPUs - ROCM kernel support
------------------------------------------------------------------

When you run on AMD GPUs (MI300) with ROCM platform, you cannot use the previous quickstart to run verl. You should follow the following steps to build a docker and run it. 

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

    #  Build the docker in the repo dir:
    # docker build -f docker/Dockerfile.rocm -t verl-rocm:03.04.2015 .
    # docker images # you can find your built docker

    # Set working directory
    # WORKDIR $PWD/app

    # Set environment variables
    ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

    # Install vllm
    RUN pip uninstall -y vllm && \
        rm -rf vllm && \
        cd vllm && \
        MAX_JOBS=$(nproc) python3 setup.py install && \
        cd .. && \
        rm -rf vllm

    # Copy the entire project directory
    COPY . .

    # Install dependencies
    RUN pip install "tensordict<0.6" --no-deps && \
        pip install accelerate \
        codetiming \
        datasets \
        dill \
        hydra-core \
        liger-kernel \
        numpy \
        pandas \
        datasets \
        peft \
        "pyarrow>=15.0.0" \
        pylatexenc \
        "ray[data,train,tune,serve]" \
        torchdata \
        transformers \
        orjson \
        pybind11 && \
        pip install -e . --no-deps

Build the image
::::::::::::::::::::::::

.. code-block:: bash

    docker build -t verl-rocm .

Launch the container
::::::::::::::::::::::::::::

.. code-block:: bash

    docker run --rm -it \
      --device /dev/dri \
      --device /dev/kfd \
      -p 8265:8265 \
      --group-add video \
      --cap-add SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --privileged \
      -v $HOME/.ssh:/root/.ssh \
      -v $HOME:$HOME \
      --shm-size 128G \
      -w $PWD \
      verl-rocm \
      /bin/bash

If you do not want to root mode and require assign yourself as the user,
Please add ``-e HOST_UID=$(id -u)`` and ``-e HOST_GID=$(id -g)`` into the above docker launch script. 

verl with AMD GPUs currently supports FSDP as the training engine, vLLM and SGLang as the inference engine. We will support Megatron in the future.
