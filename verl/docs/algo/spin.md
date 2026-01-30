# Recipe: Self-Play Fine-Tuning (SPIN)

Last updated: 05/31/2025.

`verl` provides a recipe inspired by the paper **"Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"** (SPIN). SPIN is a language model finetuning algorithm that enables iterative self-improvement through a self-play mechanism inspired by game theory.

**Core Idea:** Models learn by playing against themselves, reducing reliance on external preference datasets or stronger teacher models:

1.  **Synthetic Data Generation:** The current model generates responses, creating its own training data from previous iterations.
2.  **Two-Player Game Setup:** A game involving two players acted by a single LLM.
3.  **Iterative Training:** The model progressively improves by refining its policy, with each iteration's model becoming the opponent for the next iteration.




---

## Key Function (compute_online_dpo_loss) and Related works
SPIN (Chen et al., 2024) proposes an iterative self-play mechanism to fine-tune language models. In each iteration, SPIN's training objective, when using a logistic loss function, is equivalent to Direct Preference Optimization (DPO) loss (Rafailov et al., 2023). 

This `verl` recipe realizes SPIN's core concept by using DPO loss iteratively (Xu et al., 2023; Xiong et al., 2023; Snorkel AI, 2024). This means that in each iteration, we fine-tune the LLM using DPO loss for preference optimization. Notably, Xu et al. (2023) explored iterative preference optimization with pairwise cringe loss, while Xiong et al. (2023) discussed how to bridge theory and practice for RLHF under KL constraints using iterative training. The concept of iterative preference learning was also explored in online DPO (Guo et al., 2024), which focuses on direct alignment from online AI feedback. In online DPO, preference data is dynamically updated during training, allowing the model to learn from its own generated data.

Specifically, we developed the **`compute_online_dpo_loss`** function and built this SPIN recipe on top of it. By incorporating online preference generation, this approach enables continuously refining language models without relying on fixed external preference datasets.

**Reference Papers:**


## Our Online DPO Implementation

Our `compute_online_dpo_loss` function adapts `verl`'s existing PPO infrastructure (based on `verl` v0.3.0.post1) for this iterative online DPO. Key aspects of our implementation include:

* **Dynamic Reference Model:** An explicit reference policy (`ref_policy_wg`) is used for DPO loss. This reference model's weights can be periodically updated from the actor (`ref_update_freq`), providing a dynamic baseline.
* **Online Preference Generation:** The `compute_onlineDPO_pref` function (in `core_algos.py`) dynamically creates chosen/rejected pairs based on a reward source (e.g., rule-based ranking for math problems).
* **DPO Loss Integration:** We replace PPO's policy loss with our `compute_online_dpo_loss` (in `core_algos.py`) within the actor update (`dp_actor.py`), directly optimizing the policy using the generated preferences.
* **Iterative Training Orchestration:** The `SpinTrainer` (in `spin_trainer.py`) manages the entire self-play loop: generation, preference labeling, optional reference model updates, and policy updates, enabling continuous self-improvement aligned with SPIN's principles.

---
## Algorithm

This recipe implements an Online algorithm adapted to the `verl` Reinforcement Learning framework, which provides an alternative to PPO for fine-tuning language models.

**Online Loop:** Instead of maximizing a scalar reward signal in PPO, this approach directly optimizes the policy model to align with preference data generated *online* during training:

1.  **Generation:** The current model generates multiple responses for each prompt in a batch.
2.  **Preference Labeling:** A function evaluates these generated responses to determine which one is preferred (chosen) and which is dispreferred (rejected). This can be done using a reward function or implicit ranking based on specific rules. (In this recipe, we use rule-based ranking on the math problem).
3.  **Update:** This preference tuple (`prompt`, `chosen_response`, `rejected_response`) is used to update the actor model using `compute_online_dpo_loss`, comparing against a reference model.

**Connection with SPIN:**

---

## Reproduce the Experiment (Example Setup)


1.  **Setup Environment (Example using Docker):**
    ```bash
    # Start a container with GPU access and shared memory
    docker run -it --name spin_test --gpus all \
        --shm-size=32g \
        --ipc=host \
        -v /path/to/host/.cache:/root/.cache \
        /bin/bash

    # Inside the container or on your host machine:
    # Ensure /tmp is writable
    mkdir -p /tmp
    chmod 1777 /tmp

    # Install Python 3.10 (if not present) and venv
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv tmux
    python3 -m ensurepip --upgrade

    # Create and activate a virtual environment
    python3 -m venv ~/.python/spin_env
    source ~/.python/spin_env/bin/activate

    # Install uv (fast package installer)
    python3 -m pip install uv
    ```

2.  **Install verl and Dependencies:**
    ```bash
    # Clone the verl repository and checkout the spin branch
    cd ~

    # Install flash-attn (handle potential build issues)
    python3 -m uv pip install wheel packaging
    python3 -m uv pip install flash-attn --no-build-isolation --no-deps

    # Install verl with sglang extras
    python3 -m uv pip install -e ".[sglang]"
    ```
    *Note: If `flash-attn` installation fails, try the manual steps again or consult its documentation.*

3.  **Login & Download Data/Model:**
    ```bash
    # Login to Weights & Biases (optional, for logging)

    # Download the GSM8K dataset
    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k # Adjusted path

    ```

4.  **Configure:**
    * Modify the configuration file (e.g., `config/spin_trainer.yaml` or the one specified in the run script) with correct paths to your downloaded model, data, desired hyperparameters (`dpo_beta`, learning rate, etc.), and distributed training settings (nodes, GPUs per node).
    * Pay attention to `actor_rollout_ref.model_path`, `data` paths, `reward_model` config (if using one), and `trainer.ref_update_freq`.

5.  **Run Training:**
    ```bash
    # Set CUDA visible devices (adjust based on your hardware and config)
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    # Launch the training script (e.g., test.sh or a custom script)
    # Ensure test.sh points to the correct config and main script
    bash recipe/spin/run_spin.sh
    ```

---

## Configuration

* The primary configuration is typically managed through a YAML file specified in the launch script (e.g., `config/spin_trainer.yaml`).
* Key configuration sections:
    * `data`: Paths to training/validation prompt files, batch sizes, sequence lengths.
    * `actor_rollout_ref`: Paths to the base model (used for actor and initial reference), FSDP settings, optimization parameters (learning rate, scheduler).
    * `algorithm`: DPO-specific hyperparameters like `dpo_beta`, `dpo_loss_type`.

---

## Key Files

* `main_spin.py`: Main entry point using Hydra to load the config and launch the `SpinTrainer`.
* `spin_trainer.py`: Defines the `SpinTrainer` class, orchestrating the Online DPO training loop.
* `fsdp_workers.py`: Implements Ray workers (Actor, Reference) potentially using FSDP.
* `dp_actor.py`: Contains the actor class, including the DPO policy update logic.
* `core_algos.py`: Includes helper functions for `compute_online_dpo_loss` and `compute_onlineDPO_pref`.
* `config/spin_trainer.yaml` (or similar): Main Hydra configuration file for the recipe.
* `run_spin.sh` (or similar): Example bash script for launching a training run.
* `README.md`: This file.

---

## Acknowledgement

We sincerely thank the contribution and guidance from the `verl` community and advisors, including (adapted from SPPO):

