# Proximal Policy Optimization (PPO)

Last updated: 06/19/2025.



- High variance and sample inefficiency.
- Instability due to large policy updates.

PPO addresses this problem using a clipped surrogate objective that avoids overly large updates without requiring second-order derivatives.


## Key Components

- Actor-Critic Architecture: PPO requires both an actor model (policy) and a critic model (value function). This differs from other algorithms like GRPO and RLOO that don't require a critic model.

- Generalized Advantage Estimation (GAE): PPO uses GAE for computing advantage values, which helps reduce variance in policy gradient estimates while maintaining low bias.


## Configuration

Note that all configs containing `micro_batch_size` are used to configure the maximum sample or token count per forward or backward pass to avoid GPU OOMs, whose value should not change algorithmic/convergence behavior.



- `data.train_batch_size`: The global batch size of prompts used to generate a set of sampled trajectories/rollouts. The number of responses/trajectories is `data.train_batch_size * actor_rollout.ref.rollout.n`

- `actor_rollout_ref.actor.ppo_mini_batch_size`: The set of sampled trajectories is split into multiple mini-batches with batch_size=ppo_mini_batch_size for PPO actor updates. The ppo_mini_batch_size is a global size across all workers

- `actor_rollout_ref.critic.ppo_mini_batch_size`: The set of sampled trajectories is split into multiple mini-batches with batch_size=ppo_mini_batch_size for PPO critic updates. The ppo_mini_batch_size is a global size across all workers

- `actor_rollout_ref.actor.clip_ratio`: The PPO clip range. Default to 0.2

- `actor_rollout_ref.actor.ppo_epochs`: Number of epochs for PPO updates on one set of sampled trajectories for actor

- `critic.ppo_epochs`: Number of epochs for PPO updates on one set of sampled trajectories for critic. Defaults to `actor_rollout_ref.actor.ppo_epochs`

- `algorithm.gemma`: discount factor

- `algorithm.lam`: The lambda term that trades off between bias and variance in the GAE estimator

- `algorithm.adv_estimator`: Support gae, grpo, reinforce_plus_plus, reinforce_plus_plus_baseline, rloo

## Advanced Extensions

### KL Divergence Control


Options to use KL loss for KL divergence control: 

- `actor_rollout_ref.actor.use_kl_loss`: to use kl loss in the actor. When used, we are not applying KL in the reward function. Default is False

- `actor_rollout_ref.actor.kl_loss_coef`: The coefficient of kl loss. Default is 0.001.


Options to use KL penalty in the reward:



- `algorithm.kl_ctrl.kl_coef`: The (initial) coefficient of in-reward kl_penalty. Default is 0.001.
- `algorithm.kl_ctrl.type`: 'fixed' for FixedKLController and 'adaptive' for AdaptiveKLController.
- `algorithm.kl_ctrl.horizon`: See source code of AdaptiveKLController for details.
- `algorithm.kl_ctrl.target_kl`: See source code of AdaptiveKLController for details.

### Dual-clip PPO

The Dual-Clip PPO introduces a approach by applying a lower bound to the policy ratio when the advantage is less than zero, when multiplied by a large raito, does not exceed a specified lower bound.


- `actor_rollout_ref.actor.clip_ratio_c`: lower bound of the value for Dual-clip PPO, defaults to 3.0

## Reference Example


```bash
bash run_gemma.sh
  trainer.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  trainer.logger=['console'] \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=2 \
  critic.ppo_micro_batch_size=2
```

Reference performance with verl v0.2:

|-------------------------------|------------------|-------|------------------------------------------------------------------------------------------------|
