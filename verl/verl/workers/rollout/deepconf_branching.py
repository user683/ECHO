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
"""DeepConf entropy-confidence gating utilities for adaptive branching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class BranchPlan:
    position: int
    width: int
    candidate_token_ids: List[int]
    branch_logprob_step: Dict[int, float]


@dataclass
class PruneDecision:
    should_prune: bool
    prune_pos: Optional[int] = None
    reason: Optional[str] = None
    penalty: float = 0.0


@dataclass
class DeepConfParams:
    enable: bool = False
    min_depth: int = 4
    entropy_window: int = 4
    gentropy_roup_window: int = 4
    tail_window: int = 8
    tail_decline_steps: int = 3
    tail_threshold: float = 1.0
    delta_rise_steps: int = 3
    delta_upper: float = 0.5
    entropy_high: float = 3.5
    entropy_low: float = 1.0
    branch_threshold: float = 1.2
    entropy_delta_threshold: float = 0.2
    min_branches: int = 1
    max_branches: int = 4
    alpha: float = 2.0
    beta: float = 1.0
    prune_threshold: float = 0.4
    logprob_top_k: int = 64
    confidence_top_k: int = 5
    prune_lambda_1: float = 1.0
    prune_lambda_2: float = 1.0
    prune_lambda_3: float = 0.05


class DeepConfBranchController:
    """Implements DeepConf branching and pruning rules."""

    def __init__(self, cfg_dict: Dict):
        self.cfg = DeepConfParams(**cfg_dict)
        self._epsilon = 1e-9

    def plan_branch(
        self, logprob_steps: Sequence[Dict[int, float]], response_tokens: Sequence[int]
    ) -> Optional[BranchPlan]:
        if len(logprob_steps) == 0:
            return None
        stats = self._compute_step_stats(logprob_steps)
        cfg = self.cfg
        window = cfg.group_window
        min_depth = min(cfg.min_depth, len(stats) - 1)
        for idx in range(min_depth, len(stats)):
            stat = stats[idx]
            if (
                stat["entropy"] >= cfg.entropy_high
                and stat["group_conf"] <= cfg.branch_threshold
                and stat["delta_entropy"] >= cfg.entropy_delta_threshold
            ):
                width = self._compute_branch_width(stat["entropy"], stat["group_conf"])
                width = max(width, cfg.min_branches)
                width = min(width, cfg.max_branches)
                if width <= 1:
                    return None
                candidate_token_ids = stat["sorted_token_ids"][:width]
                return BranchPlan(
                    position=idx,
                    width=width,
                    candidate_token_ids=candidate_token_ids,
                    branch_logprob_step={k: v["logprob"] for k, v in stat["raw_logprobs"].items()},
                )
        return None

    def should_prune(self, logprob_steps: Sequence[Dict[int, float]]) -> PruneDecision:
        if len(logprob_steps) == 0:
            return PruneDecision(False)
        stats = self._compute_step_stats(logprob_steps)
        cfg = self.cfg
        running_min_group = np.inf
        tail_history: List[float] = []
        tail_decline_counter = 0
        delta_rise_counter = 0

        for idx, stat in enumerate(stats):
            running_min_group = min(running_min_group, stat["group_conf"])
            tail_history.append(stat["tail_conf"])
            if len(tail_history) >= 2 and tail_history[-1] < tail_history[-2]:
                tail_decline_counter += 1
            else:
                tail_decline_counter = 0

            if stat["delta_entropy"] > 0:
                delta_rise_counter += 1
            else:
                delta_rise_counter = 0

            if running_min_group < cfg.prune_threshold:
                penalty = self._compute_prune_penalty(stat)
                return PruneDecision(True, idx, "low_confidence", penalty)

            if (
                tail_decline_counter >= cfg.tail_decline_steps
                and stat["tail_conf"] <= cfg.tail_threshold
            ):
                penalty = self._compute_prune_penalty(stat)
                return PruneDecision(True, idx, "tail_decline", penalty)

            if delta_rise_counter >= cfg.delta_rise_steps and stat["delta_entropy"] > cfg.delta_upper:
                penalty = self._compute_prune_penalty(stat)
                return PruneDecision(True, idx, "entropy_spike", penalty)

        return PruneDecision(False)

    def _compute_branch_width(self, entropy: float, group_conf: float) -> int:
        cfg = self.cfg
        entropy_span = max(cfg.entropy_high - cfg.entropy_low, self._epsilon)
        term_entropy = cfg.alpha * (entropy - cfg.entropy_low) / entropy_span
        term_conf = cfg.beta * (group_conf - cfg.branch_threshold) / max(abs(cfg.branch_threshold), self._epsilon)
        raw = cfg.min_branches + term_entropy - term_conf
        return int(np.clip(np.round(raw), cfg.min_branches, cfg.max_branches))

    def _compute_step_stats(self, logprob_steps: Sequence[Dict[int, float]]) -> List[Dict[str, float]]:
        cfg = self.cfg
        entropies = []
        confidences = []
        c_tail_vals = []
        stats: List[Dict[str, float]] = []
        for idx, step in enumerate(logprob_steps):
            probs, sorted_tokens, logprobs_raw = self._convert_to_probs(step)
            entropy = self._entropy(probs)
            conf = self._confidence(probs)
            entropies.append(entropy)
            confidences.append(conf)
            tail_start = max(0, len(confidences) - cfg.tail_window)
            tail_slice = confidences[tail_start:]
            tail_val = float(np.mean(tail_slice)) if tail_slice else conf
            c_tail_vals.append(tail_val)
            group_conf = float(np.mean(confidences[max(0, len(confidences) - cfg.group_window) :]))
            prev_window = entropies[max(0, len(entropies) - cfg.entropy_window - 1) : len(entropies) - 1]
            bar_entropy = np.mean(prev_window) if prev_window else entropies[-1]
            delta_entropy = entropy - bar_entropy
            bottom_k = max(1, int(0.1 * len(probs)))
            c_bot10 = float(np.mean(np.sort(probs)[:bottom_k]))
            raw_log_dict = {token_id: {"logprob": float(logprob), "prob": float(prob)} for token_id, logprob, prob in zip(sorted_tokens, logprobs_raw, probs)}
            stats.append(
                {
                    "entropy": entropy,
                    "confidence": conf,
                    "tail_conf": tail_val,
                    "group_conf": group_conf,
                    "delta_entropy": delta_entropy,
                    "c_bot10": c_bot10,
                    "sorted_token_ids": sorted_tokens,
                    "raw_logprobs": raw_log_dict,
                }
            )
        return stats

    def _convert_to_probs(self, logprob_step: Dict[int, float]):
        sorted_items = sorted(logprob_step.items(), key=lambda kv: kv[1], reverse=True)
        token_ids = [token for token, _ in sorted_items][: self.cfg.logprob_top_k]
        logprobs = np.array([logprob_step[token] for token in token_ids], dtype=np.float64)
        max_logprob = np.max(logprobs) if len(logprobs) > 0 else 0.0
        probs = np.exp(logprobs - max_logprob)
        probs = probs / max(np.sum(probs), self._epsilon)
        return probs, token_ids, logprobs

    def _entropy(self, probs: np.ndarray) -> float:
        probs = np.clip(probs, self._epsilon, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def _confidence(self, probs: np.ndarray) -> float:
        top_k = min(len(probs), self.cfg.confidence_top_k)
        if top_k == 0:
            return 0.0
        top_probs = np.sort(probs)[-top_k:]
        return float(-np.mean(np.log(top_probs + self._epsilon)))

    def _compute_prune_penalty(self, stat: Dict[str, float]) -> float:
        cfg = self.cfg
        c_tail = stat["tail_conf"]
        c_bot10 = stat["c_bot10"]
        return cfg.prune_lambda_1 * c_tail - cfg.prune_lambda_2 * (1 - c_bot10) - cfg.prune_lambda_3
