"""
Early stopping callback for GRPO training.

Monitors mean reward across all reward functions and stops training
when no improvement > min_delta for patience consecutive eval windows.
"""

import wandb
from transformers import TrainerCallback


class RewardEarlyStoppingCallback(TrainerCallback):
    """
    Monitors mean reward and stops training when it plateaus.

    Convergence is detected when the mean reward does not improve by more than
    `min_delta` for `patience` consecutive eval windows. All evidence is
    logged to W&B for reproducibility and report generation.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.01, warmup_steps: int = 200):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_steps = warmup_steps

        self.best_reward = float("-inf")
        self.best_reward_step = 0
        self.no_improve_count = 0
        self.reward_history = []
        self.should_stop = False
        self.stop_reason = ""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        reward_keys = [k for k in logs if "reward" in k.lower()]
        if not reward_keys:
            return

        step = state.global_step
        reward_vals = [logs[k] for k in reward_keys if isinstance(logs.get(k), (int, float))]
        if not reward_vals:
            return

        mean_reward = sum(reward_vals) / len(reward_vals)
        self.reward_history.append({
            "step": step,
            "mean_reward": mean_reward,
            "individual_rewards": {k: logs[k] for k in reward_keys if isinstance(logs.get(k), (int, float))},
        })

        wandb.log({
            "early_stop/mean_reward": mean_reward,
            "early_stop/best_reward": self.best_reward,
            "early_stop/no_improve_count": self.no_improve_count,
            "early_stop/patience_remaining": self.patience - self.no_improve_count,
        }, step=step)

        if step < self.warmup_steps:
            return

        if mean_reward > self.best_reward + self.min_delta:
            self.best_reward = mean_reward
            self.best_reward_step = step
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            self.should_stop = True
            self.stop_reason = (
                f"Early stopping triggered at step {step}. "
                f"Mean reward plateaued at {self.best_reward:.4f} (best at step {self.best_reward_step}). "
                f"No improvement > {self.min_delta} for {self.patience} consecutive eval windows."
            )
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING: {self.stop_reason}")
            print(f"{'='*70}\n")
            control.should_training_stop = True

    def log_evidence_to_wandb(self):
        """Log comprehensive early stopping evidence to W&B for reports."""
        evidence = {
            "early_stop/triggered": self.should_stop,
            "early_stop/reason": self.stop_reason if self.should_stop else "Completed all max_steps",
            "early_stop/best_reward": self.best_reward,
            "early_stop/best_reward_step": self.best_reward_step,
            "early_stop/total_reward_checks": len(self.reward_history),
            "early_stop/patience": self.patience,
            "early_stop/min_delta": self.min_delta,
            "early_stop/warmup_steps": self.warmup_steps,
        }
        wandb.summary.update(evidence)

        if self.reward_history:
            columns = ["step", "mean_reward"]
            if self.reward_history[0].get("individual_rewards"):
                individual_keys = sorted(self.reward_history[0]["individual_rewards"].keys())
                columns.extend(individual_keys)
            else:
                individual_keys = []

            rows = []
            for entry in self.reward_history:
                row = [entry["step"], entry["mean_reward"]]
                for k in individual_keys:
                    row.append(entry.get("individual_rewards", {}).get(k, 0.0))
                rows.append(row)

            wandb.log({
                "early_stop/reward_trajectory": wandb.Table(columns=columns, data=rows),
            })
