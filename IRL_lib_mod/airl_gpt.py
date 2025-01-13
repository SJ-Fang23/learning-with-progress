import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from pathlib import Path
from typing import Optional, Mapping, List, Tuple
from copy import deepcopy

from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial.common import compute_train_stats
from imitation.data import types
from imitation.rewards import reward_nets
import imitation.scripts.train_adversarial as train_adversarial

STOCHASTIC_POLICIES = (sac_policies.SACPolicy, policies.ActorCriticPolicy)


# -------------------------------------------------------------------
# 1) Example reward net that includes a "progress head"
#    (We demonstrate how you might extend RewardNet with an extra output.)
# -------------------------------------------------------------------
class RewardNetWithProgress(reward_nets.RewardNet):
    """
    A custom RewardNet that returns:
      - 'base' reward for AIRL (like your existing reward net)
      - a "progress_hat" for each state-action (used for request #3).
    """

    def __init__(self, observation_space, action_space, use_action=True, hidden_size=64):
        super().__init__(observation_space, action_space, use_action=use_action)

        # Simple MLP to produce both reward logits & progress
        # Adjust shapes/dimensions as needed
        in_dim = self.observation_space.shape[0]
        if use_action:
            in_dim += self.action_space.shape[0]

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.reward_head = nn.Linear(hidden_size, 1)
        self.progress_head = nn.Linear(hidden_size, 1)

    def base(self, states, actions, next_states, dones):
        """
        Returns the unshaped reward portion (like standard AIRL).
        We'll treat reward_head(...) as that unshaped reward.
        """
        x = self._concat_inputs(states, actions)
        feats = self.feature_extractor(x)
        reward_out = self.reward_head(feats).squeeze(-1)
        return reward_out

    def potential(self, states: th.Tensor) -> th.Tensor:
        """
        If you want potential shaping, you can define a function here.
        For now we can return zero or something minimal.
        """
        shape = states.shape[0]
        return th.zeros(shape, device=states.device)

    def forward(self, states, actions, next_states, dones) -> th.Tensor:
        """
        Standard call in AIRL: r(s,a).
        Typically we do [reward_out - log pi(a|s)], but that's handled externally.
        So here, just return base.
        """
        return self.base(states, actions, next_states, dones)

    def predict_progress(self, states, actions) -> th.Tensor:
        """
        A separate method to get the predicted progress from the 'progress_head'.
        This is used to handle request #3: progress regression in the net itself.
        """
        x = self._concat_inputs(states, actions)
        feats = self.feature_extractor(x)
        progress_hat = self.progress_head(feats).squeeze(-1)
        return progress_hat


# -------------------------------------------------------------------
# 2) Extended AIRL class that includes new shaping losses:
#    - Demonstration range constraints (#1)
#    - Subtrajectory progress regression (#2)
#    - A small regression head to predict progress (#3)
#    - Progress-based reward regularization (#4)
# -------------------------------------------------------------------
class AIRLWithProgress(common.AdversarialTrainer):
    """
    An AIRL trainer that:
      - handles your normal training
      - includes multiple shaping losses in train_disc
      - uses a custom reward net (like RewardNetWithProgress)
      - let's you pick which shaping losses to enable via shape_reward list:
         e.g. ["demo_range_loss", "progress_regression_loss", "progress_head_loss",
               "progress_regularization", ...]
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: RewardNetWithProgress,  # must have "predict_progress" etc.
        annotation_list: List[Tuple[dict, int]],  # (progress_dict, demonstration_idx)
        demonstrations_for_shaping: List[types.Trajectory],
        shape_reward=None,  # e.g. ["demo_range_loss", "progress_regression_loss", ...]
        shaping_batch_size: int = 16,
        shaping_loss_weight: float = 1.0,
        shaping_update_freq: int = 1,
        shaping_lr: float = 1e-3,
        save_model_every=20,
        save_path="checkpoints/default",
        alpha: float = 1.0,     # scaling factor for progress-based reg.
        lambda_reg: float = 0.1,  # weight for progress regularization
        progress_head_weight: float = 0.1,  # MSE weight for progress_head
        **kwargs,
    ):
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )
        if not isinstance(self.gen_algo.policy, STOCHASTIC_POLICIES):
            raise TypeError("AIRL needs a stochastic policy to compute r - log pi(a|s).")

        assert isinstance(demonstrations_for_shaping, list)
        assert isinstance(demonstrations_for_shaping[0], types.Trajectory)
        assert isinstance(annotation_list, list)

        self.demonstrations_for_shaping = deepcopy(demonstrations_for_shaping)
        self.annotation_list = annotation_list
        self.shaping_batch_size = shaping_batch_size
        self.shaping_loss_weight = shaping_loss_weight
        self.shaping_update_freq = shaping_update_freq
        self.shaping_lr = shaping_lr
        self.shape_reward = shape_reward if shape_reward else []
        self.save_model_every = save_model_every

        self.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_path = os.path.join(self.project_path, save_path)
        os.makedirs(self.save_path, exist_ok=True)

        # For progress-based terms
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.progress_head_weight = progress_head_weight
    def logits_expert_is_high(
            self,
            state: th.Tensor,
            action: th.Tensor,
            next_state: th.Tensor,
            done: th.Tensor,
            log_policy_act_prob: Optional[th.Tensor] = None,
        ) -> th.Tensor:
            r"""Compute the discriminator's logits for each state-action sample.

            In Fu's AIRL paper (https://arxiv.org/pdf/1710.11248.pdf), the
            discriminator output was given as

            .. math::

                D_{\theta}(s,a) =
                \frac{ \exp{r_{\theta}(s,a)} } { \exp{r_{\theta}(s,a)} + \pi(a|s) }

            with a high value corresponding to the expert and a low value corresponding to
            the generator.

            In other words, the discriminator output is the probability that the action is
            taken by the expert rather than the generator.

            The logit of the above is given as

            .. math::

                \operatorname{logit}(D_{\theta}(s,a)) = r_{\theta}(s,a) - \log{ \pi(a|s) }

            which is what is returned by this function.

            Args:
                state: The state of the environment at the time of the action.
                action: The action taken by the expert or generator.
                next_state: The state of the environment after the action.
                done: whether a `terminal state` (as defined under the MDP of the task) has
                    been reached.
                log_policy_act_prob: The log probability of the action taken by the
                    generator, :math:`\log{ \pi(a|s) }`.

            Returns:
                The logits of the discriminator for each state-action sample.

            Raises:
                TypeError: If `log_policy_act_prob` is None.
            """
            if log_policy_act_prob is None:
                raise TypeError(
                    "Non-None `log_policy_act_prob` is required for this method.",
                )
            reward_output_train = self._reward_net(state, action, next_state, done)
            return reward_output_train - log_policy_act_prob

    # ---------------------------------------------------
    #   1) Demonstration Range Loss (Request #1)
    # ---------------------------------------------------
    def demo_range_loss(self) -> th.Tensor:
        """
        If final progress of demo i is >= final progress of demo j + 10,
        => total reward of i >= 1.05 * total reward of j.

        If |progress_i - progress_j| < 10 => total rewards must be within ±5%.
        We sum over pairs, average, and return.
        """
        device = self.gen_algo.device

        # 1) gather random demonstration indices
        #    or gather them all if you prefer, but we do random for efficiency
        idxs = np.random.choice(len(self.demonstrations_for_shaping), size=self.shaping_batch_size, replace=False)

        # 2) get final progress and total reward for each
        final_progress = []
        total_rewards = []
        for i in idxs:
            # find final progress from annotation_list
            # We'll define "final" as the largest end_progress for that demonstration
            fp = None
            for (pdict, demo_i) in self.annotation_list:
                if demo_i == i:
                    if fp is None or pdict["end_progress"] > fp:
                        fp = pdict["end_progress"]
            if fp is None:
                fp = 0.0

            # compute total reward from reward_net
            traj = self.demonstrations_for_shaping[i]
            states = th.tensor(traj.obs, dtype=th.float32, device=device)
            actions = th.tensor(traj.acts, dtype=th.float32, device=device)
            next_states = th.tensor(traj.obs[1:], dtype=th.float32, device=device)
            if len(next_states) < len(states):
                # pad last
                next_states = th.cat([next_states, next_states[-1:].clone()], dim=0)
            dones = th.zeros(len(states), dtype=th.float32, device=device)

            with th.no_grad():
                r_vals = self._reward_net.base(states, actions, next_states, dones)
            R_sum = r_vals.sum().item()

            final_progress.append(fp)
            total_rewards.append(R_sum)

        p_tensor = th.tensor(final_progress, dtype=th.float32, device=device)
        r_tensor = th.tensor(total_rewards, dtype=th.float32, device=device)

        # 3) pairwise compare
        p_diff = p_tensor.unsqueeze(1) - p_tensor.unsqueeze(0)  # (n,n)
        r_diff = r_tensor.unsqueeze(1) - r_tensor.unsqueeze(0)  # (n,n)
        n = len(p_tensor)
        penalty = th.tensor(0.0, device=device)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if p_diff[i, j] >= 10.0:
                    # require r_i >= 1.05*r_j => r_i - 1.05*r_j >= 0
                    target = 1.05 * r_tensor[j]
                    if r_tensor[i] < target:
                        penalty += (target - r_tensor[i])
                elif p_diff[i, j] <= -10.0:
                    # require r_j >= 1.05*r_i
                    target = 1.05 * r_tensor[i]
                    if r_tensor[j] < target:
                        penalty += (target - r_tensor[j])
                else:
                    # if |p_diff|<10 => we want r_i in [0.95*r_j, 1.05*r_j]
                    up = 1.05 * r_tensor[j]
                    low = 0.95 * r_tensor[j]
                    if r_tensor[i] > up:
                        penalty += (r_tensor[i] - up)
                    elif r_tensor[i] < low:
                        penalty += (low - r_tensor[i])

        loss = penalty / (n * (n - 1))
        return loss

    # ---------------------------------------------------
    #   2) Regression to Subtrajectory Progress (Request #2)
    # ---------------------------------------------------
    def subtraj_progress_regression_loss(self) -> th.Tensor:
        """
        We gather random subtrajectories from annotation_list, compute the sum of rewards,
        and do MSE vs. the delta progress (or scaled).
        """
        device = self.gen_algo.device
        idxs = np.random.choice(len(self.annotation_list), size=self.shaping_batch_size, replace=False)

        dp_list = []
        reward_sum_list = []
        for idx in idxs:
            pdict, demo_idx = self.annotation_list[idx]
            start_p = pdict["start_progress"]
            end_p = pdict["end_progress"]
            delta_p = end_p - start_p
            dp_list.append(delta_p)

            t = self.demonstrations_for_shaping[demo_idx]
            s, e = pdict["start_step"], pdict["end_step"]
            states = th.tensor(t.obs[s:e], dtype=th.float32, device=device)
            actions = th.tensor(t.acts[s:e], dtype=th.float32, device=device)
            next_states = th.tensor(t.obs[s+1:e+1], dtype=th.float32, device=device)
            if len(next_states) < len(states):
                next_states = th.cat([next_states, next_states[-1:].clone()], dim=0)
            dones = th.zeros(len(states), dtype=th.float32, device=device)

            with th.no_grad():
                r_vals = self._reward_net.base(states, actions, next_states, dones)
            reward_sum_list.append(r_vals.sum().item())

        dp_tensor = th.tensor(dp_list, dtype=th.float32, device=device)
        rew_tensor = th.tensor(reward_sum_list, dtype=th.float32, device=device)

        # MSE => (rew_sum - alpha*dp)^2 or just (rew_sum - dp)^2
        scale = 1.0
        loss = F.mse_loss(rew_tensor, scale * dp_tensor)
        return loss

    # ---------------------------------------------------
    #   3) Regression Head for Progress (Request #3)
    #      We'll define a method that picks random states
    #      and does MSE with the net's predict_progress(...)
    # ---------------------------------------------------
    def progress_head_loss(self) -> th.Tensor:
        """
        We gather random states (or subtrajectories) with known progress
        from annotation_list, feed them to reward_net.predict_progress,
        and compute MSE with the average progress label.
        """
        device = self.gen_algo.device
        idxs = np.random.choice(len(self.annotation_list), size=self.shaping_batch_size, replace=False)
        all_states, all_actions, all_labels = [], [], []
        for idx in idxs:
            pdict, demo_idx = self.annotation_list[idx]
            avg_p = 0.5*(pdict["start_progress"] + pdict["end_progress"])

            t = self.demonstrations_for_shaping[demo_idx]
            # pick a single middle state for example
            mid = (pdict["start_step"] + pdict["end_step"]) // 2
            if mid >= len(t.obs):
                mid = len(t.obs) - 1
            st = th.tensor(t.obs[mid], dtype=th.float32)
            ac = th.tensor(t.acts[mid], dtype=th.float32) if mid < len(t.acts) else th.zeros_like(st)
            # store
            all_states.append(st)
            all_actions.append(ac)
            all_labels.append(avg_p)

        states_cat = th.stack(all_states).to(device)
        acts_cat = th.stack(all_actions).to(device)
        labels_cat = th.tensor(all_labels, dtype=th.float32, device=device)

        # predict
        progress_hat = self._reward_net.predict_progress(states_cat, acts_cat)
        loss = F.mse_loss(progress_hat, labels_cat)
        return loss

    # ---------------------------------------------------
    #   4) Progress-Guided Reward Regularization (Request #4)
    # ---------------------------------------------------
    def progress_regularization_loss(self) -> th.Tensor:
        """
        For each sampled (state, action) with known progress,
        we add (r(s,a) - alpha*progress)^2 to the loss.
        """
        device = self.gen_algo.device
        idxs = np.random.choice(len(self.annotation_list), size=self.shaping_batch_size, replace=False)
        all_states, all_actions, all_progress = [], [], []
        for idx in idxs:
            pdict, demo_idx = self.annotation_list[idx]
            avg_p = 0.5*(pdict["start_progress"] + pdict["end_progress"])
            t = self.demonstrations_for_shaping[demo_idx]
            mid = (pdict["start_step"] + pdict["end_step"]) // 2
            if mid >= len(t.obs):
                mid = len(t.obs) - 1
            st = th.tensor(t.obs[mid], dtype=th.float32)
            ac = th.tensor(t.acts[mid], dtype=th.float32) if mid < len(t.acts) else th.zeros_like(st)
            all_states.append(st)
            all_actions.append(ac)
            all_progress.append(avg_p)

        states_cat = th.stack(all_states).to(device)
        acts_cat = th.stack(all_actions).to(device)
        progress_cat = th.tensor(all_progress, dtype=th.float32, device=device)

        # Evaluate net's unshaped reward
        with th.no_grad():
            # For single-step, we can just pass next_state=states_cat for convenience
            r_vals = self._reward_net.base(states_cat, acts_cat, states_cat, th.zeros_like(progress_cat))

        loss = F.mse_loss(r_vals, self.alpha * progress_cat)
        return loss

    # ---------------------------------------------------
    # Overriding train_disc to incorporate new shaping
    # ---------------------------------------------------
    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        """
        A single discriminator update + optional shaping step.
        """
        with self.logger.accumulate_means("disc"):
            write_summaries = self._init_tensorboard and (self._global_step % 20 == 0)

            # Standard AIRL classification step
            self._disc_opt.zero_grad()
            batch_iter = self._make_disc_train_batches(gen_samples=gen_samples, expert_samples=expert_samples)

            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                class_loss = F.binary_cross_entropy_with_logits(
                    disc_logits, batch["labels_expert_is_one"].float()
                )
                # Re-normalize
                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                class_loss *= self.demo_minibatch_size / self.demo_batch_size
                class_loss.backward()

            self._disc_opt.step()
            self._disc_step += 1

            # If we have shaping losses
            shaping_loss_total = th.tensor(0.0, device=self.gen_algo.device)
            if len(self.shape_reward) > 0 and (self._disc_step % self.shaping_update_freq == 0):
                self._disc_opt.zero_grad()

                if "demo_range_loss" in self.shape_reward:
                    shaping_loss_total += self.demo_range_loss()

                if "progress_regression_loss" in self.shape_reward:
                    shaping_loss_total += self.subtraj_progress_regression_loss()

                if "progress_head_loss" in self.shape_reward:
                    # weight the MSE from the progress head
                    shaping_loss_total += self.progress_head_weight * self.progress_head_loss()

                if "progress_regularization" in self.shape_reward:
                    shaping_loss_total += self.lambda_reg * self.progress_regularization_loss()

                shaping_loss_total *= self.shaping_loss_weight
                shaping_loss_total.backward()
                self._disc_opt.step()

            # record stats
            with th.no_grad():
                train_stats = {
                    "classification_loss": class_loss.item(),
                    "shaping_loss": shaping_loss_total.item(),
                }
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)

            if write_summaries and hasattr(self, "_summary_writer"):
                self._summary_writer.add_scalar("disc/class_loss", class_loss.item(), self._global_step)
                self._summary_writer.add_scalar("disc/shaping_loss", shaping_loss_total.item(), self._global_step)

        # Periodic saving
        if self._global_step % self.save_model_every == 0:
            step_path = os.path.join(self.save_path, f"{self._global_step}")
            os.makedirs(step_path, exist_ok=True)
            train_adversarial.save(self, Path(step_path))

        return train_stats

    # Overriding to ensure we return our custom net
    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        # Unwrap wrappers if any
        net = self._reward_net
        while isinstance(net, reward_nets.RewardNetWrapper):
            net = net.base
        return net
