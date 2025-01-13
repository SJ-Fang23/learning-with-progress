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
      - a "progress_hat" for each state-action (used for e.g. direct progress regression).
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
        """
        x = self._concat_inputs(states, actions)
        feats = self.feature_extractor(x)
        progress_hat = self.progress_head(feats).squeeze(-1)
        return progress_hat


# -------------------------------------------------------------------
# 2) Extended AIRL class that includes new shaping losses, plus alternative versions
# -------------------------------------------------------------------
class AIRLWithProgress(common.AdversarialTrainer):
    """
    An AIRL trainer that:
      - integrates multiple shaping losses in train_disc
      - uses a custom reward net (like RewardNetWithProgress)
      - let's you pick which shaping losses to enable via shape_reward list
      - if certain sign-based losses may not align with your exact progress goal,
        we also provide alternative versions with a `_alternative` suffix.
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: RewardNetWithProgress,
        annotation_list: List[Tuple[dict, int]],  # (progress_dict, demonstration_idx)
        demonstrations_for_shaping: List[types.Trajectory],
        shape_reward=None,  # e.g. ["value_sign_loss", "progress_sign_loss", "delta_progress_scale_loss", ...]
        shaping_batch_size: int = 16,
        shaping_loss_weight: float = 1.0,
        shaping_update_freq: int = 1,
        shaping_lr: float = 1e-3,
        save_model_every=20,
        save_path="checkpoints/default",
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
            raise TypeError("AIRL requires a stochastic policy to compute r - log pi(a|s).")

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

    # ---------------------------------------------------
    # Additional sign/scale-based progress shaping losses (Original)
    # ---------------------------------------------------
    def value_sign_loss(self, delta_progress: th.Tensor, delta_value: th.Tensor) -> th.Tensor:
        """
        Encourages the sign of delta_progress to match the sign of delta_value.
        Potentially consistent with progress-based shaping, 
        but depends on what delta_value represents in your pipeline.
        """
        value_agreement = th.sign(delta_progress).to(self.gen_algo.device) * th.sign(delta_value)
        loss = th.mean(th.relu(-value_agreement))
        return loss

    def advantage_sign_loss(self, delta_progress: th.Tensor, delta_advantage: th.Tensor) -> th.Tensor:
        """
        Encourages the sign of delta_progress to match the sign of advantage.
        Potentially consistent, but depends on how advantage is computed in your system.
        """
        advantage_agreement = th.sign(delta_progress).to(self.gen_algo.device) * th.sign(delta_advantage)
        loss = th.mean(th.relu(-advantage_agreement))
        return loss

    def progress_sign_loss(self, delta_progress: th.Tensor, reward_output_train: th.Tensor) -> th.Tensor:
        """
        Encourages the sign of the subtrajectory's delta progress to match the sign of the reward sum.
        Typically consistent with progress shaping.
        """
        sign_agreement = F.softsign(delta_progress).to(self.gen_algo.device) * F.softsign(reward_output_train)
        loss = th.mean(th.relu(-sign_agreement))
        return loss

    def delta_progress_scale_loss(self, delta_progress: th.Tensor, reward_output_train: th.Tensor) -> th.Tensor:
        """
        Encourages subtrajectories with higher delta progress to have higher reward sums, 
        enforcing a rank-order relationship.
        """
        dp_diff = delta_progress.unsqueeze(1) - delta_progress.unsqueeze(0)
        r_diff = reward_output_train.unsqueeze(1) - reward_output_train.unsqueeze(0)
        loss = th.mean(th.relu(-F.softsign(dp_diff * r_diff)))
        return loss

    def progress_value_loss(self, average_progress: th.Tensor, next_value_output_train: th.Tensor) -> th.Tensor:
        """
        Encourages states with higher average progress to have higher predicted next-state value.
        Possibly consistent with a typical progress-based approach.
        """
        p_diff = average_progress.unsqueeze(1) - average_progress.unsqueeze(0)
        v_diff = next_value_output_train.unsqueeze(1) - next_value_output_train.unsqueeze(0)
        loss = th.mean(th.relu(-th.sign(p_diff * v_diff)))
        return loss

    # ---------------------------------------------------
    # Alternative versions (if original sign-based constraints aren't suitable)
    # ---------------------------------------------------
    def value_sign_loss_alternative(self, delta_progress: th.Tensor, delta_value: th.Tensor) -> th.Tensor:
        """
        Hypothetical alternative for value_sign_loss that is less strict about sign agreement.
        For instance, we might allow small negative differences or scale weighting.
        """
        # e.g. we do MSE: (value_diff - alpha* delta_progress)^2
        scale = 1.0
        # interpret delta_value as "difference in potential" 
        # interpret delta_progress as direct measure. 
        # Then we want them to align in magnitude, not just sign.
        # This is just an example of how you might do it differently.
        loss = F.mse_loss(delta_value, scale * delta_progress)
        return loss

    def progress_sign_loss_alternative(self, delta_progress: th.Tensor, reward_output_train: th.Tensor) -> th.Tensor:
        """
        Another example of a more flexible approach, maybe using margin-based ranking
        instead of direct sign matching.
        """
        # For illustration: if delta_progress[i] > delta_progress[j] + margin => reward_output[i] > reward_output[j].
        margin = 0.05
        dp_diff = delta_progress.unsqueeze(1) - delta_progress.unsqueeze(0)
        r_diff = reward_output_train.unsqueeze(1) - reward_output_train.unsqueeze(0)
        # violation = relu(margin - (r_diff * sign(dp_diff)))
        violation = th.relu(margin - (r_diff * th.sign(dp_diff)))
        loss = th.mean(violation)
        return loss

    # ---------------------------------------------------
    # logits_expert_is_high (required by AdversarialTrainer)
    # ---------------------------------------------------
    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Computes the discriminator's logits for each state-action sample.
        reward - log_policy_act_prob.
        """
        if log_policy_act_prob is None:
            raise TypeError("Non-None `log_policy_act_prob` is required for this method.")
        reward_output_train = self._reward_net(state, action, next_state, done)
        return reward_output_train - log_policy_act_prob

    # ---------------------------------------------------
    # train_disc: integrate shaping
    # ---------------------------------------------------
    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        with self.logger.accumulate_means("disc"):
            # Standard AIRL classification step
            self._disc_opt.zero_grad()
            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )

            class_loss_value = th.tensor(0.0, device=self.gen_algo.device)
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
                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                class_loss *= self.demo_minibatch_size / self.demo_batch_size
                class_loss.backward()
                class_loss_value += class_loss.detach()

            self._disc_opt.step()
            self._disc_step += 1

            # Additional shaping step
            shaping_loss_total = th.tensor(0.0, device=self.gen_algo.device)
            if len(self.shape_reward) > 0 and (self._disc_step % self.shaping_update_freq == 0):
                self._disc_opt.zero_grad()
                shaping_data = self._sample_shaping_trajectories()  # gather relevant subtraj info

                # For each shaping key in shape_reward, compute the associated loss
                for shaping_key in self.shape_reward:
                    if shaping_key == "value_sign_loss":
                        shaping_loss_total += self.value_sign_loss(
                            shaping_data["delta_progress"], shaping_data["delta_value"]
                        )
                    elif shaping_key == "advantage_sign_loss":
                        shaping_loss_total += self.advantage_sign_loss(
                            shaping_data["delta_progress"], shaping_data["delta_advantage"]
                        )
                    elif shaping_key == "progress_sign_loss":
                        shaping_loss_total += self.progress_sign_loss(
                            shaping_data["delta_progress"], shaping_data["reward_output_train"]
                        )
                    elif shaping_key == "delta_progress_scale_loss":
                        shaping_loss_total += self.delta_progress_scale_loss(
                            shaping_data["delta_progress"], shaping_data["reward_output_train"]
                        )
                    elif shaping_key == "progress_value_loss":
                        shaping_loss_total += self.progress_value_loss(
                            shaping_data["average_progress"], shaping_data["next_value_output_train"]
                        )
                    # alternative versions:
                    elif shaping_key == "value_sign_loss_alternative":
                        shaping_loss_total += self.value_sign_loss_alternative(
                            shaping_data["delta_progress"], shaping_data["delta_value"]
                        )
                    elif shaping_key == "progress_sign_loss_alternative":
                        shaping_loss_total += self.progress_sign_loss_alternative(
                            shaping_data["delta_progress"], shaping_data["reward_output_train"]
                        )

                shaping_loss_total *= self.shaping_loss_weight
                shaping_loss_total.backward()
                self._disc_opt.step()

            # Summaries
            train_stats = {
                "classification_loss": class_loss_value.item(),
                "shaping_loss": shaping_loss_total.item(),
            }
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)

        # Periodic checkpoint
        if self._global_step % self.save_model_every == 0:
            step_path = os.path.join(self.save_path, f"{self._global_step}")
            os.makedirs(step_path, exist_ok=True)
            train_adversarial.save(self, Path(step_path))

        return train_stats

    # ---------------------------------------------------
    # _sample_shaping_trajectories
    # ---------------------------------------------------
    def _sample_shaping_trajectories(self) -> dict:
        """
        Helper function to sample subtrajectories and compute required shaping values
        like delta_progress, reward sums, etc.
        """
        device = self.gen_algo.device
        idxs = np.random.choice(len(self.annotation_list), size=self.shaping_batch_size, replace=False)

        delta_progress_list = []
        reward_output_list = []
        delta_value_list = []
        delta_adv_list = []
        avg_progress_list = []
        next_value_output_list = []

        for idx in idxs:
            pdict, demo_idx = self.annotation_list[idx]
            start_p = pdict["start_progress"]
            end_p = pdict["end_progress"]
            dp = (end_p - start_p)
            avg_p = 0.5 * (start_p + end_p)

            traj = self.demonstrations_for_shaping[demo_idx]
            s, e = pdict["start_step"], pdict["end_step"]

            states = th.tensor(traj.obs[s:e], dtype=th.float32, device=device)
            actions = th.tensor(traj.acts[s:e], dtype=th.float32, device=device)
            if len(states) == 0:
                # edge case: skip empty subtrajectory
                continue
            next_states = th.tensor(traj.obs[s+1:e+1], dtype=th.float32, device=device)
            if len(next_states) < len(states):
                next_states = th.cat([next_states, next_states[-1:].clone()], dim=0)
            dones = th.zeros(len(states), dtype=th.float32, device=device)

            with th.no_grad():
                # sum the unshaped reward for the subtrajectory
                r_vals = self._reward_net.base(states, actions, next_states, dones)
                # potential values
                v_s = self._reward_net.potential(states)
                v_s_next = self._reward_net.potential(next_states)
            # Delta value for the subtrajectory
            dv = (v_s_next - v_s).sum().item()
            # advantage ~ (r - v_s)? or something custom
            adv = (r_vals - v_s).sum().item()

            delta_progress_list.append(dp)
            reward_output_list.append(r_vals.sum().item())
            delta_value_list.append(dv)
            delta_adv_list.append(adv)
            avg_progress_list.append(avg_p)
            next_value_output_list.append(v_s_next.sum().item())

        return {
            "delta_progress": th.tensor(delta_progress_list, device=device),
            "reward_output_train": th.tensor(reward_output_list, device=device),
            "delta_value": th.tensor(delta_value_list, device=device),
            "delta_advantage": th.tensor(delta_adv_list, device=device),
            "average_progress": th.tensor(avg_progress_list, device=device),
            "next_value_output_train": th.tensor(next_value_output_list, device=device),
        }

    # ---------------------------------------------------
    # Properties for reward train/test
    # ---------------------------------------------------
    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        net = self._reward_net
        while isinstance(net, reward_nets.RewardNetWrapper):
            net = net.base
        return net
