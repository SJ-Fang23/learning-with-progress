# A modified version of airl.py from imitation library
# with no gradient on shaping functions, and a final factor in [0,1]
# that scales the AIRL BCE loss.

"""Adversarial Inverse Reinforcement Learning (AIRL) with shaping weight factor."""

from typing import Callable, Iterable, Iterator, Mapping, Optional, Type, overload

import torch as th
from torch.nn import functional as F

from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial.common import compute_train_stats
from imitation.rewards import reward_nets

from imitation.data import types
from copy import deepcopy
import numpy as np
import imitation.scripts.train_adversarial as train_adversarial
import os
from pathlib import Path


STOCHASTIC_POLICIES = (sac_policies.SACPolicy, policies.ActorCriticPolicy)


class AIRL(common.AdversarialTrainer):
    """Adversarial Inverse Reinforcement Learning (AIRL).

    This implementation has been *modified* so that all “shaping” signals
    (e.g. progress sign constraints) no longer produce a gradient. Instead,
    they produce an 'accuracy factor' in [0, 1], which is then used to scale
    the main AIRL BCE loss. This scaling is controlled by the hyperparameter
    `shaping_loss_weight` (also in [0,1]) and a combined factor from the
    shaping computations.
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        annotation_list: list[tuple[int, dict]],
        demostrations_for_shaping: list[types.Trajectory],
        shape_reward=[],
        shaping_batch_size: int = 16,
        shaping_loss_weight: float = 1.0,
        shaping_update_freq: int = 1,
        shaping_lr: float = 1e-3,
        save_model_every=20,
        save_path="checkpoints/default",
        traj_index=[],
        **kwargs,
    ):
        """Builds an AIRL trainer with reward-shaping-based constraints.

        Args:
            demonstrations: Expert transitions for standard AIRL.
            demo_batch_size: Number of expert samples per discriminator step.
            venv: Vectorized environments.
            gen_algo: The generator's RL algorithm (e.g. PPO).
            reward_net: The reward network to be trained.
            annotation_list: List of tuples, each containing `(dict_of_info, demonstration_index)`.
            demostrations_for_shaping: A list of full `Trajectory`s for shaping computations.
            shape_reward: Which shaping signals to enable. E.g. `["progress_sign_loss"]` or similar.
            shaping_batch_size: Batch size for shaping constraints.
            shaping_loss_weight: Global weighting factor in [0,1] for scaling BCE loss
                based on shaping constraints.
            shaping_update_freq: Frequency (in discriminator steps) to compute the shaping factor.
            shaping_lr: Not used in the no-gradient approach, but kept for compatibility.
            save_model_every: Save checkpoint frequency (global steps).
            save_path: Directory to save.
            traj_index: Indices used to help pick which demonstrations are considered in shaping.

        Raises:
            TypeError: If `gen_algo.policy` does not have an `evaluate_actions`
                attribute. (AIRL requires a stochastic policy.)
        """
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )

        # AIRL needs a policy from STOCHASTIC_POLICIES:
        if not isinstance(self.gen_algo.policy, STOCHASTIC_POLICIES):
            raise TypeError(
                "AIRL needs a stochastic policy to compute the discriminator output."
            )

        assert isinstance(demostrations_for_shaping, list), \
            "demonstrations_for_shaping must be a list of Trajectory"
        assert isinstance(demostrations_for_shaping[0], types.Trajectory), \
            "demonstrations_for_shaping must be a list of Trajectory"
        assert isinstance(annotation_list, list), "annotation_dict must be a list"

        self.demonstrations = deepcopy(demonstrations)
        self.demonstrations_for_shaping = deepcopy(demostrations_for_shaping)
        self.annotation_list = annotation_list
        self.shaping_batch_size = shaping_batch_size
        self.shaping_loss_weight = shaping_loss_weight
        self.shaping_update_freq = shaping_update_freq
        self.shaping_lr = shaping_lr
        self.shape_reward = shape_reward
        self.save_model_every = save_model_every
        self.save_path = save_path
        self.project_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.save_path = os.path.join(self.project_path, self.save_path)
        self.traj_index = traj_index

        if not os.path.exists(self.save_path):
            print("creating save path")
            os.makedirs(self.save_path)

    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        r"""Compute the AIRL discriminator logits for each state-action sample.

        In Fu's AIRL paper (https://arxiv.org/abs/1710.11248), the
        discriminator output is:

        .. math::

            D_{\theta}(s,a)
            = \frac{\exp(r_{\theta}(s,a))}{\exp(r_{\theta}(s,a)) + \pi(a|s)}

        Taking the logit yields:

        .. math::

            \logit(D_{\theta}(s,a)) = r_{\theta}(s,a) - \log \pi(a|s).

        Args:
            state: The state of the environment at the time of the action.
            action: The action taken by the expert or generator.
            next_state: The state of the environment after the action.
            done: Whether a terminal state (as defined by the MDP) has been reached.
            log_policy_act_prob: The log probability of the action under the
                generator's policy, i.e. log π(a|s).

        Returns:
            The logits of the discriminator for each sample.
        """
        if log_policy_act_prob is None:
            raise TypeError(
                "Non-None `log_policy_act_prob` is required for this method."
            )
        reward_output_train = self._reward_net(state, action, next_state, done)
        return reward_output_train - log_policy_act_prob

    # =========================================================================
    #         ALL “SHAPING” FUNCTIONS NOW RETURN *ACCURACY FACTORS* IN [0,1]
    #         AND USE NO GRADIENT.
    #
    #         The old approach returned a “loss” that contributed to grad.
    #         We now rename them and produce an ‘accuracy’ or ‘agreement’ measure,
    #         from which we can derive a factor in [0..1].
    # =========================================================================

    def subtrajectory_proportion_factor(self) -> th.Tensor:
        """
        Example transformation of your old 'subtrajectory_proportion_loss' into
        a [0,1] factor indicating how well the partial-reward proportions match
        partial progress proportions. 1 => perfect match, 0 => large mismatch.

        We do everything under `torch.no_grad()` so it does *not* produce gradients.
        """
        device = self.gen_algo.device
        # If no shaping demonstrations exist, return factor=1 (no penalty).
        if len(self.demonstrations_for_shaping) == 0:
            return th.tensor(1.0, device=device)

        # We'll pick a couple random trajectories from self.traj_index:
        idxs = np.random.choice(self.traj_index, 2, replace=False)
        differences = []
        with th.no_grad():
            for i in idxs:
                # Gather all segments for demonstration i
                segments = [
                    (ann[0]["start_step"], ann[0]["end_step"], ann[0]["end_progress"])
                    for ann in self.annotation_list
                    if ann[1] == i
                ]
                if len(segments) < 2:
                    continue
                segments.sort(key=lambda x: x[1])

                # Load entire trajectory
                traj = self.demonstrations_for_shaping[i]
                T = len(traj.obs)
                states_all = th.tensor(traj.obs, dtype=th.float32, device=device)
                actions_all = th.tensor(traj.acts, dtype=th.float32, device=device)
                next_all = th.tensor(traj.obs, dtype=th.float32, device=device)
                dones_all = th.zeros(T, dtype=th.float32, device=device)

                # Match length
                min_length = min(len(states_all), len(next_all), len(actions_all))
                states_all = states_all[:min_length]
                actions_all = actions_all[:min_length]
                next_all = next_all[:min_length]
                dones_all = dones_all[:min_length]

                # Sum of shaped reward
                all_rews = self._reward_net.base(states_all, actions_all, next_all, dones_all)
                total_reward = all_rews.sum().item()

                final_progress = segments[-1][2]  # last segment's end_progress
                if abs(final_progress) < 1e-6:
                    # skip if final progress is 0
                    continue

                partial_diffs = []
                for seg_idx, (st, en, seg_prog) in enumerate(segments):
                    # partial reward up to that segment
                    sub_r = all_rews[:en].sum().item()
                    prop_prog = seg_prog / final_progress
                    prop_rew = sub_r / (total_reward + 1e-8)

                    diff = abs(prop_prog - prop_rew)  # e.g. L1 difference
                    partial_diffs.append(diff)

                if len(partial_diffs) > 0:
                    differences.append(np.mean(partial_diffs))

        if len(differences) == 0:
            return th.tensor(1.0, device=device)

        # average difference across chosen demos
        avg_diff = float(np.mean(differences))
        # Convert difference => factor in [0,1]. For example:
        # factor = max(0, 1 - avg_diff). If avg_diff >= 1, factor=0; if avg_diff=0, factor=1.
        factor = 1.0 - avg_diff
        if factor < 0:
            factor = 0.0
        return th.tensor(factor, dtype=th.float32, device=device)

    def end_progress_factor(self) -> th.Tensor:
        """
        Example transformation of your old 'end_progress_loss' into a factor in [0,1].
        The closer we are to the user-defined constraints, the closer this factor is to 1.
        """
        device = self.gen_algo.device
        if len(self.demonstrations_for_shaping) == 0:
            return th.tensor(1.0, device=device)

        # This is just a sample approach:
        idxs = np.random.choice(self.traj_index, 2, replace=False)
        differences = []
        with th.no_grad():
            for i in idxs:
                segments = [
                    (ann[0]["start_step"], ann[0]["end_step"], ann[0]["end_progress"])
                    for ann in self.annotation_list
                    if ann[1] == i
                ]
                if len(segments) < 2:
                    continue
                segments.sort(key=lambda x: x[1])
                end_prog = segments[-1][2]

                # Load entire trajectory
                traj = self.demonstrations_for_shaping[i]
                T = len(traj.obs)
                states_all = th.tensor(traj.obs, dtype=th.float32, device=device)
                actions_all = th.tensor(traj.acts, dtype=th.float32, device=device)
                next_all = th.tensor(traj.obs, dtype=th.float32, device=device)
                dones_all = th.zeros(T, dtype=th.float32, device=device)

                min_length = min(len(states_all), len(next_all), len(actions_all))
                states_all = states_all[:min_length]
                actions_all = actions_all[:min_length]
                next_all = next_all[:min_length]
                dones_all = dones_all[:min_length]

                all_rews = self._reward_net.base(states_all, actions_all, next_all, dones_all)
                total_reward = all_rews.sum().item()
                # We'll pretend we have some known "target" total reward that matches end_prog
                # This is arbitrary. We measure difference, accumulate in `differences`.
                # For demonstration:
                wanted_reward = end_prog  # or some function
                differences.append(abs(total_reward - wanted_reward))

        if len(differences) == 0:
            return th.tensor(1.0, device=device)

        avg_diff = float(np.mean(differences))
        factor = 1.0 - avg_diff
        if factor < 0:
            factor = 0.0
        return th.tensor(factor, dtype=th.float32, device=device)

    def delta_progress_scale_factor(self, delta_progress: th.Tensor, reward_output_train: th.Tensor) -> th.Tensor:
        """
        Old code used a 'loss' to push subtrajectories with bigger progress to get bigger
        reward sums. Now we interpret it as an 'agreement' measure in [0,1].

        We'll do a rough approach: measure the old "loss" and convert it to factor=1/(1+loss).
        """
        with th.no_grad():
            # old 'loss' style:
            delta_progress_diff = delta_progress.unsqueeze(1) - delta_progress.unsqueeze(0)
            reward_output_train_diff = reward_output_train.unsqueeze(1) - reward_output_train.unsqueeze(0)

            old_loss = th.mean(th.relu(-F.softsign(delta_progress_diff * reward_output_train_diff)))
            factor = 1.0 / (1.0 + old_loss.item())  # in (0,1]
        return th.tensor(factor, dtype=th.float32, device=self.gen_algo.device)

    def value_sign_factor(self, delta_progress: th.Tensor, delta_value: th.Tensor) -> th.Tensor:
        """
        Compare sign of delta_progress with sign of (-delta_value).
        Instead of returning BCE loss, we compute fraction-of-correct-sign = an accuracy in [0,1].
        """
        with th.no_grad():
            # sign of delta_progress
            progress_bool = (delta_progress >= 0)
            # sign of -delta_value => True if delta_value <= 0
            value_bool = (delta_value <= 0)

            correct = (progress_bool == value_bool).float()
            accuracy = correct.mean().item()
        return th.tensor(accuracy, dtype=th.float32, device=self.gen_algo.device)

    def reward_sign_factor(self, delta_progress: th.Tensor, reward_output_train: th.Tensor) -> th.Tensor:
        """
        Compare sign of delta_progress with sign of reward_output_train => fraction correct.
        """
        with th.no_grad():
            progress_bool = (delta_progress >= 0)
            reward_bool = (reward_output_train >= 0)
            correct = (progress_bool == reward_bool).float()
            accuracy = correct.mean().item()
        return th.tensor(accuracy, dtype=th.float32, device=self.gen_algo.device)

    def advantage_sign_factor(self, delta_progress: th.Tensor, delta_advantage: th.Tensor) -> th.Tensor:
        """
        Compare sign of delta_progress vs sign of delta_advantage => fraction correct.
        """
        with th.no_grad():
            progress_bool = (delta_progress >= 0)
            advantage_bool = (delta_advantage >= 0)
            correct = (progress_bool == advantage_bool).float()
            accuracy = correct.mean().item()
        return th.tensor(accuracy, dtype=th.float32, device=self.gen_algo.device)

    # =========================================================================
    #  progress_shaping_factor(...) Gathers and aggregates the sub-factors
    #  depending on which shaping signals you actually want to use (self.shape_reward).
    #  The final result is in [0,1].
    # =========================================================================

    def progress_shaping_factor(self) -> th.Tensor:
        """
        Gather the various “shaping” factors (which are 0..1 accuracies), average them,
        and return one final factor in [0,1]. If you only want to incorporate some subset
        of signals, check `self.shape_reward`.
        """
        device = self.gen_algo.device
        # If shape_reward is empty, or no shaping data, return factor=1 => no scaling.
        if len(self.shape_reward) == 0 or len(self.annotation_list) == 0:
            return th.tensor(1.0, device=device)

        with th.no_grad():
            # We'll sample subtrajectories exactly as in your original code,
            # then compute sign comparisons, etc.

            indices = np.random.choice(
                len(self.annotation_list), self.shaping_batch_size, replace=False
            )
            threshold = 0.1
            annotations = [self.annotation_list[idx] for idx in indices]

            delta_progress = th.tensor(
                [
                    ann[0]["end_progress"] - ann[0]["start_progress"] - threshold
                    for ann in annotations
                ],
                dtype=th.float32,
                device=device,
            )
            avg_prog_value = th.tensor(
                [
                    0.5 * (ann[0]["start_progress"] + ann[0]["end_progress"])
                    for ann in annotations
                ],
                dtype=th.float32,
                device=device,
            )

            # Collect states, actions, next_states, etc.
            # demonstration_indices = (demo_idx, start_step, end_step)
            demonstration_indices = [
                (ann[1], ann[0]["start_step"], ann[0]["end_step"]) for ann in annotations
            ]

            states_list = []
            actions_list = []
            next_states_list = []
            dones_list = []

            for (demo_idx, start_step, end_step) in demonstration_indices:
                traj = self.demonstrations_for_shaping[demo_idx]
                obs_chunk = th.tensor(
                    traj.obs[start_step:end_step], dtype=th.float32
                )
                acts_chunk = th.tensor(
                    traj.acts[start_step:end_step], dtype=th.float32
                )
                next_chunk = th.tensor(
                    traj.obs[start_step + 1 : end_step + 1], dtype=th.float32
                )
                d = 1 if traj.terminal else 0
                # For sub-trajectory, treat them as not done except possibly at the end
                # but we only need to keep shape consistent.
                # We'll store a "done" for each step in chunk:
                chunk_dones = th.zeros(len(obs_chunk), dtype=th.float32)
                # If the demonstration is truly terminal, we might mark final step done
                # for purely illustrative reasons
                if d == 1 and len(chunk_dones) > 0:
                    chunk_dones[-1] = 1.0

                states_list.append(obs_chunk)
                actions_list.append(acts_chunk)
                next_states_list.append(next_chunk)
                dones_list.append(chunk_dones)

            # pad lengths if needed or handle them by simple concatenation
            states = th.cat(states_list, dim=0).to(device)
            actions = th.cat(actions_list, dim=0).to(device)
            next_states = th.cat(next_states_list, dim=0).to(device)
            dones = th.cat(dones_list, dim=0).to(device)

            # Make sure they match
            min_len = min(len(states), len(actions), len(next_states), len(dones))
            states = states[:min_len]
            actions = actions[:min_len]
            next_states = next_states[:min_len]
            dones = dones[:min_len]

            # Potential-based shaping terms
            v_s = self._reward_net.potential(states)
            v_s_next = self._reward_net.potential(next_states)
            # advantage_output is not necessarily in your original net, but let's keep it
            advantage_output = self._reward_net(states, actions, next_states, th.tensor(1.0, device=device))

            old_value_output = v_s.flatten()
            next_value_output = v_s_next.flatten()

            # We want *sums* per subtrajectory for each of these signals
            # so we chunk them back out
            lengths = [len(x) for x in states_list]
            lengths_t = th.tensor([0] + lengths, device=device)
            offsets = th.cumsum(lengths_t, dim=0)

            # Summation over each subtrajectory
            def sum_over_subtraj(tensor_1d):
                sums = []
                for i_sub in range(len(lengths)):
                    start_idx = offsets[i_sub]
                    end_idx = offsets[i_sub + 1]
                    sums.append(tensor_1d[start_idx:end_idx].sum())
                return th.stack(sums)

            # Summation
            delta_value = (v_s_next - v_s)
            sum_reward_output = sum_over_subtraj(
                self._reward_net.base(states, actions, next_states, dones)
            )
            sum_delta_value = sum_over_subtraj(delta_value.flatten())
            sum_advantage = sum_over_subtraj(advantage_output.flatten())
            sum_old_value = sum_over_subtraj(old_value_output)
            sum_next_value = sum_over_subtraj(next_value_output)

            # shape: number_of_subtrajectories = shaping_batch_size
            # must match delta_progress length
            subtraj_count = len(lengths)
            if subtraj_count != len(delta_progress):
                # If there's any mismatch from random sampling,
                # we'll just clamp them to the smaller size
                final_count = min(subtraj_count, len(delta_progress))
                sum_reward_output = sum_reward_output[:final_count]
                sum_delta_value = sum_delta_value[:final_count]
                sum_advantage = sum_advantage[:final_count]
                sum_old_value = sum_old_value[:final_count]
                sum_next_value = sum_next_value[:final_count]
                delta_progress = delta_progress[:final_count]
                avg_prog_value = avg_prog_value[:final_count]

            # Now compute each shaping factor we might use
            factor_list = []

            # Because user might want "progress_sign_loss", "value_sign_loss", etc.
            # We simply check which strings are in self.shape_reward. For each, compute factor.

            if "progress_sign_loss" in self.shape_reward:
                # Typically, "progress_sign_loss" might mean we want both advantage_sign and reward_sign
                # or we might only do advantage_sign. Adapt as needed:
                adv_factor = self.advantage_sign_factor(delta_progress, sum_advantage)
                rew_factor = self.reward_sign_factor(delta_progress, sum_next_value)  # or sum_reward_output
                # Combine them (e.g. average):
                factor_list.append(adv_factor)
                factor_list.append(rew_factor)

            if "value_sign_loss" in self.shape_reward:
                val_factor = self.value_sign_factor(delta_progress, sum_delta_value)
                factor_list.append(val_factor)

            if "delta_progress_scale_loss" in self.shape_reward:
                dps_factor = self.delta_progress_scale_factor(delta_progress, sum_reward_output)
                factor_list.append(dps_factor)

            if "subtrajectory_proportion_loss" in self.shape_reward:
                sub_prop_factor = self.subtrajectory_proportion_factor()
                factor_list.append(sub_prop_factor)

            if "end_progress_loss" in self.shape_reward:
                ep_factor = self.end_progress_factor()
                factor_list.append(ep_factor)

            # If we didn't compute any factors, default to 1
            if len(factor_list) == 0:
                return th.tensor(1.0, device=device)

            # final factor is average
            final_factor = th.mean(th.stack(factor_list))
            return final_factor

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        """Returns the unshaped version of reward network used for testing."""
        reward_net = self._reward_net
        while isinstance(reward_net, reward_nets.RewardNetWrapper):
            reward_net = reward_net.base
        return reward_net

    # =========================================================================
    # Main discriminator training step, now with factor-based shaping
    # =========================================================================

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Expert transition samples in dictionary form.
            gen_samples: Generator (policy) transition samples in dictionary form.

        Returns:
            Statistics for logging.
        """
        with self.logger.accumulate_means("disc"):
            # We'll do exactly one step of the disc_opt here
            self._disc_opt.zero_grad()

            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )

            # We track stats for logging
            all_disc_logits = []
            all_labels = []
            all_losses = []

            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                bce_loss = F.binary_cross_entropy_with_logits(
                    disc_logits, batch["labels_expert_is_one"].float()
                )
                # Renormalize the BCE to reflect total batch size
                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                bce_loss *= self.demo_minibatch_size / self.demo_batch_size

                # CHANGED:
                # Instead of adding a gradient-based shaping_loss,
                # we compute an overall shaping factor in [0,1] and multiply
                # the BCE loss by shaping_loss_weight * that factor.
                if (
                    len(self.shape_reward) > 0
                    and (self._disc_step % self.shaping_update_freq == 0)
                ):
                    # We do *not* want grad from shaping factor => use no_grad inside:
                    shaping_factor = self.progress_shaping_factor()
                    # shaping_factor in [0..1]. We scale final BCE by shaping_loss_weight * factor
                    final_scaling = self.shaping_loss_weight * shaping_factor
                    combined_loss = bce_loss * final_scaling
                else:
                    # If no shaping or not time to do shaping update, just do BCE
                    combined_loss = bce_loss

                combined_loss.backward()

                all_disc_logits.append(disc_logits.detach())
                all_labels.append(batch["labels_expert_is_one"].float().detach())
                all_losses.append(combined_loss.detach())

            self._disc_opt.step()
            self._disc_step += 1

            # # Logging
            # with th.no_grad():
            #     disc_logits_cat = th.cat(all_disc_logits, dim=0)
            #     labels_cat = th.cat(all_labels, dim=0)
            #     final_loss_cat = th.cat(all_losses, dim=0)
            #     train_stats = compute_train_stats(
            #         disc_logits_cat,
            #         labels_cat,
            #         final_loss_cat.mean(),
            #     )
            # self.logger.record("global_step", self._global_step)
            # for k, v in train_stats.items():
            #     self.logger.record(k, v)
            # self.logger.dump(self._disc_step)

        # Possibly save model
        if self._global_step % self.save_model_every == 0:
            save_path_this_time = os.path.join(self.save_path, f"{self._global_step}")
            if not os.path.exists(save_path_this_time):
                os.makedirs(save_path_this_time)
            save_path_this_time = Path(save_path_this_time)
            train_adversarial.save(self, save_path_this_time)

        return train_stats
