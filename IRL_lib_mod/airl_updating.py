# A modified version of airl.py from imitation library
# add reward shaping
"""Adversarial Inverse Reinforcement Learning (AIRL)."""
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
    """Adversarial Inverse Reinforcement Learning (`AIRL`_).

    .. _AIRL: https://arxiv.org/abs/1710.11248
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
        shape_reward = [],
        shaping_batch_size: int = 16,
        shaping_loss_weight: float = 1.0,
        shaping_update_freq: int = 1,
        shaping_lr: float = 1e-3,
        save_model_every = 10,
        save_path = "checkpoints/default",
        traj_index = [], 
        **kwargs,
    ):
        """Builds an AIRL trainer.

        Args:
            annotation_list: list of annotation tuples: (dict(progress_data), int(corresponding demonstration index))
            shape_reward: a list of reward shapings to use, no reward shaping if empty, can contain: ["progress_sign_loss"]
        Raises:
            TypeError: If `gen_algo.policy` does not have an `evaluate_actions`
                attribute (present in `ActorCriticPolicy`), needed to compute
                log-probability of actions.
        """
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )
        # AIRL needs a policy from STOCHASTIC_POLICIES to compute discriminator output.
        if not isinstance(self.gen_algo.policy, STOCHASTIC_POLICIES):
            raise TypeError(
                "AIRL needs a stochastic policy to compute the discriminator output.",
            )
        
        assert isinstance(demostrations_for_shaping, list), "demonstrations_for_shaping must be a list of Trajectory"
        assert isinstance(demostrations_for_shaping[0], types.Trajectory), "demonstrations_for_shaping must be a list of Trajectory"
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
        self.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        self.save_path = os.path.join(self.project_path, self.save_path)
        self.traj_index = traj_index
        if not os.path.exists(self.save_path):
            print("creating save path")
            os.makedirs(self.save_path)
        # path from str to Path
        # self.save_path = Path(self.save_path)

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

    def progress_shaping_loss(self) -> th.Tensor:

        '''
        get progress from annotations and compute the progress shaping loss
        '''
        # randomly choose some demonstrations from annotation_list
        # randomly generate a batch of indices
        indices = np.random.choice(len(self.annotation_list), self.shaping_batch_size , replace=False)
        threshold = 0.1
        # get corresponding annotations
        annotations = [self.annotation_list[idx] for idx in indices]
        #print("indices", indices)
        # get the progress change from annotations
        # print("type check",type(annotations[0][0]["start_progress"]))
        delta_progress = th.tensor([annotation[0]["end_progress" ]  - annotation[0]["start_progress"] -  threshold for annotation in annotations])
        #print("delta_progress", delta_progress)
        # print("what is this",annotations[0][0])
        #progress = th.tensor([annotation[0]["end_progress"] for annotation in annotations])

        # get the average progress value
        average_progress_value = th.tensor(([(annotation[0]["start_progress"] + annotation[0]["end_progress"]) / 2 for annotation in annotations]))

        # get corresponding states and actions
        demostration_indicies = [(annotation[1], annotation[0]["start_step"], annotation[0]["end_step"]) for annotation in annotations]
        states = [th.tensor(self.demonstrations_for_shaping[demostration_index].obs[start_step:end_step], dtype=th.float32) for demostration_index, start_step, end_step in demostration_indicies]
        actions = [th.tensor(self.demonstrations_for_shaping[demostration_index].acts[start_step:end_step], dtype=th.float32) for demostration_index, start_step, end_step in demostration_indicies]
        
        next_states = [th.tensor(self.demonstrations_for_shaping[demostration_index].obs[start_step+1:end_step+1], dtype=th.float32) for demostration_index, start_step, end_step in demostration_indicies]
        dones = [int(self.demonstrations_for_shaping[demostration_index].terminal) for demostration_index, _, end_step in demostration_indicies]

        # record corresponding index for each part of the batch
        state_lengths = th.tensor([len(state) for state in states])
        # add 0 to the beginning of the tensor
        state_lengths = th.cat((th.tensor([0]), state_lengths))

        # accumulate the lengths of the states
        state_indicies = th.cumsum(state_lengths, dim=0)
        
        # concatenate the states and actions
        states = th.cat(states)
        actions = th.cat(actions)
        next_states = th.cat(next_states)
        dones = th.tensor(dones, dtype=th.float32)

        # length check
        if len(states)!= len(next_states) or len(states) != len(actions):
            # reduce the length of states by the minimum length
            min_length = min(len(states), len(next_states), len(actions))
            states = states[:min_length]
            actions = actions[:min_length]
            next_states = next_states[:min_length]
            dones = dones[:min_length]
    

        # to device
        states = states.to(self.gen_algo.device)
        actions = actions.to(self.gen_algo.device)
        next_states = next_states.to(self.gen_algo.device)
        dones = dones.to(self.gen_algo.device)

        # get the reward output from the reward network
        
        reward_output_train = self._reward_net.base(states, actions, next_states, dones)
        v_s = self._reward_net.potential(states)
        v_s_next = self._reward_net.potential(next_states)
        delta_value = v_s_next - v_s
        advatanage_output = th.tensor([0.0 for i in range(len(states))])
        advatanage_output = self._reward_net(states, actions, next_states, th.tensor(1))


        # get the value output from the potential part of reward network
        old_value_output_train = self._reward_net.potential(states).flatten()
        next_value_output_train = self._reward_net.potential(next_states).flatten()


        # sum the reward output for each trajectory
        reward_output_train = th.stack([reward_output_train[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)])

        delta_value = th.stack([delta_value[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)])
        advatanage_output = th.stack([advatanage_output[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)]) 
        

        # sum the value output for each trajectory
        old_value_output_train = th.stack([old_value_output_train[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)])
        next_value_output_train = th.stack([next_value_output_train[state_indicies[i]:state_indicies[i+1]].sum() for i in range(len(state_lengths)-1)])
        
        # the reward sum should have same length as delta_progress
        assert len(reward_output_train) == len(delta_progress), "reward_output_train and delta_progress should have same length"
        assert len(old_value_output_train) == len(delta_progress), "old_value_output_train and delta_progress should have same length"
        assert len(next_value_output_train) == len(delta_progress), "next_value_output_train and delta_progress should have same length"

        # to device
        delta_progress = delta_progress.to(self.gen_algo.device)
        average_progress_value = average_progress_value.to(self.gen_algo.device)
        reward_output_train = reward_output_train.to(self.gen_algo.device)
        old_value_output_train = old_value_output_train.to(self.gen_algo.device)
        next_value_output_train = next_value_output_train.to(self.gen_algo.device)

        #loss_sign = self.progress_sign_loss(delta_progress, reward_output_train)
        loss_scale = self.delta_progress_reward_loss(delta_progress, reward_output_train)
        loss_value = self.value_sign_loss(delta_progress, delta_value)
        loss_advantage = self.advantage_sign_loss(delta_progress, advatanage_output)
        loss_progress_reward= self.reward_sign_loss(average_progress_value, next_value_output_train)
        if "subtrajectory_proportion_loss" in self.shape_reward:
            loss_proportion = self.subtrajectory_proportion_loss()
        if "end_progress_loss" in self.shape_reward:
            loss_end_progress = self.end_progress_loss()

        # return the loss
        return_dict = {"delta_progress_reward_loss": loss_scale,
                        "value_sign_loss": loss_value,
                        "advantage_sign_loss": loss_advantage,
                        "reward_sign_loss": loss_progress_reward,
                        #"subtrajectory_proportion_loss": loss_proportion,
                        #"end_progress_loss": loss_end_progress
                        }
        if "end_progress_loss" in self.shape_reward:
            return_dict["end_progress_loss"] = loss_end_progress
        if "subtrajectory_proportion_loss" in self.shape_reward:
            return_dict["subtrajectory_proportion_loss"] = loss_proportion

        return return_dict

    def advantage_sign_loss(self,
                            delta_progress: th.Tensor,
                            delta_advantage: th.Tensor) -> th.Tensor:
        """
        Compare the sign of delta_progress and delta_advantage using binary cross-entropy.
        We want them to match: i.e. if delta_progress >= 0 then delta_advantage >= 0, else both < 0.
        """
        device = self.gen_algo.device

        # Convert the two tensors to {0, 1} sign bits
        progress_sign = th.relu(F.softsign(delta_progress)).to(device)  
        advantage_sign = th.relu(F.softsign(delta_advantage)).to(device)
        # print("progress_sign", progress_sign)
        # print("advantage_sign", advantage_sign)

        # BCE: advantage_sign is 'prediction', progress_sign is 'target'.
        # If you prefer, you can swap them; just be consistent across your code.
        loss = F.binary_cross_entropy(advantage_sign, progress_sign)
        return loss

    def reward_sign_loss(self, 
                         delta_progress: th.Tensor, 
                         reward_output_train: th.Tensor) -> th.Tensor:
        """
        Compare sign of delta_progress with sign of reward_output_train using binary cross-entropy.
        """
        device = self.gen_algo.device

        progress_sign = th.relu(F.softsign(delta_progress)).to(device)
        reward_sign   = th.relu(F.softsign(reward_output_train)).to(device)

        loss = F.binary_cross_entropy(reward_sign, progress_sign)
        return loss

    def value_sign_loss(self,
                        delta_progress: th.Tensor,
                        delta_value: th.Tensor) -> th.Tensor:
        """
        Compare sign of delta_progress with sign of (-delta_value).
        Originally you had sign_agreement = sign(delta_progress) * sign(-delta_value).
        That means if delta_progress is positive, -delta_value should be positive (i.e. delta_value negative).
        """
        device = self.gen_algo.device

        progress_sign = th.relu(F.softsign(delta_progress)).to(device)
        # sign(-delta_value) is 1 if delta_value <= 0, else 0
        value_sign    = th.relu(F.softsign(delta_value)).to(device)

        loss = F.binary_cross_entropy(value_sign, progress_sign)
        return loss      
    def delta_progress_reward_loss(self, 
                                delta_progress: th.tensor, 
                                reward_output_train: th.tensor)-> th.tensor:
        # loss should make the subtrajectory with higher delta_progress have higher reward_output_train
        
        # compute pairwise differences
        # delta_progress_diff = delta_progress.unsqueeze(1) - delta_progress.unsqueeze(0)
        # reward_output_train_diff = reward_output_train.unsqueeze(1) - reward_output_train.unsqueeze(0)

        # # we want reward_output_train_diff to be larger when delta_progress_diff is larger
        # # idea is that if delta_progress_diff > 0, then reward_output_train_diff > 0
        # loss = th.mean(th.relu(-F.softsign(delta_progress_diff * reward_output_train_diff)))

        device = self.gen_algo.device
        delta_progress_diff = delta_progress.unsqueeze(1) - delta_progress.unsqueeze(0)
        reward_output_train_diff = reward_output_train.unsqueeze(1) - reward_output_train.unsqueeze(0)
        delta_progress_diff = th.relu(F.softsign(delta_progress_diff)).to(device)
        reward_output_train_diff = th.relu(F.softsign(reward_output_train_diff)).to(device)
        loss = F.binary_cross_entropy(reward_output_train_diff, delta_progress_diff)


        return loss
    # def advantage_sign_loss(self,
    #                         delta_progress: th.Tensor,
    #                         delta_advantage: th.Tensor) -> th.Tensor:
    #     """
    #     we want sign(delta_progress) = sign(delta_advantage)
    #     """

    #     device = self.gen_algo.device

    #     # product is positive if signs match, negative if signs differ
    #     prod = delta_progress * delta_advantage

    #     # we want prod > 0 -> penalize if prod < 0
    #     # logistic loss: log(1 + exp(prod)) is small when prod << 0
    #     loss = th.log(1.0 + th.exp(prod)).mean().to(device)

    #     return loss
    
    # def reward_sign_loss(self,
    #                         delta_progress: th.Tensor,
    #                         reward_output_train: th.Tensor) -> th.Tensor:
    #     """
    #     We want sign(delta_progress) = sign(reward_output_train),
    #     i.e. delta_progress * reward_output_train > 0.
    #     A logistic penalty can be used to push product > 0.

    #     """

    #     device = self.gen_algo.device

    #     # product is positive if signs match, negative if signs differ
    #     prod = delta_progress * reward_output_train

    #     # we want prod > 0 -> penalize if prod < 0
    #     # logistic loss: log(1 + exp(prod)) is small when prod << 0

    #     loss = th.log(1.0 + th.exp(prod)).mean().to(device)

    #     return loss
    
    # def value_sign_loss(self,
    #                 delta_progress: th.Tensor,
    #                 delta_value: th.Tensor) -> th.Tensor:
    #     """
    #     We want sign(delta_progress) = opposite of sign(delta_value),
    #     i.e. delta_progress * delta_value < 0.
    #     A logistic penalty can be used to push product < 0.
    #     """
    #     device = self.gen_algo.device

    #     # product is positive if signs match, negative if signs differ
    #     prod = delta_progress * delta_value

    #     # we want prod < 0 -> penalize if prod > 0
    #     # logistic loss: log(1 + exp(prod)) is small when prod << 0
    #     loss = th.log(1.0 + th.exp(prod)).mean().to(device)

    #     return loss


    # def advantage_sign_loss(self,
    #                  delta_progress: th.tensor,
    #                     delta_advantage: th.tensor) -> th.tensor:
    #     advantage_agreement = (F.softsign(delta_progress).to(self.gen_algo.device)) * (F.softsign(delta_advantage).to(self.gen_algo.device))
    #     #loss = th.relu(-advantage_agreement)
    #     loss = th.mean(th.relu(-advantage_agreement))
    #     return loss
    
    # def reward_sign_loss(self, 
    #                        delta_progress: th.tensor, 
    #                        reward_output_train: th.tensor) -> th.tensor:
    #     # loss should be difference in the sign of delta_progress and reward_output_train
    #     sign_agreement = (F.softsign(delta_progress).to(self.gen_algo.device)) * (F.softsign(reward_output_train))
    #     #loss = th.relu(-sign_agreement)
    #     loss = th.mean(th.relu(-sign_agreement))
    #     return loss
    
    # def value_sign_loss(self,
    #                       delta_progress: th.tensor,
    #                       delta_value: th.tensor) -> th.tensor:
    #       # loss should be difference in the sign of delta_progress and delta_value
    #       sign_agreement = (F.softsign(delta_progress).to(self.gen_algo.device)) * (F.softsign(-delta_value))
    #       #loss = th.relu(-sign_agreement)
    #       loss = th.mean(th.relu(-sign_agreement))
    #       return loss

    

    def end_progress_loss(self) -> th.Tensor:
        """
        Compare the final progress of random pairs of demonstrations. 
        If final progress is similar (difference < 10), enforce the total 
        rewards to be within 10%. If final progress is different, enforce 
        the rewards to differ by at least 5%. Uses a piecewise, differentiable 
        penalty via torch operations (no .item() or NumPy).
        """
        device = self.gen_algo.device
        if len(self.demonstrations_for_shaping) < 2:
            # If we have fewer than 2 demos, nothing to compare
            print("Not enough demonstrations for end progress loss, returning zero.")
            return th.zeros((), device=device)

        # Choose how many random pairs to sample (e.g. 2 pairs):
        num_pairs = 4
        if len(self.traj_index) < 2:
            # Not enough distinct trajectories
            print("Not enough distinct trajectories for end progress loss, returning zero.")
            return th.zeros((), device=device)

        # Sample random distinct pairs of trajectory indices
        idxs = np.random.choice(self.traj_index, size=2 * num_pairs, replace=False)
        idx_pairs = [(idxs[2*k], idxs[2*k+1]) for k in range(num_pairs)]
        print("idx_pairs", idx_pairs)

        all_penalties = []

        for (i, j) in idx_pairs:
            # Find last segment for demo i
            segs_i = [(ann[0]["start_step"], ann[0]["end_step"], ann[0]["end_progress"])
                    for ann in self.annotation_list if ann[1] == i]
            if len(segs_i) < 1:
                continue
            segs_i.sort(key=lambda x: x[1])
            end_prog_i = segs_i[-1][2]  # final progress for i (float)
            
            # Find last segment for demo j
            segs_j = [(ann[0]["start_step"], ann[0]["end_step"], ann[0]["end_progress"])
                    for ann in self.annotation_list if ann[1] == j]
            if len(segs_j) < 1:
                continue
            segs_j.sort(key=lambda x: x[1])
            end_prog_j = segs_j[-1][2]  # final progress for j (float)
            
            # Convert those final progresses to Tensors (constant w.r.t. network)
            end_prog_i_t = th.tensor(end_prog_i, dtype=th.float32, device=device)
            end_prog_j_t = th.tensor(end_prog_j, dtype=th.float32, device=device)
            
            # Compute final reward for entire trajectory i in a differentiable manner
            traj_i = self.demonstrations_for_shaping[i]
            states_i = th.tensor(traj_i.obs, dtype=th.float32, device=device)
            acts_i   = th.tensor(traj_i.acts, dtype=th.float32, device=device)
            # Next-state can be shifted by 1. If you want the same length, do:
            next_i   = th.tensor(traj_i.obs, dtype=th.float32, device=device)
            dones_i  = th.zeros(len(states_i), dtype=th.float32, device=device)
            # Align shapes to min length
            min_len_i = min(len(states_i), len(acts_i), len(next_i))
            states_i = states_i[:min_len_i]
            acts_i   = acts_i[:min_len_i]
            next_i   = next_i[:min_len_i]
            dones_i  = dones_i[:min_len_i]
            
            rews_i = self._reward_net.base(states_i, acts_i, next_i, dones_i)
            # total reward remains a Tensor
            total_rew_i = rews_i.sum()

            # Do the same for j
            traj_j = self.demonstrations_for_shaping[j]
            states_j = th.tensor(traj_j.obs, dtype=th.float32, device=device)
            acts_j   = th.tensor(traj_j.acts, dtype=th.float32, device=device)
            next_j   = th.tensor(traj_j.obs, dtype=th.float32, device=device)
            dones_j  = th.zeros(len(states_j), dtype=th.float32, device=device)
            min_len_j = min(len(states_j), len(acts_j), len(next_j))
            states_j = states_j[:min_len_j]
            acts_j   = acts_j[:min_len_j]
            next_j   = next_j[:min_len_j]
            dones_j  = dones_j[:min_len_j]

            rews_j = self._reward_net.base(states_j, acts_j, next_j, dones_j)
            total_rew_j = rews_j.sum()

            # Now define piecewise penalty:
            # 1) If |end_prog_i - end_prog_j| < 10, we want |R_i - R_j| < 0.1 * max(R_i, R_j).
            # 2) Else, we want |R_i - R_j| > 0.05 * max(R_i, R_j).
            # We'll encode both in a single differentiable expression.

            progress_diff = th.abs(end_prog_i_t - end_prog_j_t)
            reward_diff   = th.abs(total_rew_i - total_rew_j)
            reward_max    = th.max(total_rew_i, total_rew_j).clamp_min(1e-6)  # avoid / 0

            # Condition: progress_diff < 10
            similar_mask = (progress_diff < 10.0)
            dissimilar_mask = ~similar_mask  # logical NOT

            # For similar: penalty = relu( |r_i - r_j| - 0.1 * max ) / max
            # If they're "too far apart," this penalty is > 0
            similar_penalty = F.relu(reward_diff - 0.1 * reward_max) / reward_max

            # For dissimilar: penalty = relu(0.05*max - |r_i - r_j| ) / max
            # If they're "too close," penalty is > 0
            dissimilar_penalty = F.relu(0.05 * reward_max - reward_diff) / reward_max

            # Combine them using a torch.where
            pair_penalty = th.where(similar_mask, similar_penalty, dissimilar_penalty)

            all_penalties.append(pair_penalty)

        if len(all_penalties) == 0:
            print("We never found any valid loss in end progress, returning zero.")
            return th.zeros((), device=device)

        # Average penalty over the sampled pairs
        loss = th.mean(th.stack(all_penalties))
        return loss




    
    def subtrajectory_proportion_loss(self) -> th.Tensor:
        """
        For each sampled trajectory, we divide it by annotation segments [1..K].
        If segment k ends at progress p_k, and the entire trajectory ends at progress p_final,
        then the portion of total progress up to segment k is (p_k / p_final).
        We want that to match the portion of total reward up to segment k, i.e. 
        (sum_of_rewards_up_to_segment_k / total_reward).

        This function returns a loss that penalizes the L1 difference between those proportions.
        """
        device = self.gen_algo.device

        # If there are no demonstrations, return 0
        if len(self.demonstrations_for_shaping) < 1:
            print("No demonstrations found, returning zero.")
            return th.zeros((), device=device)

        # Randomly select some trajectory indices to compare
        if len(self.traj_index) < 1:
            print("Not enough distinct trajectories, returning zero.")
            return th.zeros((), device=device)

        # Suppose we pick 2 random trajectories (or pick more if you like)
        idxs = np.random.choice(self.traj_index, size=2, replace=False)

        all_losses = []

        for i in idxs:
            # Gather segments for demonstration i
            segments = [
                (ann[0]["start_step"], ann[0]["end_step"], ann[0]["end_progress"])
                for ann in self.annotation_list 
                if ann[1] == i
            ]
            if len(segments) < 1:
                # No segmentation => skip
                print("No segments found for trajectory, skipping.")
                continue
            print("segments", segments)
            # Sort segments by ascending end_step
            segments.sort(key=lambda x: x[1])

            # Load entire trajectory i
            traj = self.demonstrations_for_shaping[i]
            states_all = th.tensor(traj.obs, dtype=th.float32, device=device)
            actions_all = th.tensor(traj.acts, dtype=th.float32, device=device)
            next_all = th.tensor(traj.obs, dtype=th.float32, device=device)
            dones_all = th.zeros(len(states_all), dtype=th.float32, device=device)

            # Align lengths so they match
            min_len = min(len(states_all), len(actions_all), len(next_all))
            states_all = states_all[:min_len]
            actions_all = actions_all[:min_len]
            next_all = next_all[:min_len]
            dones_all = dones_all[:min_len]

            # Compute all rewards as a torch Tensor
            all_rews = self._reward_net.base(states_all, actions_all, next_all, dones_all)
            total_reward = all_rews.sum()  # Tensor

            # final progress from the last segment
            final_progress_val = segments[-1][2]
            # If final progress is effectively zero, skip to avoid division by zero
            if abs(final_progress_val) < 1e-6:
                continue

            # Convert to Torch tensor (constant for the calculation)
            final_progress = th.tensor(final_progress_val, dtype=th.float32, device=device)

            partial_losses = []
            # For each subtrajectory segment, compute partial sums and compare
            for (st, en, seg_prog_val) in segments:
                seg_prog = th.tensor(seg_prog_val, dtype=th.float32, device=device)

                # sub_r is the cumulative reward from step 0 to step `en`
                # in PyTorch:
                sub_r = all_rews[:en].sum()  # still a Tensor

                # proportions:
                prop_progress = seg_prog / final_progress
                prop_rewards  = sub_r / (total_reward + 1e-8)

                # L1 difference (or use MSE if you prefer)
                diff = th.abs(prop_progress - prop_rewards)
                partial_losses.append(diff)

            if len(partial_losses) > 0:
                # average difference across subtrajectory segments
                demo_loss = th.mean(th.stack(partial_losses))
                all_losses.append(demo_loss)

        if len(all_losses) == 0:
            print("We never found any valid subtrajectories loss, returning zero.")
            return th.zeros((), device=device)

        # Average over all sampled trajectories
        return th.mean(th.stack(all_losses))


    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        """Returns the unshaped version of reward network used for testing."""
        reward_net = self._reward_net
        # Recursively return the base network of the wrapped reward net
        while isinstance(reward_net, reward_nets.RewardNetWrapper):
            reward_net = reward_net.base
        return reward_net


    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        
        '''
        a modified version of train_disc from airl.py with the following changes:
        - add one more training step for the discriminator to shape the reward
        '''


        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.demo_batch_size` samples. If this argument is not provided, then
                `self.demo_batch_size` expert samples from `self.demo_data_loader` are
                used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.demo_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.

        Returns:
            Statistics for discriminator (e.g. loss, accuracy).
        """

        

        # original training code
        with self.logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0
            self._disc_opt.zero_grad()



            # compute loss
            

            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )

            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    batch["labels_expert_is_one"].float(),
                )

                # Renormalise the loss to be averaged over the whole
                # batch size instead of the minibatch size.
                #print("loss before:", loss)
                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                
                if len(self.shape_reward) > 0 and self._disc_step % self.shaping_update_freq == 0:
                #self._disc_opt.zero_grad()
                    shaping_losses = self.progress_shaping_loss()
                    # get the losses in self.shape_reward list using keys, sum them
                    shaping_loss = sum([shaping_losses[key] for key in self.shape_reward])
                    print("************************************************************")
                    for key in self.shape_reward:
                        print(key, shaping_losses[key])
                    print("AIRL loss:", loss)
                    print("************************************************************")
                else:
                    shaping_loss = th.tensor(0.0, device=self.gen_algo.device)

                combined_loss = loss + shaping_loss
                combined_loss.backward()
                #loss.backward()

            # do gradient step
            self._disc_opt.step()
            self._disc_step += 1

            # reward shaping loss

                # relase unused loss
                # del shaping_losses


                # if "progress_sign_loss" in self.shape_reward and self._disc_step % self.shaping_update_freq == 0:
                #     self._disc_opt.zero_grad()
                #     shaping_loss = self.progress_shaping_loss()
                #     print(shaping_loss)
                #     shaping_loss *= self.shaping_loss_weight
                #     shaping_loss.backward()
                #     self._disc_opt.step()

            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    batch["labels_expert_is_one"],
                    loss,
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())
        # save the model
        if self._global_step % self.save_model_every == 0:
            # set save path according to the global step
            save_path_this_time = os.path.join(self.save_path, f"{self._global_step}")
            if not os.path.exists(save_path_this_time):
                os.makedirs(save_path_this_time)
            save_path_this_time = Path(save_path_this_time)
            train_adversarial.save(self, 
                                   save_path_this_time,)
        return train_stats
    
