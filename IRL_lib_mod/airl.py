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
        save_model_every = 20,
        save_path = "checkpoints/default",
        use_online_data: bool = False,
        demonstration_online_data: Optional[Iterable[types.Trajectory]] = None,
        online_annotation_list: Optional[list[tuple[int, dict]]] = None,
        online_shaping_batch_size: int = 16,
        online_shaping_loss_weight: float = 1.0,
        online_shaping_update_freq: int = 1,
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
        
        self.use_online_data = use_online_data
        self.demonstration_online_data = demonstration_online_data
        self.online_annotation_list = online_annotation_list

        self.online_shaping_batch_size = online_shaping_batch_size
        self.online_shaping_loss_weight = online_shaping_loss_weight
        self.online_shaping_update_freq = online_shaping_update_freq
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

    def progress_shaping_loss(self, 
                              demonstrations, 
                              annotation_list) -> th.Tensor:

        '''
        get progress from annotations and compute the progress shaping loss
        '''
        # randomly choose some demonstrations from annotation_list
        # randomly generate a batch of indices
        indices = np.random.choice(len(annotation_list), self.shaping_batch_size , replace=False)
        threshold = 0.1
        # get corresponding annotations
        annotations = [annotation_list[idx] for idx in indices]

        # get the progress change from annotations
        # print("type check",type(annotations[0][0]["start_progress"]))
        delta_progress = th.tensor([annotation[0]["start_progress"] - annotation[0]["end_progress" ]  - threshold for annotation in annotations])
        # print("what is this",annotations[0][0])
        #progress = th.tensor([annotation[0]["end_progress"] for annotation in annotations])

        # get the average progress value
        average_progress_value = th.tensor(([(annotation[0]["start_progress"] + annotation[0]["end_progress"]) / 2 for annotation in annotations]))

        # get corresponding states and actions
        demostration_indicies = [(annotation[1], annotation[0]["start_step"], annotation[0]["end_step"]) for annotation in annotations]
        states = [th.tensor(demonstrations[demostration_index].obs[start_step:end_step], dtype=th.float32) for demostration_index, start_step, end_step in demostration_indicies]
        actions = [th.tensor(demonstrations[demostration_index].acts[start_step:end_step], dtype=th.float32) for demostration_index, start_step, end_step in demostration_indicies]
        
        next_states = [th.tensor(demonstrations[demostration_index].obs[start_step+1:end_step+1], dtype=th.float32) for demostration_index, start_step, end_step in demostration_indicies]
        dones = [int(demonstrations[demostration_index].terminal) for demostration_index, _, end_step in demostration_indicies]

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

        loss_sign = self.progress_sign_loss(delta_progress, reward_output_train)
        loss_scale = self.delta_progress_scale_loss(delta_progress, reward_output_train)
        loss_value = self.value_sign_loss(delta_progress, delta_value)
        loss_advantage = self.advantage_sign_loss(delta_progress, advatanage_output)
        loss_progress_value = self.progress_value_loss(average_progress_value, next_value_output_train)

        return {"progress_sign_loss": loss_sign, "delta_progress_scale_loss": loss_scale, "value_sign_loss": loss_value, "advantage_sign_loss": loss_advantage, "progress_value_loss": loss_progress_value}

        
    def value_sign_loss(self,
                   delta_progress: th.tensor,
                   delta_value: th.tensor) -> th.tensor:
        
        value_agreement = (th.sign(delta_progress).to(self.gen_algo.device)) * (th.sign(delta_value))
        loss = th.mean(th.relu(-value_agreement))
        return loss
    def advantage_sign_loss(self,
                     delta_progress: th.tensor,
                        delta_advantage: th.tensor) -> th.tensor:
        advantage_agreement = (th.sign(delta_progress).to(self.gen_algo.device)) * (th.sign(delta_advantage))
        loss = th.mean(th.relu(-advantage_agreement))
        return loss
    
    def progress_sign_loss(self, 
                           delta_progress: th.tensor, 
                           reward_output_train: th.tensor) -> th.tensor:
        # loss should be difference in the sign of delta_progress and reward_output_train
        sign_agreement = (th.sign(delta_progress).to(self.gen_algo.device)) * (th.sign(reward_output_train))
        loss = th.mean(th.relu(-sign_agreement))
        return loss
    
    def delta_progress_scale_loss(self, 
                                delta_progress: th.tensor, 
                                reward_output_train: th.tensor)-> th.tensor:
        # loss should make the subtrajectory with higher delta_progress have higher reward_output_train
        
        # compute pairwise differences
        delta_progress_diff = delta_progress.unsqueeze(1) - delta_progress.unsqueeze(0)
        reward_output_train_diff = reward_output_train.unsqueeze(1) - reward_output_train.unsqueeze(0)

        # we want reward_output_train_diff to be larger when delta_progress_diff is larger
        # idea is that if delta_progress_diff > 0, then reward_output_train_diff > 0
        loss = th.mean(th.relu(-th.sign(delta_progress_diff * reward_output_train_diff)))
        return loss
    
    def progress_value_loss(self, 
                            average_progress: th.tensor, 
                            next_value_output_train: th.tensor) -> th.tensor:
        # the higher the average progress, the higher the value output should be
        average_progress_diff = average_progress.unsqueeze(1) - average_progress.unsqueeze(0)
        next_value_output_train_diff = next_value_output_train.unsqueeze(1) - next_value_output_train.unsqueeze(0)

        # we want next_value_output_train_diff to be larger when average_progress_diff is larger
        # idea is that if average_progress_diff > 0, then next_value_output_train_diff > 0
        loss = th.mean(th.relu(-th.sign(average_progress_diff * next_value_output_train_diff)))
        return loss


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

            # compute loss
            self._disc_opt.zero_grad()

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
                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                loss.backward()

            # do gradient step
            self._disc_opt.step()
            self._disc_step += 1

            # reward shaping loss
            if len(self.shape_reward) > 0 and self._disc_step % self.shaping_update_freq == 0:
                self._disc_opt.zero_grad()
                shaping_losses = self.progress_shaping_loss(
                    self.demonstrations_for_shaping, self.annotation_list
                )
                # get the losses in self.shape_reward list using keys, sum them
                shaping_loss = sum([shaping_losses[key] for key in self.shape_reward])
                # print(shaping_loss)
                shaping_loss *= self.shaping_loss_weight
                shaping_loss.backward()
                self._disc_opt.step()

            # online reward shaping loss
            if self.use_online_data and self._disc_step % self.online_shaping_update_freq == 0:
                self._disc_opt.zero_grad()
                shaping_losses = self.progress_shaping_loss(self.demonstration_online_data, self.online_annotation_list)
                shaping_loss = sum([shaping_losses[key] for key in self.shape_reward])
                shaping_loss *= self.online_shaping_loss_weight
                shaping_loss.backward()
                self._disc_opt.step()

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
    