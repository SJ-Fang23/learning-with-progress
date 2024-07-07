"""Adversarial Inverse Reinforcement Learning (AIRL)."""
from typing import Optional

import torch as th
import torch
from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets
from utils import annotation_utils
import numpy as np
from torch.nn import functional as F

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
        **kwargs,
    ):
        """Builds an AIRL trainer.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: Reward network; used as part of AIRL discriminator.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.

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
        self.demonstrations = demonstrations

        self.annotations = annotation_utils.read_all_json("")
        # print(demonstrations[0].acts[annotations['demo_1'][2]['end_step']])
        # print("terminal:", demonstrations[0].terminal)

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
        

        self.reward_shaping()
        reward_output_train = self._reward_net(state, action, next_state, done)
        print("reward_net device:", self._reward_net.device)
        print("base device:", self._reward_net.base.device)
        # print("potential device:", self._reward_net.potential.device)

        return reward_output_train - log_policy_act_prob

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
    
    def reward_shaping(self, batch_size = 2):
        for _ in range(batch_size):
            keys = list(self.annotations.keys())
 #           print(keys)
            traj = np.random.choice(keys)
            num = traj.split("_")[1]
            num = int(num)
            # print(traj)
            # print(num)
            for i in range(len(self.annotations[traj])):
                ann_states = self.demonstrations[num].obs[self.annotations[traj][i]['start_step']:self.annotations[traj][i]['end_step']]
                ann_actions = self.demonstrations[num].acts[self.annotations[traj][i]['start_step']:self.annotations[traj][i]['end_step']]
                ann_next_states = self.demonstrations[num].obs[self.annotations[traj][i]['start_step']+1:self.annotations[traj][i]['end_step']+1]
                ann_dones = self.demonstrations[num].terminal

                ann_states = [torch.tensor(state, device='cuda', dtype=torch.float32) for state in ann_states]
                ann_actions = [torch.tensor(action, device='cuda', dtype=torch.float32) for action in ann_actions]
                ann_next_states = [torch.tensor(next_state, device='cuda', dtype=torch.float32) for next_state in ann_next_states]

                ann_states = torch.stack([torch.tensor(state, device='cuda', dtype=torch.float32) for state in ann_states])
                ann_actions = torch.stack([torch.tensor(action, device='cuda', dtype=torch.float32) for action in ann_actions])
                ann_next_states = torch.stack([torch.tensor(next_state, device='cuda', dtype=torch.float32) for next_state in ann_next_states])

                ann_dones = torch.tensor(ann_dones, device='cuda', dtype=torch.float32)

                r_i = self._reward_net.base(ann_states, ann_actions, ann_next_states, ann_dones)
                
                v_i = self._reward_net.potential(ann_states).flatten()

                # positive or negative rewards constraint loss
                if self.annotations[traj][i]['start_progress'] <  self.annotations[traj][i]['end_progress']:
                    progress_i = self.annotations[traj][i]['end_progress'] - self.annotations[traj][i]['start_progress']
                    progress_i = progress_i / 100
                    sum_r_i = torch.sum(r_i)
                    loss = torch.max(torch.tensor(0.0, device='cuda'), - sum_r_i + torch.tensor(progress_i, device='cuda'))
                    #print("r_i:", r_i, "progress_i:", progress_i, "loss:", loss)
                    loss.backward()
                else:
                    progress_i = self.annotations[traj][i]['start_progress'] - self.annotations[traj][i]['end_progress']
                    progress_i = progress_i / 100
                    sum_r_i = torch.sum(r_i)
                    loss = torch.max(torch.tensor(0.0, device='cuda'), sum_r_i - torch.tensor(progress_i, device='cuda'))
                    #print("r_i:", r_i, "progress_i:", progress_i, "loss:", loss)
                    loss.backward()






                    # start_r = self._reward_net.base(ann_states[0], ann_actions[0], ann_next_states[0], ann_dones[0])
                    # end_r = self._reward_net.base(ann_states[-1], ann_actions[-1], ann_next_states[-1], ann_dones[-1])

                    # loss_r = F.binary_cross_entropy_with_logits(start_r, end_r)


                # progress_i = self.annotations[traj][i]['end_progress'] - self.annotations[traj][i]['start_progress']

                #slope loss
                # for j in range(i+1, len(self.annotations[traj])):
                #     ann_states = self.demonstrations[num].obs[self.annotations[traj][j]['start_step']:self.annotations[traj][j]['end_step']]
                #     ann_actions = self.demonstrations[num].acts[self.annotations[traj][j]['start_step']:self.annotations[traj][j]['end_step']]
                #     ann_next_states = self.demonstrations[num].obs[self.annotations[traj][j]['start_step']+1:self.annotations[traj][j]['end_step']+1]
                    # ann_states = torch.stack([torch.tensor(state, device='cuda', dtype=torch.float32) for state in ann_states])
                    # ann_actions = torch.stack([torch.tensor(action, device='cuda', dtype=torch.float32) for action in ann_actions])
                    # ann_next_states = torch.stack([torch.tensor(next_state, device='cuda', dtype=torch.float32) for next_state in ann_next_states])

                #     ann_dones = self.demonstrations[num].terminal

                #     r_j = self._reward_net.base(ann_states, ann_actions, ann_next_states, ann_dones)
                #     v_j = self._reward_net.potential(ann_states).flatten()
                #     progress_j = self.annotations[traj][j]['end_progress'] - self.annotations[traj][j]['start_progress']
                #     diff = progress_j - progress_i
                #     diff = diff / 100
                #     loss = torch.max(torch.tensor(0.0, device='cuda'), r_i - r_j + torch.tensor(diff, device='cuda'))



                    


        # # Slicing the arrays
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # ann_states = self.demonstrations[0].obs[self.annotations['demo_1'][2]['start_step']:self.annotations['demo_1'][2]['end_step']]
        # ann_actions = self.demonstrations[0].acts[self.annotations['demo_1'][2]['start_step']:self.annotations['demo_1'][2]['end_step']]
        # ann_next_states = self.demonstrations[0].obs[self.annotations['demo_1'][2]['start_step']+1:self.annotations['demo_1'][2]['end_step']+1]
        # ann_dones = self.demonstrations[0].terminal

        # # Convert each element to a torch tensor
        # ann_states = [torch.tensor(state, device=device, dtype=torch.float32) for state in ann_states]
        # ann_actions = [torch.tensor(action, device=device, dtype=torch.float32) for action in ann_actions]
        # ann_next_states = [torch.tensor(next_state, device=device, dtype=torch.float32) for next_state in ann_next_states]
        # ann_dones = torch.tensor(ann_dones, device=device, dtype=torch.float32)


        # reward_output_ann = self._reward_net(ann_states, ann_actions, ann_next_states, ann_dones)

        # base_reward_net_output = self._reward_net.base(state, action, next_state, done)
        # new_shaping_output = self._reward_net.potential(next_state).flatten()
        # old_shaping_output = self._reward_net.potential(state).flatten()
        # new_shaping_output = (1 - done.float()) * new_shaping_output


        # Q_s = base_reward_net_output + new_shaping_output * self._reward_net.discount_factor
        # V_s = old_shaping_output


        # print(Q_s)
        # print(V_s)

        # print("*************************************")
        # print("reward_output_ann:", reward_output_ann)
        # print("*************************************")