from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent

from cs285.infrastructure import pytorch_util as ptu

class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        loss = OrderedDict()
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)

        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            loss['Critic_Loss'] = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        advantages = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        advantages = ptu.from_numpy(advantages)
        
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            loss['Actor_Loss'] = self.actor.update(ob_no, ac_na, adv_n=advantages)

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        v_s_n = self.critic(ob_no)
        v_s_prime_n = self.critic(next_ob_no)
        # setting V(s') to zero if the next state is a terminal state
        q_n = re_n + self.gamma * v_s_prime_n * (1 - terminal_n)
        adv_n = q_n - v_s_n
        assert adv_n.size() == re_n.size()
        adv_n = adv_n.detach().cpu().numpy()

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
