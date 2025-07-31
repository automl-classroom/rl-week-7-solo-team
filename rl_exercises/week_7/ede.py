from typing import Tuple

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import trange
from collections import deque, namedtuple

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def set_seed(env: gym.Env, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    except TypeError:
        pass


def huber(u: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    return torch.where(u.abs() <= k, 0.5 * u.pow(2), k * (u.abs() - 0.5 * k))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.Replay = namedtuple(
            "Replay", ("state", "action", "reward", "next_state", "done")
        )
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(self.Replay(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        batch = self.Replay(*zip(*batch))
        return (
            torch.tensor(batch.state, dtype=torch.float32, device=DEVICE),
            torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(-1),
            torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(
                -1
            ),
            torch.tensor(batch.next_state, dtype=torch.float32, device=DEVICE),
            torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)


class QuantileEnsemble(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, num_heads: int, num_quantiles: int
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.num_quantiles = num_quantiles

        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, action_dim * num_quantiles),
                )
                for _ in range(num_heads)
            ]
        )

        self.register_buffer(
            "taus",
            (torch.arange(num_quantiles, dtype=torch.float32) + 0.5) / num_quantiles,
        )

    def forward(self, x: torch.Tensor, detach_feature_for: int | None = None):
        """
        detach_feature_for: index of the head allowed to back-prop through the shared torso
        """
        feat = self.feature(x)
        outputs = []
        for i, head in enumerate(self.heads):
            f = (
                feat.detach()
                if (detach_feature_for is not None and i != detach_feature_for)
                else feat
            )
            out = head(f).view(-1, self.action_dim, self.num_quantiles)
            outputs.append(out)
        return torch.stack(outputs)

    def q_values(self, quantiles: torch.Tensor) -> torch.Tensor:
        return quantiles.mean(-1)

    def mean_and_sigma(
        self, quantiles: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_values(quantiles)
        q_mean = q.mean(0)
        sigma = q.var(0, unbiased=False).sqrt()
        return q_mean, sigma


class EDEAgent:
    def __init__(self, cfg: DictConfig):
        self.env = gym.make(cfg.env.name)
        set_seed(self.env, cfg.seed)
        self.seed = cfg.seed

        self.gamma = cfg.agent.gamma
        self.lr = cfg.agent.learning_rate
        self.batch = cfg.agent.batch_size
        self.start = cfg.agent.start_steps
        self.device = torch.device(DEVICE)
        self.target_update_freq = cfg.agent.target_update_freq

        self.N = cfg.agent.num_quantiles
        self.M = cfg.agent.num_heads
        self.phi = cfg.agent.phi_ucb

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.net = QuantileEnsemble(obs_dim, act_dim, self.M, self.N).to(self.device)
        self.tgt = QuantileEnsemble(obs_dim, act_dim, self.M, self.N).to(self.device)
        self.tgt.load_state_dict(self.net.state_dict())

        self.buffer = ReplayBuffer(cfg.agent.buffer_capacity)
        self.opt = optim.Adam(self.net.parameters(), lr=self.lr, eps=1.5e-4)

        self.step = 0
        self.steps = []
        self.returns = []

    def choose_action(self, state: np.ndarray) -> int:
        # UCB action selection
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            quant = self.net(s)
            q_mean, sigma = self.net.mean_and_sigma(quant)
            ucb = q_mean + self.phi * sigma
            return int(torch.argmax(ucb, dim=-1))

    def update(self):
        """Sample a batch and do one gradient step on the EDE loss."""
        if len(self.buffer) < self.batch:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch)

        # choose one head that actually backpropagates to maintain diversity
        gradient_head = random.randrange(self.M)
        quant = self.net(states, detach_feature_for=gradient_head)
        B = states.size(0)
        idx = torch.arange(B, device=self.device)
        q_pred_a = quant[:, idx, actions.squeeze(-1)]

        with torch.no_grad():
            next_qs = self.tgt(next_states)
            qmean, _ = self.net.mean_and_sigma(next_qs)
            next_a = torch.argmax(qmean, dim=-1)
            q_next_a = next_qs[:, idx, next_a]
            target = rewards + self.gamma * (1.0 - dones) * q_next_a

        taus = self.net.taus
        td = target.unsqueeze(-2) - q_pred_a.unsqueeze(-1)
        loss = (
            (torch.abs(taus.view(1, 1, -1, 1) - (td.detach() < 0).float()) * huber(td))
            .mean(3)
            .mean(2)
            .mean(0)
        ).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0, foreach=True)
        self.opt.step()

    def train(self, num_frames: int, eval_interval: int):
        state, _ = self.env.reset()
        ep_ret = 0.0
        ep_count = 0
        recent_returns = []

        for frame in trange(1, num_frames + 1):
            a = (
                self.env.action_space.sample()
                if self.step < self.start
                else self.choose_action(state)
            )
            ns, r, terminated, truncated, _ = self.env.step(a)
            done = terminated or truncated
            self.buffer.push(state, a, r, ns, float(done))
            state = ns
            ep_ret += r
            self.step += 1

            self.update()

            if self.step % self.target_update_freq == 0:
                self.tgt.load_state_dict(self.net.state_dict())

            if done:
                ep_count += 1
                recent_returns.append(ep_ret)
                self.returns.append(ep_ret)
                self.steps.append(frame)
                state, _ = self.env.reset()
                ep_ret = 0.0

                avg = float(np.mean(recent_returns[-eval_interval:]))
                # print(
                #     f"Frame {frame} | Episodes {ep_count} | AvgReturn({eval_interval}) {avg:.2f}"
                # )

        df = pd.DataFrame({"episode": self.steps, "return": self.returns})
        fname = f"training_data_seed_{self.seed}.csv"
        df.to_csv(fname, index=False)
        print(f"Training complete. Saved data to {fname}")


@hydra.main(config_path="../configs/agent", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    agent = EDEAgent(cfg)
    agent.train(
        num_frames=cfg.train.num_frames,
        eval_interval=cfg.train.eval_interval,
    )


if __name__ == "__main__":
    main()
