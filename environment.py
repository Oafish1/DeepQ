import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class balanceStick:
    def __init__(self):
        # Units: meters, kg, seconds
        self.length = 1
        self.mass = 1
        self.friction_coef = .2
        self.gravity = 9.8
        self.bounds = (-5, 5)
        self.tickrate = 16
        self.reset()

    def reset(self):
        self.angle = (np.pi / 16) * (np.pi * (np.random.rand() - .5)) + (np.pi / 2)  # Tilt up to pi/16
        self.x = 0  # np.random.rand() * (self.bounds[1] - self.bounds[0]) + self.bounds[0]  # Anywhere in bounds
        self.velocity = 400 * (np.random.rand() - .5)  # -200 to 200
        self.tip_ang_vel = 0  # 6 * (np.random.rand() - .5)  # -3 to 3

        # Return game state
        return torch.tensor([[self.x, self.angle, self.velocity]]).float()

    def tick(self, force_x):
        # Apply force
        old_velocity = self.velocity
        self.velocity += force_x / (self.mass * self.tickrate)
        # Apply friction (omitting mass)
        friction_vel = self.friction_coef * self.gravity / self.tickrate
        if np.abs(self.velocity) > np.abs(friction_vel):
            self.velocity += -np.sign(self.velocity) * friction_vel
        else:
            self.velocity = 0

        # Update position
        self.x += self.velocity / self.tickrate
        if self.x < self.bounds[0]:
            self.x = self.bounds[0]
            self.velocity = 0
        elif self.x > self.bounds[1]:
            self.x = self.bounds[1]
            self.velocity = 0

        # Approximate angle
        vel_change = self.velocity - old_velocity
        self.tip_ang_vel += vel_change * np.sin(self.angle) / (self.tickrate * self.length)

        # Apply gravity
        tip_accel = -np.cos(self.angle) * self.gravity
        self.tip_ang_vel += tip_accel / (self.tickrate**2 * self.length)

        # Update angle (L = theta r)
        self.angle += self.tip_ang_vel
        self.angle = self.angle % (2 * np.pi)

        # Return game state
        return torch.tensor([[self.x, self.angle, self.velocity]]).float()

    def render(self, ax=None, state=None):
        if state is not None:
            x, angle, _ = state
        else:
            x, angle = self.x, self.angle

        x = (x, x + np.cos(angle))
        y = (0, np.sin(angle))
        if ax is not None:
            ax.plot(x, y, 'g-', linewidth=2, markersize=10)
            ax.set_aspect('equal', 'box')
            ax.set(xlim=self.bounds, ylim=(-.1, self.length))

        return x, y


class Agent(nn.Module):
    """Thin, simple NN model"""
    def __init__(self, input_dim, output_dim, hidden_dim=10):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, input_dim),
            nn.LeakyReLU(),

            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        """Forward pass for the model"""
        return self.layers(x)


if __name__ == "__main__":
    # Params
    headless = True
    replay_best = True
    replay_best_interval = 100
    show_plt = not headless or replay_best

    training_plot = True
    training_plot_interval = 10

    cli_print_interval = 50

    # Env setup
    env = balanceStick()
    fig, (ax_env, ax_plot) = plt.subplots(2, 1)
    if show_plt:
        plt.ion()
        plt.show()

    # Model setup
    model = Agent(3, 5)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.MSELoss()
    q_model = Agent(3, 5)
    q_model.eval()

    # Hyperparams
    model_train_interval = 4
    q_model_train_interval = 100
    lambda_coef = .95
    max_epsilon, min_epsilon = 1, .3
    decay = .002
    batch_size = 32
    q_lr = .7

    epsilon = max_epsilon
    best_ticks = []
    task = None
    for epoch in range(1001):
        # CLI
        if epoch % cli_print_interval == 0:
            print(f'Epoch {epoch}\nEpsilon: {epsilon}')

        # Step
        max_ticks = 0
        replay_memory = []
        for step in range(q_model_train_interval):
            # Reset env
            state = env.reset()

            ticks = 0
            while np.sin(state[0][1]) > 0:
                # Ask model
                if np.random.rand() > epsilon:
                    logits = model(state)
                    logit_max = torch.argmax(logits, 1)
                else:
                    logit_max = np.random.randint(0, 5)
                actions = np.array([-200, -100, 0, 100, 200])
                force = actions[logit_max]

                # Tick
                old_state = state
                state = env.tick(force)
                replay_memory.append([old_state, logit_max, 1, state, False])
                ticks += 1

                # Render
                if not headless:
                    ax_env.cla()
                    env.render(ax=ax_env)
                    ax_env.title(f'Epoch {epoch}')
                    plt.pause(1e-10)
                    exit()

            # End memory
            if ticks > max_ticks:
                max_ticks = ticks
                max_tick_memory = []
                for transition in replay_memory[::-1]:
                    if transition[4]:
                        break
                    max_tick_memory.append(transition[0][0].detach())
                max_tick_memory = max_tick_memory[::-1]
            replay_memory[-1][-1] = True

            # Update main network
            if (step+1) % 4 == 0 and len(replay_memory) > batch_size * 2:
                mini_batch = random.sample(replay_memory, batch_size)
                current_states = torch.stack([transition[0][0] for transition in mini_batch], dim=0)
                new_states = torch.stack([transition[3][0] for transition in mini_batch], dim=0)
                current_qs = model(current_states)
                new_qs = q_model(new_states).detach()

                optimizer.zero_grad()
                loss = 0
                for i, (_, idx, reward, _, done) in enumerate(mini_batch):
                    current_q = current_qs[i]
                    new_q = new_qs[i]

                    if done:
                        max_q = reward
                    else:
                        max_q = reward + new_q.max()

                    desired_q = current_q.clone().detach()
                    desired_q[idx] = (1-q_lr) * desired_q[idx] + q_lr * max_q
                    loss = loss + criterion(current_q, desired_q)
                loss.backward()
                optimizer.step()

        # Update target network
        q_model.load_state_dict(model.state_dict())

        # Update params
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * epoch)

        # CLI
        if epoch % cli_print_interval == 0:
            print(f'Max Ticks: {max_ticks}\n')

        # Replay best
        if replay_best and epoch % replay_best_interval == 0:
            for state in max_tick_memory:
                ax_env.cla()
                env.render(ax=ax_env, state=state)
                ax_env.set_title(f'Epoch {epoch}')
                plt.pause(1/env.tickrate)

        # Best ticks plot
        best_ticks.append(max_ticks)
        if training_plot and epoch % training_plot_interval == 0:
            ax_plot.cla()
            ax_plot.plot(best_ticks, color='orange')
            ax_plot.set(xlabel='Epoch', ylabel='Best Ticks')
            plt.pause(1e-10)

    # while True:
    #     try:
    #         force = float(input())
    #     except:
    #         force = 0
    #         print('Incompatible input...')
    #     tick = env.tick(force)
    #     plt.cla()
    #     env.render(ax)
    #     plt.pause(1e-10)