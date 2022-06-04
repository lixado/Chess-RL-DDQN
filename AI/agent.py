from collections import deque
import random
import sys
import numpy as np
import torch
from AI.model import DDQN

class AIPlayer:
    def __init__(self, action_space, save_dir):
        self.action_space_dim = len(action_space)
        self.save_dir = save_dir
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.net = DDQN(self.action_space_dim).to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999
        self.exploration_rate_min = 0.01
        self.curr_step = 0

        """
            Memory
        """
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.save_every = 5e5   # no. of experiences between saving Net

        """
            Q learning
        """
        self.gamma = 0.9 # discount factor
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.000250)
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99999985)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = 1e4 # min. experiences before training
        self.learn_every = 3 # no. of experiences between updates to Q_online
        self.sync_every = 1e3 # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
            Given a state, choose an epsilon-greedy action and update value of step.

            Inputs:
            state(LazyFrame): A single observation of the current state, dimension is (state_dim)
            Outputs:
            action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if (random.random() < self.exploration_rate):
            actionIdx = random.randint(0, self.action_space_dim-1)
        # EXPLOIT
        else:
            state = torch.tensor(state).float().to(device=self.device)
            state = state.unsqueeze(0)
            
            neuralNetOutput = self.net(state, model="online")
            actionIdx = torch.argmax(neuralNetOutput, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return actionIdx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = np.array(state)
        next_state = np.array(next_state)

        state = torch.tensor(state).float().to(device=self.device)
        next_state = torch.tensor(next_state).float().to(device=self.device)
        action = torch.tensor([action]).to(device=self.device)
        reward = torch.tensor([reward]).to(device=self.device)
        done = torch.tensor([done]).to(device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.net.target.load_state_dict(self.net.online.state_dict()) # sync_Q_target

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return 0, 0

        if self.curr_step % self.learn_every != 0:
            return 0, 0

        # Sample from memory get self.batch_size number of memories
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate, make predictions for the each memory
        td_est = self.td_estimate(state, action)

        # Get TD Target make predictions for next state of each memory
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #self.scheduler.step() # decrease lr overtime

        return loss.item()

    def td_estimate(self, state, action):
        """
            Output is batch_size number of rewards = Q_online(s,a) * 32
        """
        modelOutPut = self.net(state, model="online")
        current_Q = modelOutPut[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
            Output is batch_size number of Q*(s,a) = r + (1-done) * gamma * Q_target(s', argmax_a'( Q_online(s',a') ) )
        """
        next_state_Q = self.net(next_state, model="online") 
        best_action = torch.argmax(next_state_Q, axis=1) # argmax_a'( Q_online(s',a') ) 
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action] # Q_target(s', argmax_a'( Q_online(s',a') ) )
        return (reward + (1 - done.float()) * self.gamma * next_Q).float() # Q*(s,a)

    def loadModel(self, path):
        dt = torch.load(path, map_location=torch.device(self.device))
        self.net.load_state_dict(dt["model"])
        self.exploration_rate = dt["exploration_rate"]
        print(f"Loading model at {path} with exploration rate {self.exploration_rate}")

    def save(self):
        """
            Save the state to directory
        """
        save_path = (self.save_dir / f"agent_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"DDQN saved to {save_path} at step {self.curr_step}")
