import random
from typing import List
import gym
import numpy as np
import torch
from gym.spaces import Box
from ChessEngine.chess_game import Chess
from UI.chess_ui import UI


class ChessGym(gym.Env):
    def __init__(self, chess: Chess, action_space):
        self.chess = chess
        self.action_space = action_space

        self.translateDict = {"__": 0,"wP":1, "wH":2, "wB":3, "wR": 4, "wQ":5, "wK":6,
                                      "bP":-1, "bH":-2, "bB":-3, "bR": -4, "bQ":-5, "bK":-6}

        self.observation_space = Box(low=min(self.translateDict.values()), high=max(self.translateDict.values()), shape=(8, 8), dtype=np.float16)

        print("Observation_space: ", self.observation_space)
        self.ilegalInArow = 0


    def step(self, actionId, color):
        """
            return observation, reward, done, info
        """
        action = self.action_space[actionId]

        possibleMoves = self.chess.GetCurrentTurnPossibleMoves()

        """
            Calculate reward
        """
        done = False
        reward = 0
        if self.ilegalInArow > 1000: # do not let the game go for to long
            return self.getObservation(), -1000, True, {}
        if action in possibleMoves: # if move is allowed
            self.chess.MakeMove(action) # make action

            winner = self.chess.GetWinner() 
            if winner == 0: # if no winner
                reward = self.evaluateBoard(color)*10 # get basic score

                # tick
                # make next move using randomeness to test
                possibleMoves = self.chess.GetCurrentTurnPossibleMoves()
                self.chess.MakeMove(random.choice(possibleMoves))

            elif self.chess.GetWinner() == color: # if we won
                reward = 100
                done = True
            elif self.chess.GetWinner() != color: # if we lost
                reward = -100
                done = True

            self.ilegalInArow = 0
        else: # if move not allowed
            reward = -50
            self.ilegalInArow +=1

        return self.getObservation(), reward, done, {}

    def reset(self):
        """
            return observation
        """
        self.ilegalInArow = 0
        self.chess.ResetAndFillBoard()
        return self.getObservation()

    def evaluateBoard(self, color):
        # calculate simple sum of how many alive and dead
        if color == "w":
            myDead = sum([abs(self.translateDict[p])*-1 for p in self.chess.whiteDead]) # -6
            enemyDead = sum([abs(self.translateDict[p])*1 for p in self.chess.blackDead]) # +7
            return enemyDead + myDead # +7 + -6 = 1 
        if color == "b":
            myDead = sum([abs(self.translateDict[p])*-1 for p in self.chess.blackDead]) 
            enemyDead = sum([abs(self.translateDict[p])*1 for p in self.chess.whiteDead])
            return enemyDead + myDead

    def getObservation(self):
        return self.translateBoard(self.chess.board).unsqueeze(0)

    def translateBoard(self, board: List[List]):
        translated = []
        for row in board:
            newRow = []
            for elem in row:
                newRow.append(self.translateDict[elem])

            translated.append(newRow)

        return torch.FloatTensor(translated)

    def render(self, ui: UI):
        ui.update(self.chess)

        ui.root.update() # draw

