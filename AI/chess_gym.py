import random
from typing import List
import gym
import numpy as np
import torch
from gym.spaces import Box
from ChessEngine.chess_game import Chess
from UI.chess_ui import UI


class ChessGym(gym.Env):
    def __init__(self, chess: Chess, ui: UI, action_space):
        self.chess = chess
        self.ui = ui
        self.action_space = action_space

        self.translateDict = {"__": 0,"wP":1, "wH":2, "wB":3, "wR": 4, "wQ":5, "wK":6,
                                      "bP":-1, "bH":-2, "bB":-3, "bR": -4, "bQ":-5, "bK":-6}

        self.observation_space = Box(low=min(self.translateDict.values()), high=max(self.translateDict.values()), shape=(8, 8), dtype=np.float16)

        print("Observation_space: ", self.observation_space)


    def step(self, actionId):
        """
            return observation, reward, done, info
        """
        action = self.action_space[actionId]

        currentColor = self.chess.GetCurrentColor()
        possibleMoves = self.chess.GetAvaliableMoves(currentColor)

        """
            Calculate reward
        """
        done = False
        reward = 0
        if self.ilegalInArow > 1000: # do not let the AI make more then 1000 ilegal in a row
            return self.getObservation(), -1000, True, {}
        if action in possibleMoves: # if move is allowed
            self.chess.MakeMove(action) # make action

            winner = self.chess.GetWinner() 
            if winner == 0: # if no winner
                reward = self.evaluateBoard(self.chess.GetCurrentColor()) # get basic score

            elif self.chess.GetWinner() == currentColor: # if we won
                reward = 100
                done = True
            else: # if we lost
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
        simpleEvaluation = 0
        if color == "w":
            myDead = sum([abs(self.translateDict[p])*-1 for p in self.chess.whiteDead]) # -6
            enemyDead = sum([abs(self.translateDict[p])*1 for p in self.chess.blackDead]) # +7
            simpleEvaluation = enemyDead + myDead # +7 + -6 = 1 
        if color == "b":
            myDead = sum([abs(self.translateDict[p])*-1 for p in self.chess.blackDead]) 
            enemyDead = sum([abs(self.translateDict[p])*1 for p in self.chess.whiteDead])
            simpleEvaluation = enemyDead + myDead

        simpleEvaluation = simpleEvaluation*10

        # calulcate length to the oposite side to give reward

        return simpleEvaluation

    def getObservation(self):
        return np.expand_dims(self.translateBoard(self.chess.board), axis=0)

    def translateBoard(self, board: List[List]):
        translated = []
        for row in board:
            newRow = []
            for elem in row:
                newRow.append(self.translateDict[elem])

            translated.append(newRow)

        return np.array(translated)

    def render(self):
        self.ui.update(self.chess)

        self.ui.root.update() # draw

