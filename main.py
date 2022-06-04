import datetime
import sys
from time import sleep
from AI.agent import AIPlayer
from AI.chess_gym import ChessGym
from AI.logger import Logger
from ChessEngine.chess_game import Chess, Move
from UI.chess_ui import UI
from gym.wrappers import NormalizeObservation, NormalizeReward
from pathlib import Path

if __name__ == '__main__':
    """
        Choose mode
    """
    modes = ["P vs P", "P vs white AI", "P vs black AI", "Train", "Train UI"]
    for cnt, modeName in enumerate(modes, 1):
        sys.stdout.write("[%d] %s\n\r" % (cnt, modeName))

    mode = int(input("Select mode[1-%s]: " % cnt)) - 1

    """
        Start board
    """
    chess = Chess()

    if mode >= 3: # Train AI
        ui = UI(chess.board) if mode == 4 else None

        """
            Variables
        """
        episodes = 1000
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = Path("checkpoints") / now

        action_space = []
        for y in range(8):
            for x in range(8):
                for yNext in range(8):
                    for xNext in range(8):
                        action_space.append(Move([y,x], [yNext, xNext]))

        print("Action space size: ", len(action_space)) # 8^4 = 8*8*8*8 = 4096

        env = ChessGym(chess, ui, action_space) # start Enviroment 

        env = NormalizeObservation(env)  # normalize the observation values
        env = NormalizeReward(env)  # normalize reward values

        saveDirPlayer1 = save_dir / 'player1'
        aiPlayer1 = AIPlayer(action_space, saveDirPlayer1) # start AI
        loggerPlayer1 = Logger(saveDirPlayer1)

        saveDirPlayer2 = save_dir / 'player2'
        aiPlayer2 = AIPlayer(action_space, saveDirPlayer2) # start AI
        loggerPlayer2 = Logger(saveDirPlayer2)

        whiteWin = 0
        blackWin = 0
        for e in range(episodes):
            observation = env.reset() # reset game
            if e == 0:
                print("Observation: ", observation)

            while True: # while playing
                if mode == 4:
                    env.render()

                """
                    Player 1 makes move
                """
                actionId = aiPlayer1.act(observation)
                next_observation, reward, done, info = env.step(actionId)

                # Remember
                aiPlayer1.cache(observation, next_observation, actionId, reward, done)
                # Learn
                q, loss = aiPlayer1.learn()
                # Log
                loggerPlayer1.log_step(reward, loss, q)
                # Update state
                observation = next_observation

                """
                    Player 2 makes move
                """
                actionId = aiPlayer2.act(observation)
                next_observation, reward, done, info = env.step(actionId)

                # Remember
                aiPlayer2.cache(observation, next_observation, actionId, reward, done)
                # Learn
                q, loss = aiPlayer2.learn()
                # Log
                loggerPlayer2.log_step(reward, loss, q)
                # Update state
                observation = next_observation


                if loggerPlayer1.steps % 500 == 0:
                    print(".", end="")

                if done or loggerPlayer1.steps > 10000:
                    if chess.GetWinner() == "w":
                        whiteWin += 1
                    elif chess.GetWinner() == "b":
                        blackWin += 1
                    break # stop the game and reset

            # Episode is done
            loggerPlayer1.log_episode(e, aiPlayer1.exploration_rate, whiteWin, blackWin)
            loggerPlayer2.log_episode(e, aiPlayer2.exploration_rate, blackWin, whiteWin)

        # when finished training save model
        aiPlayer1.save()
        aiPlayer2.save()

    if mode == 0: # P vs P
        """
            UI
        """
        ui = UI(chess.board)
        
        """
            Start game
        """
        move = None
        whitePreviousMoves = []
        blackPreviousMoves = []
        possibleMoves = []

        chess.PrintBoard()
        ui.root.update() # draw
        possibleMoves = chess.GetCurrentTurnPossibleMoves()
        print("Possible Moves: ", possibleMoves)
        while True:
            if ui.done:
                exit() # exit program if wanted to

            ui.root.update() # draw UI

            """
                Check winner
            """
            if chess.GetWinner() == 'w':
                ui.winner("White")
            if chess.GetWinner() == 'b':
                ui.winner("Black")

            """
                Get input 
            """
            # Handle if first move is ilegal
            if len(ui.move) == 1 and ui.loading == False:
                possibleNextPositions = []
                # check if input is possible
                for element in possibleMoves:
                    # this first click is possible
                    if ui.move[0][0] == element.oldPosition[0] and ui.move[0][1] == element.oldPosition[1]:
                        possibleNextPositions.append(element.newPosition)

                if len(possibleNextPositions) > 0: # if it is a legal move
                    ui.drawPossible(possibleNextPositions)
                else:
                    ui.resetClick()


            if len(ui.move) == 2 and ui.loading == False: # get input from UI
                move = Move(ui.move[0], ui.move[1])
                ui.resetClick()

            if (move is not None): # a made was made
                print(f'Move: {str(move)}') 

                if (move in possibleMoves): # if it is allowed
                    chess.MakeMove(move)
                    if chess.currentTurn % 2 == 0: # white turn
                        whitePreviousMoves.append(move)
                    else: # black turn
                        blackPreviousMoves.append(move)

                    """
                    Update visuals
                    """
                    possibleMoves = chess.GetCurrentTurnPossibleMoves()
                    ui.update(chess) # update buttons
                    print("Black Dead: ", chess.blackDead)
                    print("White Dead: ", chess.whiteDead)
                    chess.PrintBoard()
                    print("Possible Moves: ", possibleMoves)

                    move = None # reset move
                else: # if not allowed
                    print("NOT ALLOWED!!!!!")
                    move = None # reset move
