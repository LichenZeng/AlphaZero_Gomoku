# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game_M import Board, Game
from mcts_pure_M import MCTSPlayer as MCTS_Pure
from mcts_alphaZero_M import MCTSPlayer
from policy_value_net_pytorch_M import PolicyValueNet  # Pytorch


# from fiveinrow import fiveinrow
# from fiveinrow.fiveinrow import GRID_WIDTH, WHITE, BLACK


class Human(object):
    """
    human player
    """

    def __init__(self, display):
        self.player = None
        self.display = display

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            # location = input("Your move: ")
            # if isinstance(location, str):  # for python3
            #     location = [int(n, 10) for n in location.split(",")]
            print("It is your turn:")
            location = self.display.get_coordinate()
            print("\tYour move: {}\n".format(location))
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 8, 8
    model_file = 'current_policy.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        # display = fiveinrow.GobangState()
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow
        best_policy = PolicyValueNet(width, height, model_file=model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
