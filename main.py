# -*- coding: utf-8 -*-

import argparse

from game.GameClient import *

from control.rand import *
from control.hand import *


def play(game, control="hand", show_state=False, show_result=True):
    """Play the game. """

    """Choose a control strategy. """
    strategy_dict = {"rand": Rand.random_action, "hand": Hand.hand_control, }
    control_strategy = strategy_dict[control]

    is_game_over = False

    if control == "hand":
        show_state, show_result = True, True
        print("#####################################################\n"
              "    ---         ------           /|       /-------\  \n"
              "  /     \     /        \        / |      |         | \n"
              " |       |   |          |      /  |      |         | \n"
              "        /    |          |     /   |       \_______/  \n"
              "      /      |          |    /    |       /       \  \n"
              "    /        |          |   /_____|_____ |         | \n"
              "  /           \        /          |      |         | \n"
              " ---------      ------            |       \_______/  \n"
              "PLEASE INPUT [ACTION DIRECTION] TO PLAY THIS GAME.\n"
              "Left: [L] or [l] \n" "Right:[R] or [r] \n" "Up:   [U] or [u] \n" "Down: [D] or [d] \n"
              "#####################################################")

    """Until this episode is over."""
    while not is_game_over:
        """Print step state."""
        if show_state:
            Game.print_terminal(np.array(game.state_matrix))
        """Action and Feedback."""
        action = control_strategy(game.state_matrix)
        state_matrix, _, is_game_over = game.step(action)

    """Print result."""
    if show_result:
        Game.print_terminal(game.state_matrix)

    return np.sum(game.state_matrix)


def main():
    """Main Function. """

    """Attach arguments from terminal."""
    parser = argparse.ArgumentParser(description="Play terminal 2048...")
    # Control Strategy
    parser.add_argument('-c', '--control', type=str, dest='control', default='hand',
                        help='Auto-control or hand-control')
    # Print game state
    parser.add_argument('-v', '--visual', type=str, dest='visual', default='y')
    args = parser.parse_args()

    """Process input argument."""
    if args.control in ["rand", "Rand", "RAND", "r", "R"]:
        control = "rand"
    elif args.control in ["hand", "Hand", "HAND", "h", "H"]:
        control = "hand"
    else:
        control = "hand"

    visual = True if args.visual in ["Y", "y", 'Yes', "yes", "Yes"] else False

    """Init game."""
    game = Game()
    play(game=game, control=control, show_result=visual)


if __name__ == '__main__':
    main()
