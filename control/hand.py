# -*- coding: utf-8 -*-


class Hand:

    @staticmethod
    def hand_control(*args):
        print("Input action direction, then press ENTER button: ", end="")
        action = input()

        """Input checking. """
        while action not in ["UP", "Up", "U", "up", "u", 0,
                             "DOWN", "Down", "D", "down", "d", 1,
                             "LEFT", "Left", "L", "left", "l", 2,
                             "RIGHT", "Right", "R", "right", "r", 3]:
            print("\n##########[Error]########## \n"
                  "Input action signal is invalid, you must input valid value...\n"
                  "########################### \n")
            action = input()

        return action
