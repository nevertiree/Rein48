# -*- coding: utf-8 -*-

import random


class Rand:

    @staticmethod
    def random_action(*args):
        action_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", }
        return action_dict[random.randint(0, 3)]
