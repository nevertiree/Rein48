# -*- coding: utf-8 -*-

"""
@author: Lance Wang
@github: https://github.com/nevertiree
@zhihu: https://www.zhihu.com/people/wang-ling-xiao-37-31
@license: Apache Licence
"""

import argparse
import copy
import random
import numpy as np


class Game:

    state_matrix, state_space_size = None, 0

    def __init__(self, table_matrix_size=4):
        """Attach game space size. """
        self.reward_space_size = 1
        self.action_space_size = 4

        if table_matrix_size < 4:
            self.state_space_size = 4
        else:
            self.state_space_size = table_matrix_size

        """Create a new chess table, and fill grids randomly. """
        self.reset()

    """ Public method for machine learning engine. """

    def reset(self, display=False):
        self.state_matrix = self.create_matrix(self.state_space_size)
        self.state_matrix = self.random_fill_grid(self.state_matrix)
        if display:
            Game.print_terminal(np.mat(self.state_matrix))
        return self.state_matrix

    def step(self, action):
        """Input the action signal ,and update game state to next step.
        :return: new game state, reward, death signal """

        """Update the game according to current state and action. """
        self.state_matrix, reward, has_changed = self.update_matrix(self.state_matrix, action)

        """Because this game is not over, we fill a grid randomly. """
        if has_changed:
            self.state_matrix = Game.random_fill_grid(self.state_matrix)

        return self.state_matrix, reward, Game.has_game_over(self.state_matrix)

    def play(self, strategy="hand", show_result=True):
        self.reset()
        is_game_over = False

        if strategy == "hand":
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

        while not is_game_over:
            if show_result:
                Game.print_terminal(np.array(self.state_matrix))

            if strategy == "rand":
                action = self.random_action()
            elif strategy == "hand":
                print("Input action direction, then press ENTER button: ", end="")
                action = input()
            else:
                break

            self.state_matrix, _, is_game_over = self.step(action)

        if show_result:
            Game.print_terminal(self.state_matrix)
        return np.sum(self.state_matrix)

    """ Private method for game logic. """

    @staticmethod
    def create_matrix(table_size=4):
        matrix = [[] for _ in range(table_size)]
        for i in range(table_size):
            row = []
            for j in range(table_size):
                row.append(0)
            matrix[i].extend(row)
        return matrix

    @staticmethod
    def has_game_over(game_matrix):
        """ Check whether this game is over.
        :return: has_game_over (Boolean) """

        """If game table isn't filled, this game isn't over. """
        if not Game.has_table_filled(game_matrix):
            return False
        else:
            length = len(game_matrix)
            """ This game can be continue , if some grid has an adjacent grid which has the same value with it, """
            for i in range(length):
                for j in range(length):
                    # Gird which not in the first row.
                    if i != 0:
                        if game_matrix[i][j] == game_matrix[i-1][j]:
                            return False
                    # Gird which not in the first column. .
                    if j != 0:
                        if game_matrix[i][j] == game_matrix[i][j-1]:
                            return False
                    # Gird which not in the last row.
                    if i != length - 1:
                        if game_matrix[i][j] == game_matrix[i+1][j]:
                            return False
                    # Gird which not in the last column. .
                    if j != length - 1:
                        if game_matrix[i][j] == game_matrix[i][j+1]:
                            return False
            return True

    @staticmethod
    def has_table_filled(game_matrix):
        """ Check whether the game table has been filled.
        :return: has_table_filled (boolean) """
        return 0 not in [x for item in game_matrix for x in item]

    @staticmethod
    def random_fill_grid(game_matrix):
        """ Fill blank grid in the game matrix.
        :param game_matrix:
        :return: new_game_matrix """

        """Attach blank grid index list"""
        blank_grid_index_list = []
        length = len(game_matrix)
        for i in range(length):
            for j in range(length):
                if game_matrix[i][j] == 0:
                    blank_grid_index_list.append((i, j))

        """Exit function, while game matrix is filled."""
        if not blank_grid_index_list:
            return game_matrix

        """Choose blank grid randomly. """
        random_grid_index = random.randint(0, len(blank_grid_index_list)-1)
        i, j = blank_grid_index_list[random_grid_index]

        """Fill chose blank grid with number 2 or 4. """
        game_matrix[i][j] = 2 if (random.uniform(0, 1) > 0.1) else 4

        return game_matrix

    @staticmethod
    def random_action():
        action_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", }
        return action_dict[random.randint(0, 3)]

    @staticmethod
    def update_matrix(matrix, action):
        """ Input the action signal ,and update game state to next step.
        :return: new_matrix, reward, has_changed """

        """If the action has effect (has_changed = True), 
        then the new matrix must be different from the origin one (origin_matrix != matrix) = True
        P.S. Must use Python deep copy module !!! """
        origin_matrix = copy.deepcopy(matrix)
        reward = 0

        if action in ["UP", "Up", "U", "up", "u", 0]:
            for col_num in range(len(matrix[0])):
                """i-pointer in the rear, while j-pointer in the frontier. """
                i, j = 0, 1
                """Iterate until j-pointer reached right side. """
                while j < len(matrix):
                    """j-pointer moves to find the next non-zero value."""
                    while j < len(matrix) and matrix[j][col_num] == 0:
                        j += 1

                    if j == len(matrix):
                        """Exit if index out of range."""
                        break
                    else:
                        """j-pointer finds an non-zero value. """

                        if matrix[i][col_num] == 0:
                            """[Switch] i-pointer has zero-value number, while j-point has non-zero number."""
                            """Assign j-pointer value to i-pointer, then set j-pointer value to 0."""
                            matrix[i][col_num] += matrix[j][col_num]
                            matrix[j][col_num] = 0

                        elif matrix[i][col_num] == matrix[j][col_num]:
                            """[Merge] i and j pointer have same non-zero number"""
                            """Double i-pointer number, than set j-pointer value to 0."""
                            matrix[i][col_num] += matrix[j][col_num]
                            matrix[j][col_num] = 0
                            i += 1

                        elif matrix[i][col_num] != matrix[j][col_num]:
                            """[Move] i and j pointer have different non-zero number"""
                            """Assign j-pointer value to the node behind i-pointer. """
                            """(if the node isn't j-pointer itself)"""
                            if i+1 != j:
                                matrix[i+1][col_num] += matrix[j][col_num]
                                matrix[j][col_num] = 0
                            i += 1

                        """Move j-pointer toward next node."""
                        j += 1
            return matrix, reward, (origin_matrix != matrix)

        if action in ["DOWN", "Down", "D", "down", "d", 1]:
            for col_num in range(len(matrix[0])):
                i, j = len(matrix)-1, len(matrix)-2
                while j >= 0:
                    while j >= 0 and matrix[j][col_num] == 0:
                        j -= 1
                    if j < 0:
                        break
                    else:
                        if matrix[i][col_num] == 0:
                            matrix[i][col_num] += matrix[j][col_num]
                            matrix[j][col_num] = 0
                        elif matrix[i][col_num] == matrix[j][col_num]:
                            matrix[i][col_num] += matrix[j][col_num]
                            matrix[j][col_num] = 0
                            i -= 1
                        elif matrix[i][col_num] != matrix[j][col_num]:
                            if i-1 != j:
                                matrix[i-1][col_num] += matrix[j][col_num]
                                matrix[j][col_num] = 0
                            i -= 1
                        j -= 1
            return matrix, reward, (origin_matrix != matrix)

        if action in ["LEFT", "Left", "L", "left", "l", 2]:
            for col_num in range(len(matrix)):
                i, j = 0, 1
                while j < len(matrix[col_num]):
                    while j < len(matrix[col_num]) and matrix[col_num][j] == 0:
                        j += 1
                    if j == len(matrix[col_num]):
                        break
                    else:
                        if matrix[col_num][i] == 0:
                            matrix[col_num][i] += matrix[col_num][j]
                            matrix[col_num][j] = 0
                        elif matrix[col_num][i] == matrix[col_num][j]:
                            matrix[col_num][i] += matrix[col_num][j]
                            matrix[col_num][j] = 0
                            i += 1
                        elif matrix[col_num][i] != matrix[col_num][j]:
                            if i+1 != j:
                                matrix[col_num][i+1] += matrix[col_num][j]
                                matrix[col_num][j] = 0
                            i += 1
                        j += 1
            return matrix, reward, (origin_matrix != matrix)

        if action in ["RIGHT", "Right", "R", "right", "r", 3]:
            for col_num in range(len(matrix)):
                i, j = len(matrix[col_num])-1, len(matrix[col_num])-2
                while j >= 0:
                    while j > 0 and matrix[col_num][j] == 0:
                        j -= 1
                    if j == -1:
                        break
                    else:
                        if matrix[col_num][i] == 0:
                            matrix[col_num][i] += matrix[col_num][j]
                            matrix[col_num][j] = 0
                        elif matrix[col_num][i] == matrix[col_num][j]:
                            matrix[col_num][i] += matrix[col_num][j]
                            matrix[col_num][j] = 0
                            i -= 1
                        elif matrix[col_num][i] != matrix[col_num][j]:
                            if i-1 != j:
                                matrix[col_num][i-1] += matrix[col_num][j]
                                matrix[col_num][j] = 0
                            i -= 1
                        j -= 1
            return matrix, reward, (origin_matrix != matrix)

        raise ValueError("Input action signal is wrong. ")

    @staticmethod
    def print_terminal(matrix):
        width, height = len(matrix[0]), len(matrix)
        print("-" * (1 + 7 * width))
        for i in range(height):
            print("|", end="")
            for j in range(width):
                if matrix[i][j] != 0:
                    print(str(matrix[i][j]).center(6), end="")
                else:
                    print(" " * 6, end="")
                print("|", end="")
            print("\n", end="")
            print("-" * (1 + 7 * width))


def main():
    parser = argparse.ArgumentParser(description="Play terminal 2048...")
    parser.add_argument('-t', '--type', dest='type', type=str, default='hand',
                        help='Auto-control or hand-control')

    args = parser.parse_args()

    game = Game()
    game.play(strategy=args.type)


if __name__ == '__main__':
    main()
