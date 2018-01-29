# -*- coding: utf-8 -*-
import random
import numpy as np


class Game:

    state_matrix, state_space_size = None, 0

    ACTION_SPACE = ["UP", "DOWN", "RIGHT", "LEFT"]

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

    """
    Public method for machine learning engine.
     """

    def reset(self, display=False):
        self.state_matrix = self.create_matrix(self.state_space_size)
        self.state_matrix = self.random_fill_grid(self.state_matrix)
        if display:
            print(np.mat(self.state_matrix))
        return self.state_matrix

    """Input the action signal ,and update game state to next step. """
    def step(self, action):
        """ Return: new game state, reward, death signal """

        """Marks whether this game can continue. """
        is_dead = False
        has_change = True

        """Check if the game is over. """
        if Game.has_game_over(self.state_matrix):
            is_dead = True
            return self.state_matrix, 0, is_dead

        """Because this game is not over, we fill a grid randomly. """
        if has_change:
            self.state_matrix = Game.random_fill_grid(self.state_matrix)

        """Update the game according to current state and action. """
        self.state_matrix, reward, has_change = self.update_matrix(self.state_matrix, action)

        return self.state_matrix, reward, is_dead

    def play(self, strategy="RAND", show_result=False):
        self.reset()
        is_game_over = False

        # 完成一局游戏
        while not is_game_over:
            print(np.array(self.state_matrix))
            if strategy == "RAND":
                action = self.random_action()
            else:
                action = input()

            self.state_matrix, _, is_game_over = self.step(action)

        # 游戏结束
        return np.sum(self.state_matrix)

    """
    Private method for game logic.
    """

    @staticmethod
    def create_matrix(table_size=4):
        # 生成一个空的棋盘，这个实现比较蠢
        matrix = [[] for _ in range(table_size)]
        for i in range(table_size):
            row = []
            for j in range(table_size):
                row.append(0)
            matrix[i].extend(row)
        return matrix

    """Check whether this game is over. """
    @staticmethod
    def has_game_over(game_matrix):
        """If game table isn't filled, this game isn't over. """
        if not Game.has_table_filled(game_matrix):
            return False
        else:
            """ This game can be continue , if some grid has an adjacent grid which has the same value with it, """

            length = len(game_matrix)
            """ We must find the grid-tuple. """
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

    """Check whether the game table has been filled. """
    @staticmethod
    def has_table_filled(game_matrix): return 0 not in [x for item in game_matrix for x in item]

    """Fill blank grid in the game matrix. """
    @staticmethod
    def random_fill_grid(game_matrix):
        """Attach blank grid index list"""
        blank_grid_index_list = []
        length = len(game_matrix)
        for i in range(length):
            for j in range(length):
                if game_matrix[i][j] == 0:
                    blank_grid_index_list.append((i, j))

        # If game matrix is filled, then end this function.
        if not blank_grid_index_list:
            return game_matrix

        """Choose a blank grid randomly. """
        blank_gird_list_length = len(blank_grid_index_list)
        random_grid_index = random.randint(0, blank_gird_list_length-1)
        i, j = blank_grid_index_list[random_grid_index]

        """Fill this grid with number 2 or 4. """
        if random.uniform(0, 1) > 0.5:
            game_matrix[i][j] = 2
        else:
            game_matrix[i][j] = 4

        return game_matrix

    # 随机的做出移动
    @staticmethod
    def random_action():
        rand_num = random.uniform(0, 1)
        # Up Down Left Right :  0.25 0.25 0.25 0.25
        if rand_num <= 0.25:
            return "UP"
        elif rand_num <= 0.5:
            return "DOWN"
        elif rand_num <= 0.75:
            return "LEFT"
        else:
            return "RIGHT"

    """Input the action signal ,and update game state to next step. """
    @staticmethod
    def update_matrix(matrix, action):
        """ Return: new_matrix, reward, has_changed"""
        backup_matrix = matrix
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

            return matrix, reward, backup_matrix != matrix

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
            return matrix, reward, backup_matrix != matrix

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
            return matrix, reward, backup_matrix != matrix

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
            return matrix, reward, backup_matrix != matrix


if __name__ == '__main__':
    game = Game()
    game.play(strategy="H")
