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

    def reset(self):
        self.state_matrix = self.create_matrix(self.state_space_size)
        self.state_matrix = self.random_fill_grid(self.state_matrix)
        return self.state_matrix

    """Input the action signal ,and update game state to next step. """
    def step(self, action):
        """ Return: new game state, reward, death signal """

        """Marks whether this game can continue. """
        is_dead = False

        """Check if the game is over. """
        if Game.has_game_over(self.state_matrix):
            is_dead = True
            return self.state_matrix, 0, is_dead

        """Because this game is not over, we fill a grid randomly. """
        self.state_matrix = Game.random_fill_grid(self.state_matrix)

        """Update the game according to current state and action. """
        self.state_matrix, reward = self.update_matrix(self.state_matrix, action)

        return self.state_matrix, reward, is_dead

    def play(self, strategy="RAND", show_result=False):
        self.reset()
        is_game_over = False

        # 完成一局游戏
        while not is_game_over:
            if strategy == "RAND":
                action = self.random_action()
            else:
                action = self.random_action()

            self.state_matrix, _, is_game_over = self.step(action)

        # 展示最终结果
        if show_result:
            print(np.array(self.state_matrix))

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
        if not Game.has_matrix_filled(game_matrix):
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
    def has_matrix_filled(game_matrix): return 0 not in [x for item in game_matrix for x in item]

    """Fill blank grid in the game matrix. """
    @staticmethod
    def random_fill_grid(game_matrix):
        """Attach blank grid index list"""
        blank_grid_index_list = []
        length = len(game_matrix)
        for i in range(length):
            for j in range(length):
                if game_matrix[i][j] != 0:
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

    # 根据信号完成Matrix的更新 [8,2,0,2] -> [8,2,2,0] -> [8,4,0,0]
    # 返回新的State和Reward
    @staticmethod
    def update_matrix(matrix, signal):
        # 先把Matrix中的全部滑块一到一侧
        matrix = Game.move_block(matrix, signal)
        # 做一次合并
        matrix, reward = Game.merge_block(matrix, signal)
        # 再滑动一次，填补合并滑块时产生的空隙。
        matrix = Game.move_block(matrix, signal)
        return matrix, reward

    # 根据移动的方向，对滑块做出合并
    @staticmethod
    def merge_block(matrix, action):
        matrix_size = len(matrix)
        reward_block_list = []

        if action == "LEFT" or action == 0:
            # [8,2,2,2] - [8,4,0,2]
            for row_num in range(matrix_size):
                for col_num in range(matrix_size-1):
                    if matrix[row_num][col_num] == matrix[row_num][col_num+1]:
                        # 把这个Block的值记录下来
                        reward_block_list.append(matrix[row_num][col_num])
                        matrix[row_num][col_num] *= 2
                        matrix[row_num][col_num+1] = 0

        if action == "RIGHT" or action == 1:
            for row_num in range(matrix_size):
                for col_num in range(matrix_size-1, 1, -1):
                    if matrix[row_num][col_num] == matrix[row_num][col_num-1]:
                        # 把这个Block的值记录下来
                        reward_block_list.append(matrix[row_num][col_num])
                        matrix[row_num][col_num] *= 2
                        matrix[row_num][col_num-1] = 0

        if action == "UP" or action == 2:
            for col_num in range(matrix_size):
                for row_num in range(matrix_size-1):
                    if matrix[row_num][col_num] == matrix[row_num+1][col_num]:
                        # 把这个Block的值记录下来
                        reward_block_list.append(matrix[row_num][col_num])
                        matrix[row_num][col_num] *= 2
                        matrix[row_num+1][col_num] = 0

        if action == "DOWN" or action == 3:
            for col_num in range(matrix_size):
                for row_num in range(matrix_size-1, 1, -1):
                    if matrix[row_num][col_num] == matrix[row_num-1][col_num]:
                        # 把这个Block的值记录下来
                        reward_block_list.append(matrix[row_num][col_num])
                        matrix[row_num][col_num] *= 2
                        matrix[row_num-1][col_num] = 0

        reward = sum(reward_block_list)
        return matrix, reward

    # 把滑块移动到一侧，使用双指针算法
    @staticmethod
    def move_block(matrix, signal):

        matrix_size = len(matrix)

        # 水平（左右）的处理
        if signal == "RIGHT" or signal == "LEFT":
            for row_num in range(matrix_size):
                i, j = 0, 0
                while i < matrix_size and j < matrix_size:
                    # i和j同时向前移动，寻找Zero-Value
                    while i < matrix_size and j < matrix_size and matrix[row_num][i] != 0:
                        i, j = i+1, j+1
                    # i指针遇到了Zero-Value，所以停下来；j指针继续向前走。
                    while j < matrix_size and matrix[row_num][j] == 0:
                        j += 1
                    # j指针在Non-Zero-Value处停下，i和j互换（前提是两者的位置都是合法的）
                    if j < matrix_size and matrix[row_num][j] != 0:
                        matrix[row_num][i], matrix[row_num][j] = matrix[row_num][j], 0
                        i, j = i+1, j+1

                if signal == "RIGHT":
                    matrix[row_num] = matrix[row_num][::-1]

        if signal == "UP":
            for col_num in range(matrix_size):
                i, j = 0, 0
                while i < matrix_size and j < matrix_size:
                    # i和j同时向下移动，寻找Zero-Value
                    while i < matrix_size and j < matrix_size and matrix[i][col_num] != 0:
                        i, j = i+1, j+1
                    # i指针遇到了Zero-Value，所以停下了；j指针继续向下寻找Non-Zero-Value
                    while j < matrix_size and matrix[j][col_num] == 0:
                        j += 1
                    # j指针在Non-Zero-Value处停下，i和j互换（前提是两者的位置都是合法的）
                    if j < matrix_size and matrix[j][col_num] != 0:
                        matrix[i][col_num], matrix[j][col_num] = matrix[j][col_num], 0
                        i, j = i+1, j+1

        if signal == "DOWN":
            for col_num in range(matrix_size):
                i, j = 0, 1
                while i < matrix_size and j < matrix_size:
                    while i < matrix_size and j < matrix_size and matrix[matrix_size-1-i][col_num] != 0:
                        i, j = i+1, j+1
                    while j < matrix_size and matrix[matrix_size-1-j][col_num] == 0:
                        j += 1
                    if j < matrix_size and matrix[matrix_size-1-j][col_num] != 0:
                        matrix[matrix_size-1-i][col_num], matrix[matrix_size-1-j][col_num] = \
                            matrix[matrix_size-1-j][col_num], 0
                        i, j = i+1, j+1

        return matrix
