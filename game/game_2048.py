# -*- coding: utf-8 -*-
import random
import numpy as np


class Game:

    def __init__(self, table_matrix_size=4):
        if table_matrix_size < 4:
            self.table_matrix_size = 4
        else:
            self.table_matrix_size = table_matrix_size
        # 生成初始化的Matrix（最小为4）
        self.table_matrix = self.create_matrix(self.table_matrix_size)
        # 初始化游戏分数
        self.score = 0

    # 随机游戏模式
    def random_game(self, game_time=1):
        for _ in range(game_time):
            while True:
                keyboard_signal = self.random_signal()
                print(keyboard_signal)
                if not self.move_episode(keyboard_signal=keyboard_signal):
                    break

    # 游戏场景管理
    def move_episode(self, keyboard_signal):
        # 这是2048的核心控件

        # 首先判断游戏是否可以继续进行
        if Game.is_game_over(self, Game.is_matrix_full(self.table_matrix)):
            return False

        # 如果可以进行游戏，则在空格处随机生成一个新的滑块
        self.table_matrix = Game.add_random_grid(self.table_matrix)

        # 根据信号对滑块进行移动
        self.table_matrix = self.update_matrix(self.table_matrix, keyboard_signal)

        return True

    @staticmethod
    def create_matrix(table_size):
        matrix = [[] for _ in range(table_size)]
        for i in range(table_size):
            row = []
            for j in range(table_size):
                row.append(0)
            matrix[i].extend(row)
        return matrix

    # 判断Matrix是否已经被填满
    @staticmethod
    def is_matrix_full(matrix):
        if 0 not in [x for item in matrix for x in item]:
            print("Ops! Game Over !!!")
            return True
        else:
            return False

    # 判断游戏是否已经结束
    def is_game_over(self, is_full_matrix):
        if is_full_matrix:
            print(np.matrix(self.table_matrix))
            self.table_matrix = self.create_matrix(self.table_matrix_size)
            # 显示本局游戏得分
            self.score = np.sum(self.table_matrix)
            self.score = 0
            return True
        else:
            return False

    # 在空位随机添加滑块
    @staticmethod
    def add_random_grid(matrix):
        while True:
            for row_num in range(len(matrix)):
                for item_num in range(len(matrix[row_num])):
                    if matrix[row_num][item_num] == 0 and \
                                    random.uniform(0, 1) > 0.95:
                        matrix[row_num][item_num] = 2
                        return matrix

    # 随机的做出移动信号
    @staticmethod
    def random_signal():
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
    @staticmethod
    def update_matrix(matrix, signal):
        # 先把Matrix中的全部滑块一到一侧
        matrix = Game.move_block(matrix, signal)
        # 做一次合并
        matrix = Game.merge_block(matrix, signal)
        # 再滑动一次，填补合并滑块时产生的空隙。
        return Game.move_block(matrix, signal)

    # 根据移动的方向，对滑块做出合并
    @staticmethod
    def merge_block(matrix, signal):
        matrix_size = len(matrix)

        # Move to Left
        if signal == "LEFT":
            # [8,2,2,2] - [8,4,0,2]
            for row_num in range(matrix_size):
                for col_num in range(matrix_size-1):
                    if matrix[row_num][col_num] == matrix[row_num][col_num+1]:
                        matrix[row_num][col_num] *= 2
                        matrix[row_num][col_num+1] = 0

        # Move to Right
        if signal == "RIGHT":
            for row_num in range(matrix_size):
                for col_num in range(matrix_size-1, 1, -1):
                    if matrix[row_num][col_num] == matrix[row_num][col_num-1]:
                        matrix[row_num][col_num] *= 2
                        matrix[row_num][col_num-1] = 0

        # Move Upside
        if signal == "UP":
            for col_num in range(matrix_size):
                for row_num in range(matrix_size-1):
                    if matrix[row_num][col_num] == matrix[row_num+1][col_num]:
                        matrix[row_num][col_num] *= 2
                        matrix[row_num+1][col_num] = 0

        # Move Downside
        if signal == "DOWN":
            for col_num in range(matrix_size):
                for row_num in range(matrix_size-1, 1, -1):
                    if matrix[row_num][col_num] == matrix[row_num-1][col_num]:
                        matrix[row_num][col_num] *= 2
                        matrix[row_num-1][col_num] = 0

        return matrix

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


if __name__ == '__main__':
    test_game = Game(table_matrix_size=4)
    test_game.random_game()

