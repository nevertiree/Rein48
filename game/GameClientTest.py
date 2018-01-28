# -*- coding: utf-8 -*-

from game.GameClient import *


class TestGameClient:
    test_game_client = Game()

    def test_is_matrix_full(self):
        # Test 1
        test_matrix = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        assert Game.is_matrix_full(test_matrix)

        # Test 2
        test_matrix = [
            [1, 1, 1, 1],
            [1, 1, 4, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        assert Game.is_matrix_full(test_matrix)

        # Test 3
        test_matrix = [
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        assert not Game.is_matrix_full(test_matrix)

    def test_is_game_over(self):
        test_matrix = [
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        assert not Game.has_game_over(test_matrix)

        test_matrix = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        assert Game.has_game_over(test_matrix)

        test_matrix = [
            [2, 4, 2, 4],
            [2, 4, 2, 4],
            [2, 4, 2, 4],
            [2, 4, 2, 4],
        ]
        assert not Game.has_game_over(test_matrix)

    def test_add_random_grid(self):
        pass

    def test_random_action(self):
        pass
