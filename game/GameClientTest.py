# -*- coding: utf-8 -*-

import numpy as np
from game.GameClient import *


class TestGameClient:
    test_game_client = Game()

    def test_has_matrix_filled(self):
        # Test 1
        test_matrix = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], ]
        assert Game.has_table_filled(test_matrix)

        # Test 2
        test_matrix = [[1, 1, 1, 1], [1, 1, 4, 1], [1, 1, 1, 1], [1, 1, 1, 1], ]
        assert Game.has_table_filled(test_matrix)

        # Test 3
        test_matrix = [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], ]
        assert not Game.has_table_filled(test_matrix)

    def test_is_game_over(self):
        test_matrix = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], ]
        assert not Game.has_game_over(test_matrix)

        test_matrix = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2], ]
        assert Game.has_game_over(test_matrix)

        test_matrix = [[2, 4, 2, 4], [2, 4, 2, 4], [2, 4, 2, 4], [2, 4, 2, 4], ]
        assert not Game.has_game_over(test_matrix)

    def test_random_fill_grid(self):
        test_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], ]
        assert np.sum(Game.random_fill_grid(test_matrix)) - np.sum(test_matrix) == 2 or 4

        test_matrix = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], ]
        assert np.sum(Game.random_fill_grid(test_matrix)) - np.sum(test_matrix) == 2 or 4

        test_matrix = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 0], ]
        assert np.sum(Game.random_fill_grid(test_matrix)) - np.sum(test_matrix) == 2 or 4

        test_matrix = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2], ]
        assert np.sum(Game.random_fill_grid(test_matrix)) - np.sum(test_matrix) == 0

    def test_random_action(self):
        pass

    def test_update_matrix_up(self):
        # Test 01
        test_matrix = [[0], [0], [1], [0]]
        except_matrix = [[1], [0], [0], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 02
        test_matrix = [[1], [0], [1], [0]]
        except_matrix = [[2], [0], [0], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 03
        test_matrix = [[2], [0], [1], [0]]
        except_matrix = [[2], [1], [0], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 04
        test_matrix = [[2], [2], [1], [0]]
        except_matrix = [[4], [1], [0], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 05
        test_matrix = [[2], [2], [2], [2]]
        except_matrix = [[4], [4], [0], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 06
        test_matrix = [[8], [8], [4], [0]]
        except_matrix = [[16], [4], [0], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 07
        test_matrix = [[8], [4], [4], [4]]
        except_matrix = [[8], [8], [4], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 08
        test_matrix = [[2], [0], [0], [2]]
        except_matrix = [[4], [0], [0], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 09
        test_matrix = [[0], [4], [2], [2]]
        except_matrix = [[4], [4], [0], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 10
        test_matrix = [[8], [8], [8], [0]]
        except_matrix = [[16], [8], [0], [0]]
        actual_matrix, _ = Game.update_matrix(test_matrix, "U")
        print(actual_matrix)
        assert except_matrix == actual_matrix

    def test_update_matrix_down(self):
        # Test 01
        test_matrix = [[0], [0], [1], [0]]
        except_matrix = [[0], [0], [0], [1]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 02
        test_matrix = [[1], [0], [1], [0]]
        except_matrix = [[0], [0], [0], [2]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 03
        test_matrix = [[2], [0], [1], [0]]
        except_matrix = [[0], [0], [2], [1]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 04
        test_matrix = [[2], [2], [1], [0]]
        except_matrix = [[0], [0], [4], [1]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 05
        test_matrix = [[2], [2], [2], [2]]
        except_matrix = [[0], [0], [4], [4]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 06
        test_matrix = [[8], [8], [4], [0]]
        except_matrix = [[0], [0], [16], [4]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 07
        test_matrix = [[8], [4], [4], [4]]
        except_matrix = [[0], [8], [4], [8]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 08
        test_matrix = [[2], [0], [0], [2]]
        except_matrix = [[0], [0], [0], [4]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 09
        test_matrix = [[0], [4], [2], [2]]
        except_matrix = [[0], [0], [4], [4]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 10
        test_matrix = [[8], [8], [8], [0]]
        except_matrix = [[0], [0], [8], [16]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "D")
        print(actual_matrix)
        assert except_matrix == actual_matrix

    def test_update_matrix_left(self):
        # Test 01
        test_matrix = [[0, 0, 1, 0]]
        except_matrix = [[1, 0, 0, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 02
        test_matrix = [[1, 0, 1, 0]]
        except_matrix = [[2, 0, 0, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 03
        test_matrix = [[2, 0, 1, 0]]
        except_matrix = [[2, 1, 0, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 04
        test_matrix = [[2, 2, 1, 0]]
        except_matrix = [[4, 1, 0, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 05
        test_matrix = [[2, 2, 2, 2]]
        except_matrix = [[4, 4, 0, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 06
        test_matrix = [[8, 8, 4, 0]]
        except_matrix = [[16, 4, 0, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 07
        test_matrix = [[8, 4, 4, 4]]
        except_matrix = [[8, 8, 4, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 08
        test_matrix = [[2, 0, 0, 2]]
        except_matrix = [[4, 0, 0, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 09
        test_matrix = [[0, 4, 2, 2]]
        except_matrix = [[4, 4, 0, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 10
        test_matrix = [[8, 8, 8, 0]]
        except_matrix = [[16, 8, 0, 0]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "LEFT")
        print(actual_matrix)
        assert except_matrix == actual_matrix

    def test_update_matrix_right(self):
        # Test 01
        test_matrix = [[0, 0, 1, 0]]
        except_matrix = [[0, 0, 0, 1]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 02
        test_matrix = [[1, 0, 1, 0]]
        except_matrix = [[0, 0, 0, 2]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 03
        test_matrix = [[2, 0, 1, 0]]
        except_matrix = [[0, 0, 2, 1]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 04
        test_matrix = [[2, 2, 1, 0]]
        except_matrix = [[0, 0, 4, 1]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 05
        test_matrix = [[2, 2, 2, 2]]
        except_matrix = [[0, 0, 4, 4]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 06
        test_matrix = [[8, 8, 4, 0]]
        except_matrix = [[0, 0, 16, 4]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 07
        test_matrix = [[8, 4, 4, 4]]
        except_matrix = [[0, 8, 4, 8]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 08
        test_matrix = [[2, 0, 0, 2]]
        except_matrix = [[0, 0, 0, 4]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 09
        test_matrix = [[0, 4, 2, 2]]
        except_matrix = [[0, 0, 4, 4]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

        # Test 10
        test_matrix = [[8, 8, 8, 0]]
        except_matrix = [[0, 0, 8, 16]]
        actual_matrix, _, _ = Game.update_matrix(test_matrix, "R")
        print(actual_matrix)
        assert except_matrix == actual_matrix

    def test_print_terminal(self):
        test_matrix = [[4, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [8, 0, 0, 0]]
        Game.print_terminal(test_matrix)


if __name__ == '__main__':
    tester = TestGameClient()
    tester.test_print_terminal()
