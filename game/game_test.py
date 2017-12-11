# -*- coding: utf-8 -*-

from game.game_cli import *

if __name__ == '__main__':
    new_game = Game()
    SCORE = []

    for _ in range(10):
        SCORE.append(new_game.play(show_result=True))

    print(SCORE)
    print(min(SCORE))
    print(max(SCORE))
    print(sum(SCORE)/len(SCORE))
