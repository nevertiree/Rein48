# -*- coding: utf-8 -*-
import random
import numpy as np

MINI_BATCH_SIZE = 10


class Replay:

    def __init__(self, replay_size=100):
        self.max_size = replay_size
        self.cur_size = 0
        self.buffer = []

    def filled(self):
        return self.max_size <= self.cur_size

    def store(self, trans):
        if self.cur_size < self.max_size:
            self.cur_size += 1
            self.buffer.append(trans)

    def sample(self, batch_size=MINI_BATCH_SIZE):
        batch_list = Replay.sub_list(self.buffer, num=batch_size)
        batch_dict = Replay.list_2_dict(batch_list)
        self.clear()
        return batch_dict

    @staticmethod
    def sub_list(raw_list, num):
        if num > len(raw_list):
            return raw_list
        else:
            return random.sample(raw_list, num)

    @staticmethod
    def list_2_dict(raw_list):
        return {
            'state': np.array([x[0] for x in raw_list]),
            'action': np.array([x[1] for x in raw_list]),
            'reward': np.array([x[2] for x in raw_list]),
            'next_state': np.array([x[3] for x in raw_list])
        }

    def clear(self):
        self.cur_size = 0
        self.buffer = []
