from __future__ import division, absolute_import, print_function


class Action(object):
    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other

    def __gt__(self, other):
        return self.index > other
