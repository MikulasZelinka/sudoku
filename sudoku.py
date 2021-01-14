# From: https://qqwing.com/
# Input sample:
# ......4....69....3..2.4..6.4.7.98.....82.3..4.2..7.1....97...3.......2.78..3.2.9.
#
# Solution:
#
#  9 8 1 | 5 3 6 | 4 7 2
#  5 4 6 | 9 2 7 | 8 1 3
#  7 3 2 | 8 4 1 | 9 6 5
# -------|-------|-------
#  4 5 7 | 1 9 8 | 3 2 6
#  1 9 8 | 2 6 3 | 7 5 4
#  6 2 3 | 4 7 5 | 1 8 9
# -------|-------|-------
#  2 6 9 | 7 1 4 | 5 3 8
#  3 1 5 | 6 8 9 | 2 4 7
#  8 7 4 | 3 5 2 | 6 9 1
#
# Number of Givens: 27
# Number of Singles: 73
# Number of Hidden Singles: 22
# Number of Naked Pairs: 2
# Number of Hidden Pairs: 0
# Number of Pointing Pairs/Triples: 3
# Number of Box/Line Intersections: 1
# Number of Guesses: 8
# Number of Backtracks: 8
# Difficulty: Expert

from functools import wraps
from time import time

import numpy as np
from numba import njit
from tqdm import tqdm


# https://stackoverflow.com/a/51503837/8971202
def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end = int(round(time() * 1000)) - start
            print(f'{func.__name__}({args}, {kwargs}) time measurement: {end if end > 0 else 0} ms')

    return _time_it


# https://stackoverflow.com/a/41578614/8971202
@njit
def index_of(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            yield idx


def to_numpy(sudoku_text):
    """
    sudoku_text expects 81 chars
    "Output format" to "One line", e.g.:
      ......4....69....3..2.4..6.4.7.98.....82.3..4.2..7.1....97...3.......2.78..3.2.9.
    """
    return np.array([int(x) for x in sudoku_text.replace('.', '0')], dtype=np.uint8).reshape((9, 9))


all_numbers = np.arange(1, 10, 1, dtype=np.uint8)


def solve(sudoku):
    if np.all(sudoku):
        print('done')
        print(sudoku)
        return

    x, y = next(index_of(sudoku, 0))
    x_left = 0 if x <= 2 else 3 if x <= 5 else 6
    y_left = 0 if y <= 2 else 3 if y <= 5 else 6
    forbidden_numbers = np.unique(
        np.concatenate((
            sudoku[:, y],
            sudoku[x, :],
            sudoku[x_left:x_left + 3, y_left:y_left + 3].ravel()
        ))
    )
    candidates = np.setdiff1d(all_numbers, forbidden_numbers)

    if len(candidates) == 0:
        return

    for c in candidates:
        sudoku[x, y] = c
        solve(sudoku)
        sudoku[x, y] = 0


@measure
def solve_wrapper(sudoku):
    solve(sudoku)


sudoku_text = '......4....69....3..2.4..6.4.7.98.....82.3..4.2..7.1....97...3.......2.78..3.2.9.'
sudoku_array = to_numpy(sudoku_text)

print(f'attempting to solve\n{sudoku_array}')
solve_wrapper(sudoku_array)

# https://codegolf.stackexchange.com/questions/190727/the-fastest-sudoku-solver
# "10000 easier Sudokus"
with open('hard_sudokus.txt') as f:
    sudokus = [x.strip() for x in f.readlines()[1:]]
    for sudoku in tqdm(sudokus):
        solve(to_numpy(sudoku))

# https://codegolf.stackexchange.com/questions/190727/the-fastest-sudoku-solver
with open('all_17_clue_sudokus.txt') as f:
    sudokus = [x.strip() for x in f.readlines()[1:]]
    for sudoku in tqdm(sudokus):
        solve(to_numpy(sudoku))
