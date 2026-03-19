from math import exp
from random import random, randint
from os.path import isfile
import matplotlib.pyplot as plt
import argparse

# VALIDATION OF COMMAND LINE ARGUMENTS
def validate_positive_int(data: int) -> int:
    if data > 0:
        return data
    else:
        raise TypeError("Only positive numbers can be used")
    
def validate_filepath(path: str) -> str:
    if isfile(path):
        return path
    else:
        raise TypeError(f"File {path} does not exist")

# SUDOKU FUNCTIONS
# Read sudoku as double matrix
def read_sudoku(path: str) -> tuple:
    S, C = [], []
    with open(path, 'r') as file:
        for line in file:
            data = [0 if x == 'x' else int(x) for x in line.strip().split()]
            constant = [x if x > 0 else 0 for x in data]
            S.append(data)
            C.append(constant)

    return S, C

# Print given sudoku
def print_sudoku(S: list) -> None:
    for line in S:
        for character in line:
            print(character, end=' ')
        print("")

# PROBABILITY
def schedule_prob(E: int, T: float) -> float:
    if E < 0: return 1
    return exp(-E / T)

# NEIGHBOUR SELECTION
def schedule_neighbour(S:list, C: list) -> tuple:
    x1 = x2 = y1 = y2 = 0

    while True:
        x1 = randint(0, 8)
        x2 = randint(0, 8)
        y1 = randint(0, 8)
        y2 = randint(0, 8)
        if C[x1][y1] == 0 and C[x2][y2] == 0 and S[x1][y1] != S[x2][y2]: break

    return x1, y1, x2, y2

# TEMPERATURE
def schedule_temp(T: float, a: float) -> float:
    return T * (1 - a)

# SIMULATED ANNEALING
# Check point incorrectness 
def is_point_incorrect(S: list, x: int, y: int) -> bool:
    value = S[x][y]

    for i in range(9):
        if i == y: continue
        if S[x][i] == value:
            return True
        
    for i in range(9):
        if i == x: continue
        if S[i][y] == value:
            return True
            
    cell_x = x // 3
    cell_y = y // 3

    for i in range(cell_x * 3, (cell_x + 1) * 3):
        for j in range(cell_y * 3, (cell_y + 1) * 3):
            if i == x and j == y: continue
            if S[i][j] == value:
                return True
            
    return False         

# Check point correctness
def is_point_correct(S: list, x: int, y: int) -> bool:
    return not is_point_incorrect(S, x, y)

# Count total errors in sudoku
def count_sudoku_errors(Sudoku: list) -> int:
    errors = 0
    for i in range(9):
        for j in range(9):
                errors += is_point_incorrect(Sudoku, i, j)
    return errors
    
# Energy difference when wanting to swap P1 with P2
def count_energy_difference(Sudoku: list, P1: tuple, P2: tuple) -> int:
    x1, y1 = P1
    x2, y2 = P2

    start_energy = count_sudoku_errors(Sudoku)

    Sudoku[x1][y1], Sudoku[x2][y2] = Sudoku[x2][y2],Sudoku[x1][y1]

    end_energy = count_sudoku_errors(Sudoku)

    Sudoku[x1][y1], Sudoku[x2][y2] = Sudoku[x2][y2],Sudoku[x1][y1]

    return end_energy - start_energy

# Making GIF is optional since it takes a lot of time
def simulated_annealing(path: str, max_iter: int, init_temp: int, a: float,
                        limit: int, make_gif: bool = False) -> None:
    xs, ys = [], []
    S, C = read_sudoku(path)

    # Naiwnie uzupełniam puste miejsca w każdym rzędzie brakującymi liczbami
    for i in range(9):
        present = [x for x in C[i] if x > 0]
        missing = [x for x in [i for i in range(1, 10)] if x not in present]
        iterator = 0
        for j in range(9):
            if S[i][j] == 0:
                S[i][j] = missing[iterator]
                iterator += 1

    current_energy = count_sudoku_errors(S)
    T = init_temp
    
    for i in range(1, max_iter + 1):
        T = schedule_temp(T, a)
        if T <= limit: break
        
        x1, y1, x2, y2 = schedule_neighbour(S, C)
    
        dE = count_energy_difference(S, (x1, y1), (x2, y2))    

        if (schedule_prob(dE, T) > random()):
            S[x1][y1], S[x2][y2] = S[x2][y2], S[x1][y1]
            current_energy = count_sudoku_errors(S) 

        ys.append(current_energy)
        xs.append(i)

        if current_energy <= 0: break   

    plt.figure()
    plt.title(f"Błąd końcowy dla {path} to {current_energy}")
    plt.xlabel('Iteracja')
    plt.ylabel('Wartość')
    plt.plot(xs, ys)
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    fig.tight_layout()

    ax.axis('off')
    ax.axis('tight')

    table = ax.table(S, loc='center')
    table.scale(1, 3)

    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(ha='center', va='center')

        if is_point_correct(S, row, col):
            cell.set_facecolor((0.5, 1, 0.5))
        else:
            cell.set_facecolor((1, 0.5, 0.5))

    plt.show()

# Argparse argument validation
def main() -> None:
    parser = argparse.ArgumentParser(description = "Generate sudoku solution with simulated annealing")

    parser.add_argument(
        "path",
        help = "Path to text file with initial sudoku state",
        type = str
    )

    parser.add_argument(
        "max_iterations",
        help = "Number of iterations for algorithm",
        type = int
    )

    parser.add_argument(
        "initial_temperature",
        help = "Initial temperature for the algorithm",
        type = int
    )

    parser.add_argument(
        "temperature_slope",
        help = "How fast should temperature decrease. The higher the number, the slower the fall",
        type = int
    )

    parser.add_argument(
        "limit",
        help = "n where 1e-n is lower limit for temperature",
        type = int
    )

    parser.add_argument(
        "--gif",
        help = "Make a GIF showcasing how the algorithm works",
        action = 'store_true'
    )

    args = parser.parse_args()

    path = validate_filepath(args.path)
    max_iter = validate_positive_int(args.max_iterations)
    init_temp = validate_positive_int(args.initial_temperature)
    slope = validate_positive_int(args.temperature_slope)
    limit = validate_positive_int(args.limit)
    make_gif = args.gif

    simulated_annealing(
        path,
        max_iter,
        init_temp,
        10 ** (-slope),
        10 ** (-limit),
        make_gif
    )

if __name__ == "__main__":
    main()
