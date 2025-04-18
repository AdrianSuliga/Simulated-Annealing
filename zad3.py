from math import exp
from random import random, randint
import matplotlib.pyplot as plt

def read_sudoku(path):
    S, C = [], []
    with open(path, 'r') as file:
        for line in file:
            data = [0 if x == 'x' else int(x) for x in line.strip().split()]
            constant = [x if x > 0 else 0 for x in data]
            S.append(data)
            C.append(constant)

    return S, C

def print_sudoku(S):
    for line in S:
        for character in line:
            print(character, end=' ')
        print("")

def schedule_prob(E, T):
    if E < 0: return 1
    return exp(-E / T)

def schedule_neighbour(S, C):
    x1 = x2 = y1 = y2 = 0

    while True:
        x1 = randint(0, 8)
        x2 = randint(0, 8)
        y1 = randint(0, 8)
        y2 = randint(0, 8)
        if C[x1][y1] == 0 and C[x2][y2] == 0 and S[x1][y1] != S[x2][y2]: break

    return x1, y1, x2, y2

def schedule_temp(T, a):
    return T * (1 - a)

def is_point_incorrect(S, x, y):
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

def count_sudoku_errors(Sudoku):
    errors = 0
    for i in range(9):
        for j in range(9):
            if is_point_incorrect(Sudoku, i, j):
                errors += 1
    return errors
    
def count_energy_difference(Sudoku, P1, P2):
    x1, y1 = P1
    x2, y2 = P2

    start_energy = count_sudoku_errors(Sudoku)

    Sudoku[x1][y1], Sudoku[x2][y2] = Sudoku[x2][y2],Sudoku[x1][y1]

    end_energy = count_sudoku_errors(Sudoku)

    Sudoku[x1][y1], Sudoku[x2][y2] = Sudoku[x2][y2],Sudoku[x1][y1]

    return end_energy - start_energy

def simulated_annealing(path, max_iter, init_temp, a, limit):
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

    print_sudoku(S) 
    plt.show()
  
#simulated_annealing("sudokus/sudoku1.txt", 100000, 4, 1e-4, 1e-2)
#simulated_annealing("sudokus/sudoku2.txt", 1000000, 1000, 1e-5, 1e-2)
#simulated_annealing("sudokus/sudoku3.txt", 1000000, 1000, 1e-5, 1e-4)