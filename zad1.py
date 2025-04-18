from random import randint, random
from math import exp
import matplotlib.pyplot as plt 
import numpy as np
import os, tempfile
from PIL import Image

# GENEROWANIE CHMUR PUNKTÓW
def random_cloud(n):
    return [(randint(0, 100), randint(0, 100)) for _ in range(n)]

def normal_distribution(n, clusters):
    points = []
    num_clusters = len(clusters)
    base_pts = n // num_clusters
    remainder = n % num_clusters

    for i, cluster in enumerate(clusters):
        pts_for_this_cluster = base_pts + (1 if i < remainder else 0)
        x, y = np.random.multivariate_normal(cluster["mean"], cluster["cov"], pts_for_this_cluster).T
        points.extend(zip(x, y))

    return points

def nine_groups(n, padding):
    clusters = [
        {"mean": [-padding, padding], "cov": [[600, 400], [400, 600]]},
        {"mean": [0, padding], "cov": [[600, 400], [400, 600]]},
        {"mean": [padding, padding], "cov": [[600, 400], [400, 600]]},
        {"mean": [-padding, 0], "cov": [[600, 400], [400, 600]]},
        {"mean": [0, 0], "cov": [[600, 400], [400, 600]]},
        {"mean": [padding, 0], "cov": [[600, 400], [400, 600]]},
        {"mean": [-padding, -padding], "cov": [[600, 400], [400, 600]]},
        {"mean": [0, -padding], "cov": [[600, 400], [400, 600]]},
        {"mean": [padding, -padding], "cov": [[600, 400], [400, 600]]}
    ]

    return normal_distribution(n, clusters)

# DŁUGOŚĆ ŚCIEŻKI
def path_length(points, distance):
    size = len(points)
    sum = 0

    for i in range(size - 1):
        sum += distance(points[i], points[i + 1])

    return sum

# DYSTANS
def euclidean_distance(p1, p2): 
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def manhatan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# TEMPERATURA
def euler_temp(x, T0):
    return T0 * exp(-x / T0)

def t0_const_temp(x, T0):
    return 2000 * exp(-x / T0)

def div_const_temp(x, T0):
    return T0 * exp(-x / 2000)

# PRAWDOPOBIEŃSTWO
def schedule_prob(E, T):
    return exp(E / T)

# WYSZUKIWANIE NASTĘPNEGO STANU
def arbitrary_swap(state: list):
    result = state[:]
    size = len(state)
    i = j = 0

    while i == j:
        i, j = randint(0, size - 1), randint(0, size - 1)

    result[i], result[j] = result[j], result[i]
    return result

def consecutive_swap(state: list):
    result = state[:]
    size = len(state)
    i = randint(0, size - 2)

    result[i], result[i + 1] = result[i + 1], result[i]
    return result

# SYMULOWANE WYŻARZANIE
# Generowanie GIFa pozostawiam jako opcjonalne, ponieważ zabiera dużo czasu.
def simulated_annealing(points, max_iter, initial_temp, 
                        neighbour_fun, distance_fun, make_gif = False):
    xs, ys = [], []
    frame_paths = []  
    temp_dir = None

    # W celu nieprzepełnienia RAMu tworzę tymczasowy folder, gdzie
    # będę zapisywał wygenerowane klatki GIFa
    if make_gif:
        fig, ax = plt.subplots()
        ax.set_xlim(min([x for x, _ in points]) - 5, max([x for x, _ in points]) + 5)
        ax.set_ylim(min([y for _, y in points]) - 5, max([y for _, y in points]) + 5)
        line, = ax.plot([], [], "ro-")
        title = ax.set_title("Wizualizacja działania")
        plt.tight_layout()
        plt.grid(True)
        temp_dir = tempfile.mkdtemp()

    # Algorytm wyżarzania
    for i in range(1, max_iter + 1):
        T = euler_temp(i, initial_temp)
        if T < 1e-12: break

        candidate = neighbour_fun(points)
        
        E = path_length(points, distance_fun) - \
            path_length(candidate, distance_fun)
        if E > 0:
            points = candidate[:]
        else:
            prob = schedule_prob(E, T)
            if random() < prob:
                points = candidate[:]

        # W każdej iteracji zapisujemy dane do późniejszego 
        # wykresu (numer iteracji, długość znalezionej ścieżki)
        xs.append(i)
        ys.append(path_length(points, distance_fun))

        if make_gif and (i - 1) % 500 == 0:
            # Zapis klatki to pliku
            x_coords = [x for x, _ in points]
            y_coords = [y for _, y in points]
            x_coords.append(points[0][0])
            y_coords.append(points[0][1])
            line.set_data(x_coords, y_coords)

            title.set_text(f"Progres = {round(i * 100 / max_iter, 2)}%")
            print(f"Generowanie klatek... {round(i * 100 / max_iter, 2)}%")
            
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            fig.savefig(frame_path, dpi=80)
            frame_paths.append(frame_path)

    if make_gif:
        print("Wygenerowano dane\nTworzenie pliku GIF, może to chwilę potrwać...")
        try:
            with Image.open(frame_paths[0]) as first_frame:
                other_frames = [Image.open(f) for f in frame_paths[1:]]
                first_frame.save(
                    'animation.gif',
                    format = 'GIF',
                    append_images = other_frames,
                    save_all = True,
                    duration = 10,
                    loop = 0
                )
                for frame in other_frames:
                    frame.close()
        except Exception as e:
            print(f"Error creating GIF: {e}")
            
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        plt.close(fig)

    # Przedstawiamy na wykresie uzyskane rozwiązanie
    fig, axs = plt.subplots(2)
    axs[0].plot(xs, ys)
    axs[0].set_title(f"Wynik algorytmu dla {max_iter} wykonań, n = {len(points)}")
    axs[0].set_xlabel("Ilość iteracji")
    axs[0].set_ylabel("Długość ścieżki")
    
    x_coords = [x for x, _ in points]
    y_coords = [y for _, y in points]

    x_coords.append(points[0][0])
    y_coords.append(points[0][1])

    axs[1].plot(x_coords, y_coords, "ro-")
    axs[1].set_title(f"Znaleziona ścieżka o długości " 
                    f"{round(path_length(points, distance_fun), 2)}")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def simulated_annealing_temp_test(points, max_iter, initial_temp, temp_fun):
    xs, ys = [], []
    tx, ty = [], []

    for i in range(1, max_iter + 1):
        T = temp_fun(i, initial_temp)
        tx.append(i)
        ty.append(T)
        if T < 1e-12: break
        candidate = arbitrary_swap(points)
        
        E = path_length(points, euclidean_distance) - \
            path_length(candidate, euclidean_distance)
        if E > 0:
            points = candidate[:]
        else:
            prob = schedule_prob(E, T)
            if random() < prob:
                points = candidate[:]

        xs.append(i)
        ys.append(path_length(points, euclidean_distance))

    _, axs = plt.subplots(3)
    axs[0].plot(xs, ys)
    axs[0].set_title(f"Wynik algorytmu dla {max_iter} wykonań, n = {len(points)}")
    axs[0].set_xlabel("Ilość iteracji")
    axs[0].set_ylabel("Długość ścieżki")

    axs[1].plot(tx, ty)
    axs[1].set_title(f"Funkcja temperatury")

    axs[2].plot([x for x, _ in points], [y for _, y in points], "ro-")
    axs[2].set_title(f"Znaleziona ścieżka o długości " 
                    f"{round(path_length(points, euclidean_distance), 2)}")

    plt.tight_layout()
    plt.show()


simulated_annealing(nine_groups(70, 200), 7 * iter, 8 * temp, arbitrary_swap, euclidean_distance, True)