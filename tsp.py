from random import randint, random
from math import exp
import matplotlib.pyplot as plt 
import argparse
import numpy as np
import os, tempfile
from PIL import Image

# VALIDATION OF COMMAND LINE ARGUMENTS
def validate_positive_int(data: int) -> int:
    if data > 0:
        return data
    else:
        raise TypeError("Only positive numbers can be used")

def validate_mode(data: str, modes: list) -> str:
    if data in modes:
        return data
    else:
        raise TypeError(f"Invalid mode given, received {data} but expected one of {modes}")
    
def parse_neighbour_fun(fun: str, funs: list) -> callable:
    if fun == "r": 
        return random_swap
    elif fun == "c":
        return consecutive_swap
    else:
        raise TypeError(f"Invalid neighbour function was given. Got {fun}, expected one of {funs}")

def parse_distance_mode(fun: str, funs: list) -> callable:
    if fun == "e":
        return euclidean_distance
    elif fun == "m":
        return manhatan_distance
    else:
        raise TypeError(f"Invalid distance function was given. Got {fun}, expected one of {funs}")

# GENERATING CLOUD OF POINTS
def generate_points(n: int, mode: str) -> list:
    if mode == "r":
        return random_cloud(n)
    elif mode == "n":
        clusters = [
            {"mean": [-100, -100], "cov": [[600, 400], [400, 600]]},
            {"mean": [0, 100], "cov": [[600, 400], [400, 600]]},
            {"mean": [100, -100], "cov": [[600, 400], [400, 600]]}
        ]
        return normal_distribution(n, clusters)
    elif mode == "g":
        return nine_groups(n, 300)
    else:
        raise TypeError(f"Incorrect generation mode given - {mode}")

# Random generation
def random_cloud(n):
    return [(randint(0, 100), randint(0, 100)) for _ in range(n)]

# Generate points in clusters specified by user
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

# Generate points in 9 predefined clusters separated by padding
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

# CYCLE LENGTH
def cycle_length(points, distance):
    size = len(points)
    sum = 0

    for i in range(size - 1):
        sum += distance(points[i], points[i + 1])

    sum += distance(points[0], points[size - 1])

    return sum

# DISTANCE
def euclidean_distance(p1, p2): 
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def manhatan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# TEMPERATURE
def euler_temp(x, T0):
    return T0 * exp(-x / T0)

# PROBABILITY
def schedule_prob(E, T):
    return exp(E / T)

# CHOOSING NEIGHBOUR
def random_swap(state: list):
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

# SIMULATED ANNEALING
# Generating GIF is optional as it takes a lot of time
def simulated_annealing(points, max_iter, initial_temp, 
                        neighbour_fun, distance_fun, make_gif):
    xs, ys = [], []
    frame_paths = []  
    temp_dir = None

    # GIF's frames are stored in temporary folder
    # to avoid filling of RAM
    if make_gif:
        fig, ax = plt.subplots()
        ax.set_xlim(min([x for x, _ in points]) - 5, max([x for x, _ in points]) + 5)
        ax.set_ylim(min([y for _, y in points]) - 5, max([y for _, y in points]) + 5)
        line, = ax.plot([], [], "ro-")
        title = ax.set_title("Visualization")
        plt.tight_layout()
        plt.grid(True)
        temp_dir = tempfile.mkdtemp()

    # Annealing
    for i in range(1, max_iter + 1):
        T = euler_temp(i, initial_temp)
        if T < 1e-12: break

        candidate = neighbour_fun(points)
        
        E = cycle_length(points, distance_fun) - \
            cycle_length(candidate, distance_fun)
        if E > 0:
            points = candidate[:]
        else:
            prob = schedule_prob(E, T)
            if random() < prob:
                points = candidate[:]

        xs.append(i)
        ys.append(cycle_length(points, distance_fun))

        # We save only 1% of frames
        if make_gif and (max_iter < 1000 or i % (max_iter // 100) == 0):
            x_coords = [x for x, _ in points]
            y_coords = [y for _, y in points]
            x_coords.append(points[0][0])
            y_coords.append(points[0][1])
            line.set_data(x_coords, y_coords)

            title.set_text(f"Progress = {round(i * 100 / max_iter, 2)}%")
            print(f"Generating frames... {round(i * 100 / max_iter, 2)}%")
            
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            fig.savefig(frame_path, dpi=80)
            frame_paths.append(frame_path)

    if make_gif:
        print("Data generated\nCreating GIF, this may take a while...")
        try:
            with Image.open(frame_paths[0]) as first_frame:
                other_frames = [Image.open(f) for f in frame_paths[1:]]
                first_frame.save(
                    'output/animation.gif',
                    format = 'GIF',
                    append_images = other_frames,
                    save_all = True,
                    duration = 10,
                    loop = 0
                )
                for frame in other_frames:
                    frame.close()
        except Exception as e:
            print(f"Error while creating GIF: {e}")
            
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
        print("GIF generated in output/animation.gif")

    # Plotting solutions
    fig, axs = plt.subplots(2)
    axs[0].plot(xs, ys)
    axs[0].set_title(f"Result for {max_iter} iterations, n = {len(points)}")
    axs[0].set_xlabel("Iteration numbers")
    axs[0].set_ylabel("Cycle length")
    
    x_coords = [x for x, _ in points]
    y_coords = [y for _, y in points]

    x_coords.append(points[0][0])
    y_coords.append(points[0][1])

    axs[1].plot(x_coords, y_coords, "ro-")
    axs[1].set_title(f"Found cycle with length " 
                    f"{round(cycle_length(points, distance_fun), 2)}")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description = "Solve TSP using simulated annealing")

    parser.add_argument(
        "number_of_points",
        help = "Number of points to generate",
        type = int
    )

    parser.add_argument(
        "how_to_generate", 
        help = "How to generate cloud of points? (r / n / g)", 
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
        "neighbour_function",
        help = "How to choose next state in an algorithm? \
                You can choose random or consecutive swap (r / c)",
        type = str
    )

    parser.add_argument(
        "distance_metric",
        help = "How to calculate distance between two cities?\n\
                You can choose either euclidean or manhatan metric (e / m)",
        type = str
    )

    parser.add_argument(
        '--gif',
        help = "Do you want to make a GIF showcasing how the algorithm works?",
        action = argparse.BooleanOptionalAction,
        default = False
    )

    args = parser.parse_args()

    n = validate_positive_int(args.number_of_points)
    generation_mode = validate_mode(args.how_to_generate, ["r", "n", "g"])
    max_iter = validate_positive_int(args.max_iterations)
    initial_temperature = validate_positive_int(args.initial_temperature)
    neighbour_mode = validate_mode(args.neighbour_function, ["r", "c"])
    distance_mode = validate_mode(args.distance_metric, ["e", "m"])
    make_gif = args.gif 

    print(n, generation_mode, max_iter, initial_temperature, neighbour_mode, distance_mode, make_gif)
    simulated_annealing(
        generate_points(n, generation_mode),
        max_iter,
        initial_temperature,
        parse_neighbour_fun(neighbour_mode, ["r", "c"]),
        parse_distance_mode(distance_mode, ["e", "m"]),
        make_gif
    )

if __name__ == "__main__":
    main()
