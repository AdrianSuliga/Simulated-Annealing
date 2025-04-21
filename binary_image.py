import matplotlib.pyplot as plt
from random import random, randint
from math import exp
import argparse
import os, tempfile
from PIL import Image
import numpy as np

# VALIDATION OF COMMAND LINE ARGUMENTS
def validate_positive_int(data: int) -> int:
    if data > 0:
        return data
    else:
        raise TypeError("Only positive numbers can be used")
    
def validate_float(data: float) -> float:
    if 0 < data < 1:
        return data
    else:
        raise TypeError("Float values has to be between 0 and 1")
    
def validate_neighbour_fun(fun: int) -> callable:
    all_funs = [
        point_energy_8_neighbours,
        point_energy_8_neighbours_v2,
        point_energy_4_neighbours_plus,
        point_energy_4_neighbours_cross,
        point_energy_8_neighbours_cross
    ]

    if 0 <= fun <= 4: return all_funs[fun]
    else: raise TypeError("Index out of range, has to be 0 - 4")

def random_image(n: int, delta: float):
    M = [[1 if random() > delta else 0 for _ in range(n)] for _ in range(n)]

    plt.imshow(M, cmap = 'gray', interpolation = 'nearest')
    plt.axis('off')
    plt.show()

    return M

def point_energy_8_neighbours(M, i, j):
    result, n = 0, len(M)
    points = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i][j] != M[i + x_offset][j + y_offset] else 0

    return result

def point_energy_8_neighbours_v2(M, i, j):
    result, n = 0, len(M)
    points = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i][j] == M[i + x_offset][j + y_offset] else 0

    return result

def point_energy_4_neighbours_plus(M, i, j):
    result, n = 0, len(M)
    points = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i][j] != M[i + x_offset][j + y_offset] else 0

    return result

def point_energy_4_neighbours_cross(M, i, j):
    result, n = 0, len(M)
    points = [(-1, -1), (1, 1), (-1, 1), (1, -1)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i][j] != M[i + x_offset][j + y_offset] else 0

    return result

def point_energy_8_neighbours_cross(M, i, j): 
    result, n = 0, len(M)

    points = [(-1, -1), (1, 1), (1, -1), (-1, 1), (-2, -2), (2, 2), (2, -2), (-2, 2)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i + x_offset][j + y_offset] == M[i][j] else 0

    return result

def calculate_energy(M, point_energy_function):
    sum, n = 0, len(M)

    for i in range(n):
        for j in range(n):
            sum += point_energy_function(M, i, j)

    return sum

def temp_fun(T0, a):
    return T0 * (1 - a)

def schedule_prob(E, T):
    if E < 0: return 1
    return exp(-E / T)

def schedule_neighbour(points):
    n = len(points)

    while True:
        x1, y1 = randint(0, n - 1), randint(0, n - 1)
        x2, y2 = randint(0, n - 1), randint(0, n - 1)
        if points[x1][y1] != points[x2][y2]: break

    return ((x1, y1), (x2, y2))

def calculate_point_energy_difference(points, P1, P2, point_energy_function, offsets):
    x1, y1 = P1
    x2, y2 = P2

    n = len(points)
    start_energy, end_energy = 0, 0

    for x_offset, y_offset in offsets:
        if -1 < x1 + x_offset < n and -1 < y1 + y_offset < n:
            start_energy += point_energy_function(points, x1 + x_offset, y1 + y_offset)
        if -1 < x2 + x_offset < n and -1 < y2 + y_offset < n:
            start_energy += point_energy_function(points, x2 + x_offset, y2 + y_offset)

    start_energy += point_energy_function(points, x1, y1)
    start_energy += point_energy_function(points, x2, y2)

    points[x1][y1], points[x2][y2] = points[x2][y2], points[x1][y1]

    for x_offset, y_offset in offsets:
        if -1 < x1 + x_offset < n and -1 < y1 + y_offset < n:
            end_energy += point_energy_function(points, x1 + x_offset, y1 + y_offset)
        if -1 < x2 + x_offset < n and -1 < y2 + y_offset < n:
            end_energy += point_energy_function(points, x2 + x_offset, y2 + y_offset)

    end_energy += point_energy_function(points, x1, y1)
    end_energy += point_energy_function(points, x2, y2)

    points[x1][y1], points[x2][y2] = points[x2][y2], points[x1][y1]

    return end_energy - start_energy

def simulated_annealing(points, max_iter, init_temp, point_energy_function, offsets, a, make_gif = False):
    xs, ys = [], []
    T = init_temp
    temp_dir = None

    all_energy = calculate_energy(points, point_energy_function)

    if make_gif:
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

    for i in range(1, max_iter + 1):
        T = temp_fun(T, a)
        if T < 1e-12: break

        P1, P2 = schedule_neighbour(points)

        dE = calculate_point_energy_difference(points, P1, P2, point_energy_function, offsets)

        x1, y1 = P1
        x2, y2 = P2

        if schedule_prob(dE, T) > random():
            all_energy += dE
            points[x1][y1], points[x2][y2] = points[x2][y2], points[x1][y1]

        xs.append(i)
        ys.append(all_energy)

        # We save only 1% of frames
        if make_gif and (max_iter < 1000 or i % (max_iter // 100) == 0):
            print(f"Generating frames... {round(i * 100 / max_iter, 2)}%")
            img_array = np.array(points, dtype = np.uint8) * 255
            img = Image.fromarray(img_array, mode='L')
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            img.save(frame_path)
            frame_paths.append(frame_path)

    if make_gif:
        print("Data generated\nCreating GIF, this may take a while...")
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
        print("GIF generated in output/animation.gif")

    plt.imshow(points, cmap = 'gray', interpolation = 'nearest')
    plt.axis('off')
    plt.grid(True)
    plt.show()

    plt.plot(xs, ys)
    plt.grid(True)
    plt.show()

offsets = {
    point_energy_8_neighbours: [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)],
    point_energy_8_neighbours_v2: [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)],
    point_energy_4_neighbours_plus: [(0, -1), (1, 0), (0, 1), (-1, 0)],
    point_energy_4_neighbours_cross: [(-1, -1), (1, 1), (-1, 1), (1, -1)],
    point_energy_8_neighbours_cross: [(-1, -1), (1, 1), (1, -1), (-1, 1), (-2, -2), (2, 2), (2, -2), (-2, 2)]
}

def main() -> None:
    parser = argparse.ArgumentParser(description = "Generate cool images with simulated annealing")

    parser.add_argument(
        "image_size",
        help = "Number of pixels in one row of square image",
        type = int
    )

    parser.add_argument(
        "black_points_density",
        help = "Density of black points - number between 0 and 1",
        type = float
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
        "neighbour_function",
        help = "How do we define neighbours?",
        type = int
    )

    parser.add_argument(
        '--gif',
        help = "Do you want to make a GIF showcasing how the algorithm works?",
        action = argparse.BooleanOptionalAction,
        default = False
    )

    args = parser.parse_args()

    n = validate_positive_int(args.image_size)
    delta = validate_float(args.black_points_density)
    max_iter = validate_positive_int(args.max_iterations)
    initial_temp = validate_positive_int(args.initial_temperature)
    slope = validate_positive_int(args.temperature_slope)
    func = validate_neighbour_fun(args.neighbour_function)
    make_gif = args.gif

    simulated_annealing(
        random_image(n, delta),
        max_iter,
        initial_temp,
        func,
        offsets[func],
        10 ** (-slope),
        make_gif
    )


if __name__ == "__main__":
    main()