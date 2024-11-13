import os
from argparse import ArgumentParser
from typing import Tuple, List

import numpy as np
import time

from utils.data_processing import parse_body, parse_mesh, parse_samplereadings, dataset_prefixes, parse_output, save_output
from utils.closest_point import build_triangle_centroid_kdtree, closest_point_on_mesh_slow, \
    closest_point_on_mesh_fast
from utils.pcd_2_pcd_reg import pcd_to_pcd_reg_w_known_correspondence

current_script_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(current_script_path)

def get_file_paths(dataset_prefix: str):
    """Returns the file paths for the dataset files."""
    problem_name = "Problem3" if "pa3" in dataset_prefix else "Problem5" if "pa5" in dataset_prefix else "Problem4"
    body_a_file_name = f"{problem_name}-BodyA.txt"
    body_b_file_name = f"{problem_name}-BodyB.txt"
    mesh_file_name = f"{problem_name}MeshFile.sur"
    sample_readings_file_name = f"{dataset_prefix}SampleReadingsTest.txt"
    body_a_path = os.path.join(CUR_DIR, f"../DATA/{body_a_file_name}")
    body_b_path = os.path.join(CUR_DIR, f"../DATA/{body_b_file_name}")
    mesh_path = os.path.join(CUR_DIR, f"../DATA/{mesh_file_name}")
    sample_readings_path = os.path.join(CUR_DIR, f"../DATA/{sample_readings_file_name}")
    return body_a_path, body_b_path, mesh_path, sample_readings_path

def retrieve_data(dataset_prefix: str) -> Tuple[dict, dict, dict, List[dict], int]:
    """Retrieve and parse all necessary data based on the dataset prefix."""
    body_a_path, body_b_path, mesh_path, sample_readings_path = get_file_paths(dataset_prefix)
    body_a = parse_body(body_a_path)
    body_b = parse_body(body_b_path)
    mesh = parse_mesh(mesh_path)
    sample_readings, num = parse_samplereadings(sample_readings_path, body_a["N"], body_b["N"])
    return body_a, body_b, mesh, sample_readings, num

def compute_displacement_vec(F_A: any, F_B: any, A_tip: np.ndarray) -> np.ndarray:
    """Compute the displacement vector d_k."""
    F_B_inv = F_B.inverse()
    A_tip_homogeneous = A_tip.reshape(1, 3)
    A_tip_tracker = F_A.transform_pts(A_tip_homogeneous)[0]
    return F_B_inv.transform_pts(A_tip_tracker.reshape(1, 3))[0]

def process_frame(k: int, sample_readings: List[dict], body_a: dict, body_b: dict, mesh: dict, kdtree: any,
                  triangle_indices_list: List) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Process a single frame and return the computed values."""

    # Get markers for body A and body B
    A_markers_tracker, B_markers_tracker = sample_readings[k]["A"], sample_readings[k]["B"]
    A_markers_body, B_markers_body = body_a["Y"], body_b["Y"]

    # Compute transformations F_A_k and F_B_k
    F_A_k = pcd_to_pcd_reg_w_known_correspondence(A_markers_body, A_markers_tracker)
    F_B_k = pcd_to_pcd_reg_w_known_correspondence(B_markers_body, B_markers_tracker)

    # Compute displacement d_k and s_k
    d_k = compute_displacement_vec(F_A_k, F_B_k, body_a["t"])
    s_k = d_k  # Because F_reg = I, so s_k = F_reg * d_k = d_k

    # Slow method
    start_time_slow = time.time()
    c_k_slow, distance_k_slow = closest_point_on_mesh_slow(s_k, mesh)
    end_time_slow = time.time()
    elapsed_slow = end_time_slow - start_time_slow

    # Fast method
    start_time_fast = time.time()
    c_k_fast, distance_k_fast = closest_point_on_mesh_fast(s_k, mesh, kdtree, triangle_indices_list, num_neighbors=5)
    end_time_fast = time.time()
    elapsed_fast = end_time_fast - start_time_fast

    # Check if the fast method is correct
    difference = np.linalg.norm(c_k_slow - c_k_fast)
    if difference > 1e-3:
        print(
            f"WARNING: Difference between Slow and Fast Methods Result in Non-Negligible Errors @ frame {k}: {difference}")

    return c_k_slow, s_k, distance_k_slow, elapsed_slow, elapsed_fast

def calculate_and_output_mse(data: list, dataset_prefix: str):
    if "Debug" in dataset_prefix:
        output_file_path = os.path.join(CUR_DIR, f"../DATA/{dataset_prefix}Output.txt")
        output_data = parse_output(output_file_path)
        mse = 0.0
        num_elems = 0
        for idx, row in enumerate(data):
            for i in range(len(row)):
                mse += (row[i] - output_data[idx][i]) ** 2
                num_elems += 1

        mse /= num_elems
        print("Mean Squared Error: {:.5f}\n".format(mse))
    else:
        print("")


def print_performance_improvements(slow_time: float, fast_time: float):
    print("Slow Method Time: {:.5f} seconds".format(slow_time))
    print("Fast Method Time: {:.5f} seconds".format(fast_time))
    speedup = slow_time / fast_time if slow_time > 0 else float("-1")
    print("Speedup Multiple: {:.5f}x".format(speedup))


def main(dataset_prefix: str):
    """The main script for programming assignment #3. Specify the prefix
    of the data you wish to run for (e.g. PA3-A-Debug-)."""

    # Retrieve and parse data
    body_a, body_b, mesh, sample_readings, num = retrieve_data(dataset_prefix)
    num_frames = len(sample_readings)

    # Build KDTree for the closest point search (part of fast version)
    kdtree, centroids, triangle_indices_list = build_triangle_centroid_kdtree(mesh)

    slow_time = 0.0
    fast_time = 0.0
    data = []

    for k in range(num_frames):
        c_k_slow, s_k, distance_k_slow, elapsed_slow, elapsed_fast = process_frame(
            k, sample_readings, body_a, body_b, mesh, kdtree, triangle_indices_list
        )
        slow_time += elapsed_slow
        fast_time += elapsed_fast
        data.append([c_k_slow[0], c_k_slow[1], c_k_slow[2], s_k[0], s_k[1], s_k[2], distance_k_slow])

    # Print Performance improvements: fast method vs. slow method
    print_performance_improvements(slow_time, fast_time)

    # Calculate MSE
    calculate_and_output_mse(data, dataset_prefix)

    # Write output to file
    save_output(dataset_prefix, num_frames, data, num)

def full_run():
    """Runs the main function on every dataset in the DATA folder."""
    for prefix in dataset_prefixes:
        main(prefix)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_prefix", type=str)
    parser.add_argument("--full_run", default=False, action="store_true")
    args = parser.parse_args()
    if args.full_run:
        full_run()
    else:
        main(args.dataset_prefix)