import os
from argparse import ArgumentParser
from PROGRAMS.utils.data_processing import parse_body, parse_mesh, parse_samplereadings
from utils.data_processing import dataset_prefixes

current_script_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(current_script_path)

def get_file_paths(dataset_prefix: str):
    """Returns the file paths for the dataset files."""
    problem_name = "Problem3" if "pa3" in dataset_prefix else "Problem5" if "pa5" in dataset_prefix else "Problem4"
    body_a_file_name = f"{problem_name}-BodyA.txt"
    body_b_file_name = f"{problem_name}BodyB.txt"
    mesh_file_name = f"{problem_name}MeshFile.sur"
    sample_readings_file_name = f"{dataset_prefix}SampleReadingsTest.txt"
    body_a_path = os.path.join(CUR_DIR, f"../DATA/{body_a_file_name}")
    body_b_path = os.path.join(CUR_DIR, f"../DATA/{body_b_file_name}")
    mesh_path = os.path.join(CUR_DIR, f"../DATA/{mesh_file_name}")
    sample_readings_path = os.path.join(CUR_DIR, f"../DATA/{sample_readings_file_name}")
    return body_a_path, body_b_path, mesh_path, sample_readings_path

def main(dataset_prefix: str):
    """The main script for programming assignment #3. Specify the prefix
    of the data you wish to run for (e.g. pa1-debug-a-)."""

    # Retrieve data
    body_a_path, body_b_path, mesh_path, sample_readings_path = get_file_paths(dataset_prefix)
    body_a = parse_body(body_a_path)
    body_b = parse_body(body_b_path)
    mesh = parse_mesh(mesh_path)
    sample_readings = parse_samplereadings(sample_readings_path, body_a["N"], body_b["N"])




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