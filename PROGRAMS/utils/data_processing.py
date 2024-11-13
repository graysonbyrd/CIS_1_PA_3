from typing import Dict, List
import numpy as np

dataset_prefixes = [
    "PA3-A-Debug-",
    "PA3-B-Debug-",
    "PA3-C-Debug-",
    "PA3-D-Debug-",
    "PA3-E-Debug-",
    "PA3-F-Debug-",
    "PA3-G-Unknown-",
    "PA3-H-Unknown-",
    "PA3-J-Unknown-"
]

def parse_body(path: str):
    """Parses a body.txt file according to the specifications
    in the homework description.

    Params:
        path (str): file path to the body.txt file

    Returns:
        Dict: dictionary containing the Y, t, and N values
    """
    assert "Body" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    N_markers, _ = data[0].split(" ")
    N_markers = int(N_markers)

    Y = list()
    for i in range(N_markers):
        Y.append([float(x) for x in data[i + 1].split(" ") if x != ""])
    t = [float(x) for x in data[N_markers + 1].split(" ") if x != ""]
    return {"Y": np.array(Y), "t": np.array(t), "N": N_markers}

def parse_mesh(path: str):
    """Parses a mesh.sur file according to the specifications
    in the homework description.

    Params:
        path (str): file path to the mesh.sur file

    Returns:
        Dict: dictionary containing the V, i, and n values
    """
    assert "Mesh" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    N_vertices = data[0]
    N_vertices = int(N_vertices)

    V = list()
    for i in range(N_vertices):
        V.append([float(x) for x in data[i + 1].split(" ") if x != ""])

    I = list()
    N = list()
    N_triangles = data[N_vertices + 1]
    N_triangles = int(N_triangles)
    for i in range(N_triangles):
        cur_row = [int(x) for x in data[N_vertices + 2 + i].split(" ") if x != ""]
        I.append(cur_row[:3])
        N.append(cur_row[3:])

    return {"V": np.array(V), "i": np.array(I), "n": np.array(N)}

def parse_samplereadings(path: str, N_A, N_B) -> (List[Dict], int):
    """Parses a samplereadingstest.txt file according to the specifications
    in the homework description.

    Params:
        path (str): file path to the samplereadingstest.text file

    Returns:
        List: list of dictionaries containing the A, B, and D values

    """
    assert "SampleReadings" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    N_S, N_samples, _, num = data[0].split(" ")
    N_S, N_samples, num = int(N_S), int(N_samples), int(num) # N_S = N_A + N_B + N_D
    N_D = N_S - N_A - N_B
    idx = 1
    samples = list()
    for _ in range(N_samples):
        A = list()
        B = list()
        D = list()
        for i in range(N_A):
            A.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_A
        for i in range(N_B):
            B.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_B
        for i in range(N_D):
            D.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_D
        samples.append({"A": np.array(A), "B": np.array(B), "D": np.array(D)})
    return samples, num

def parse_output(path: str):
    assert "Output" in path, "Wrong file."
    with open(path, "r") as file:
        output_data = file.readlines()

    parsed_output_data = list()
    for idx, line in enumerate(output_data):
        if idx == 0:
            continue
        parsed_output_data.append([float(x) for x in line.split(" ") if x != ""])

    return parsed_output_data




if __name__ == "__main__":
    path = "/Users/akhildeo/Desktop/Johns-Hopkins/Senior-Class-Materials/Fall/CIS/CIS_PA3_4_5/DATA/Problem3-BodyA.txt"
    body_a = parse_body(path)
    N_a = body_a["N"]
    print("Parsed Body A: ", body_a)

    path = "/Users/akhildeo/Desktop/Johns-Hopkins/Senior-Class-Materials/Fall/CIS/CIS_PA3_4_5/DATA/Problem3-BodyB.txt"
    body_b = parse_body(path)
    N_b = body_b["N"]
    print("Parsed Body B: ", body_b)

    path = "/Users/akhildeo/Desktop/Johns-Hopkins/Senior-Class-Materials/Fall/CIS/CIS_PA3_4_5/DATA/PA3-A-Debug-SampleReadingsTest.txt"
    data, _ = parse_samplereadings(path, N_a, N_b)
    print("Sample Readings Parse: ", data)

    path = "/Users/akhildeo/Desktop/Johns-Hopkins/Senior-Class-Materials/Fall/CIS/CIS_PA3_4_5/DATA/Problem3MeshFile.sur"
    mesh = parse_mesh(path)
    print("Parsed Mesh: ", mesh)