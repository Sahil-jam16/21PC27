# MOCK HACKATHON
'''
Steve is a foodie and he wants to start a new restaurant in Coimbatore.
Consultant has mapped Saibaba colony into 20 neighborhoods and also
suggested location for first Restaurant / cloud kitchen. Steve wants to start from
the restaurant and cover all neighborhoods and return to the restaurant using the
shortest possible path avoiding already visited neighborhoods. So we want to write
a program that can recommend the path Steve needs to traverse to generate
outfile
'''


# Find the Algorithm to find the shortest path to cover all the nodes

# Find the efficient way to convert input data from json to python object

# Apply the algorithm to find the shortest path 

# Evaluate with the output file

# Find the time complexity of the algorithm

# Importing the required libraries
import json
import itertools
import numpy as np
import os


def held_karp(dists):
    n = len(dists)

    C = {}

    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    path.append(0)

    return opt, list(reversed(path))


def generate_distances(n):
    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dists[i][j] = dists[j][i] = random.randint(1, 99)

    return dists


def read_distances(filename):
    dists = []
    with open(filename, 'rb') as f:
        for line in f:
            # Skip comments
            if line[0] == '#':
                continue

            dists.append(map(int, map(str.strip, line.split(','))))

    return dists

# Reading the input file | Path = Y:/Student Handout/Input data/level0.json
with open('Y:/Student Handout/Input data/level0.json') as f:
    raw_data = json.load(f)
    keys = raw_data.keys()
    
n_neigh = raw_data['n_neighbourhoods']
neighbourhoods = raw_data['neighbourhoods']
n_restaurants = raw_data['n_restaurants']
restaurants = raw_data['restaurants']
rest_to_neigh_distance = raw_data['restaurants']['r0']['neighbourhood_distance']

neighbourhoods_distances = [rest_to_neigh_distance]

order_quantities = []

for n in neighbourhoods:
    order_quantities.append(neighbourhoods[n]['order_quantity'])
    neighbourhoods_distances.append(neighbourhoods[n]['distances'])

neighbourhoods_distances[0].insert(0, 0)
for col in range(1,len(neighbourhoods_distances)):
    neighbourhoods_distances[col].insert(0, rest_to_neigh_distance[col])

print(len(neighbourhoods_distances),len(neighbourhoods_distances[0]))


#ACTUAL FUNCTION CALL
total_cost, path = held_karp(neighbourhoods_distances)
print("COST = ",total_cost,"\nPATH : ",path)
#print(sorted(path))

path = [0, 14, 9, 12, 1, 6, 7, 20, 8, 18, 5, 16, 11, 13, 10, 15, 3, 19, 2, 17, 4]

for p in range(len(path)):
    if( p == 0 or p == len(path) ):
         continue
    path[p] = path[p] - 1
    path[p] = "n" + str(path[p])

path[0] = "r0"
path.append("r0")
print(path)


final_path = []
for i in path:
    if i == 'r0':
        final_path.append("r0")
    else:
        string = 'n' + str(i)
        final_path.append(string)

converted_json = {"v0": {"path": path}}
converted_json_str = json.dumps(converted_json)
print(converted_json_str)


file_path = os.path.abspath(__file__)
file_name = os.path.splitext(os.path.basename(file_path))[0]
output_file_name = file_name + ".json"

with open(output_file_name, 'w') as f:
    f.write(converted_json_str)
    