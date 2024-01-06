import json
import itertools
import os

def held_karp(dists):
    n = len(dists)
    delivery_path = []

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

# Reading the input file | Path = Y:/Student Handout/Input data/level1a.json
with open('Y:/Student Handout/Input data/level1a.json') as f:
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
for col in range(1, len(neighbourhoods_distances)):
    neighbourhoods_distances[col].insert(0, rest_to_neigh_distance[col])

print(order_quantities)

# ACTUAL FUNCTION CALL
total_cost, path = held_karp(neighbourhoods_distances)
print("COST = ", total_cost, "\nPATH : ", path)

for p in range(len(path)):
    if p == 0 or p == len(path):
        continue
    path[p] = path[p] - 1

final_paths = []
for p in path:
    if p == 0:
        continue
    string = 'n' + str(p)
    final_paths.append(string)

output_data = {"v0": {}}

# Assuming `num_paths` is the number of paths you want to create
num_paths = 4

for i in range(1, num_paths + 1):
    output_data["v0"]["path" + str(i)] = ["r0"] + final_paths[i-1:i+1] + ["r0"]

output_json_str = json.dumps(output_data)
print(output_json_str)

file_path = os.path.abspath(__file__)
file_name = os.path.splitext(os.path.basename(file_path))[0]
output_file_name = file_name + "_output.json"

with open(output_file_name, 'w') as f:
    f.write(output_json_str)


'''
analyse the problem statement and correctly modify the code

Problem Statement : Steve after walking thru the location is confident about prospects of pastry business in Saibaba colony and has finalized the location of Pastry cloud kitchen.
Steve is very risk averse so Steve advertised pre-order for Pastry to be delivered as evening snack. Steve has hired one delivery person and has a scooter with a
modified carrier. Based on the day's orders he wants to communicate the delivery slots to his customers. Since the scooter carrier has finite capacity and customers
may order cupcakes, birthday cake, bread etc We are tasked with writing a program that will allow Steve to enter all orders for the day and we create different
slots. We want to make sure we can fill the scooter carrier to max possible capacity and also conserve petrol by making each drop slot cover a minimum distance.
Goal is reduce number trips and total distance traveled.


'''

'''
5 paths 
COST =  8094 
PATH :  [0, 13, 7, 20, 1, 11, 19, 18, 14, 15, 12, 9, 8, 2, 17, 6, 16, 10, 4, 5, 3]
{"v0": {"path1": ["r0", "n12", "n6", "r0"], "path2": ["r0", "n6", "n19", "r0"], "path3": ["r0", "n19", "n10", "r0"], "path4": ["r0", "n10", "n18", "r0"], "path5": ["r0", "n18", "n17", "r0"]}}
'''
