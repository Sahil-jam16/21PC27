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
import heapq

import heapq
import numpy as np

def tsp_dijkstra(matrix):
    num_nodes = len(matrix)
    start_node = 0

    # Initialize the result path and distance
    result_path = [start_node]
    result_distance = 0

    # Continue until all nodes are visited
    while len(result_path) < num_nodes:
        # Set self-node distances to a large integer
        large_integer = 999999  # Choose a sufficiently large integer value
        modified_matrix = np.copy(matrix)
        np.fill_diagonal(modified_matrix, large_integer)

        # Apply Dijkstra's algorithm
        distances = dijkstra(modified_matrix, start_node)

        # Find the minimum distance and the corresponding node
        min_distance = float('inf')
        min_node = None
        for node in range(num_nodes):
            if node not in result_path:
                if distances[node] < min_distance:
                    min_distance = distances[node]
                    min_node = node

        # Update the result path and distance
        result_path += [min_node]
        result_distance += min_distance

        # Remove visited node from the matrix
        modified_matrix[:, min_node] = large_integer
        modified_matrix[min_node, :] = large_integer

    # Return to the starting node
    result_distance += matrix[result_path[-1]][start_node]
    result_path += [start_node]

    return result_path, result_distance

def dijkstra(graph, start):
    num_nodes = len(graph)
    distances = {node: float('inf') for node in range(num_nodes)}
    distances[start] = 0

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in enumerate(graph[current_node]):
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


def tsp_held_karp(matrix):
    num_nodes = len(matrix)
    all_nodes = set(range(num_nodes))
    start_node = 0  # Assuming the starting node is 0

    # Initialize memoization table
    memo = {}

    # Helper function to compute the minimum distance for a subset of nodes ending at a particular node
    def held_karp_helper(subset, end_node):
        if not subset:
            return matrix[start_node][end_node]

        subset_key = tuple(sorted(list(subset)))
        if (subset_key, end_node) in memo:
            return memo[(subset_key, end_node)]

        min_distance = float('inf')
        for node in subset:
            new_subset = subset - {node}
            distance = held_karp_helper(new_subset, node) + matrix[node][end_node]
            min_distance = min(min_distance, distance)

        memo[(subset_key, end_node)] = min_distance
        return min_distance

    # Generate all subsets of nodes excluding the start node
    subsets = [set(combo) for combo in itertools.combinations(all_nodes - {start_node}, num_nodes - 1)]

    # Calculate minimum distance for each subset ending at the start node
    min_distance = float('inf')
    min_path = None
    for subset in subsets:
        distance = held_karp_helper(subset, start_node) + matrix[list(subset)[0]][start_node]
        if distance < min_distance:
            min_distance = distance
            min_path = [start_node] + list(subset) + [start_node]

    return min_path, min_distance



# Reading the input file | Path = Y:/Student Handout/Input data/level0.json
with open('Y:/Student Handout/Input data/level0.json') as f:
    raw_data = json.load(f)
    keys = raw_data.keys()
    
# Extracting the required data from the json file
n_neigh = raw_data['n_neighbourhoods']
neighbourhoods = raw_data['neighbourhoods']
n_restaurants = raw_data['n_restaurants']
restaurants = raw_data['restaurants']
rest_to_neigh_distance = raw_data['restaurants']['r0']['neighbourhood_distance']

neighbourhoods_distances = []
order_quantities = []



for n in neighbourhoods:
    order_quantities.append(neighbourhoods[n]['order_quantity'])
    neighbourhoods_distances.append(neighbourhoods[n]['distances'])

#neighbourhoods_distances = np.array(neighbourhoods_distances)


starting_neighbourhood = rest_to_neigh_distance.index( min(rest_to_neigh_distance) )
print(starting_neighbourhood)


best_path, best_distance = tsp_dijkstra(neighbourhoods_distances)

print("Best Path:", best_path)
print("Best Distance:", best_distance)


print("----------------------------------------------------------------------------------------------------------------")

neighbourhoods_distances.insert(0, rest_to_neigh_distance)


for i in range(len(neighbourhoods_distances)):
    neighbourhoods_distances[i].insert(0, rest_to_neigh_distance[i])

print(len(neighbourhoods_distances),len(neighbourhoods_distances[0]))





#output = {'v0': { 'path' : }}

#Best Path: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0]
#Best Distance: 12856

#Best Path: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0]
#Best Distance: 11985

output = {"v0": {"path": ["r0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15", "n16", "n17" , "n18", "n19", "r0"]}}


