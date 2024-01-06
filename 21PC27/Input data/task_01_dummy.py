import itertools
import numpy as np

'''

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

# Example usage:
# Assuming 'matrix' is a 2D matrix representing distances between nodes
matrix = [
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
]

best_path, best_distance = tsp_held_karp(matrix)

print("Best Path:", best_path)
print("Best Distance:", best_distance)


import heapq

def tsp_dijkstra(graph):
    num_nodes = len(graph)
    start_node = 0  # Assuming the starting node is 0

    # Initialize memoization table
    memo = {}

    # Helper function to compute the minimum distance for a subset of nodes ending at a particular node
    def tsp_dijkstra_helper(subset, end_node):
        if not subset:
            return graph[start_node][end_node]

        subset_key = tuple(sorted(list(subset)))
        if (subset_key, end_node) in memo:
            return memo[(subset_key, end_node)]

        min_distance = float('inf')
        for node in subset:
            new_subset = subset - {node}
            distance = tsp_dijkstra_helper(new_subset, node) + graph[node][end_node]
            min_distance = min(min_distance, distance)

        memo[(subset_key, end_node)] = min_distance
        return min_distance

    # Generate all nodes excluding the start node
    nodes = set(range(1, num_nodes))
    
    # Calculate minimum distance for each node ending at the start node
    min_distance = float('inf')
    min_path = None
    for node in nodes:
        distance = tsp_dijkstra_helper(nodes - {node}, node) + graph[node][start_node]
        if distance < min_distance:
            min_distance = distance
            min_path = [start_node, node, start_node]

    return min_path, min_distance

# Example usage:
# Assuming 'graph' is a 2D matrix representing distances between nodes
matrix = [
    [100, 2, 3, 4, 5],
    [3, 100, 1, 5, 6],
    [1, 2, 100, 5, 6],
    [4, 3, 2, 100, 1],
    [3, 1, 3, 5, 100]
]

best_path, best_distance = tsp_dijkstra(matrix)

print("Best Path:", best_path)
print("Best Distance:", best_distance)
'''
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

# Example usage:
matrix = np.array([
    [100, 2, 3, 4, 5],
    [3, 100, 1, 5, 6],
    [1, 2, 100, 5, 6],
    [4, 3, 2, 100, 1],
    [3, 1, 3, 5, 100]
])

best_path, best_distance = tsp_dijkstra(matrix)

print("Best Path:", best_path)
print("Best Distance:", best_distance)


'''
------------------------------------------------------------------------------------------------------------------------


import heapq

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

import heapq

def tsp_dijkstra(matrix):
    num_nodes = len(matrix)
    start_node = 0  # Assuming the starting node is 0

    distances = {(node, frozenset()): float('infinity') for node in range(num_nodes)}
    distances[(start_node, frozenset())] = 0

    priority_queue = [(0, start_node, frozenset([start_node]))]

    while priority_queue:
        current_distance, current_node, visited_nodes = heapq.heappop(priority_queue)

        if len(visited_nodes) == num_nodes and current_node == start_node:
            # All nodes are visited, and we returned to the starting point
            return current_distance

        for neighbor in range(num_nodes):
            if neighbor not in visited_nodes:
                distance = current_distance + matrix[current_node][neighbor]

                new_visited_nodes = visited_nodes | {neighbor}
                state = (neighbor, frozenset(new_visited_nodes))

                if distance < distances[state]:
                    distances[state] = distance
                    heapq.heappush(priority_queue, (distance, neighbor, new_visited_nodes))

    # If the loop completes without finding a solution, it means the graph is not connected.
    return "No valid Hamiltonian path exists."

print(tsp_dijkstra(neighbourhoods_distances))

------------------------------------------------------------------------------------------------------------------------
'''