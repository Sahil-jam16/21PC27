
def tsp_dijkstra(matrix,start_node):
    num_nodes = len(matrix)

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

