import json
import numpy as np
import os


def travellingsalesman(c):
    global cost
    global path
    adj_vertex = 999999
    min_val = 999999
    visited[c] = 1
    path.append(c)
    print(c, end=" ")
    for k in range(n):
        if (tsp_g[c][k] != 0) and (visited[k] == 0):
            if tsp_g[c][k] < min_val:
                min_val = tsp_g[c][k]
                adj_vertex = k
    if min_val != 999999:
        cost = cost + min_val
    if adj_vertex == 999999:
        adj_vertex = 0
        path.append(adj_vertex)
        print(adj_vertex, end=" ")
        cost = cost + tsp_g[c][adj_vertex]
        return
    travellingsalesman(adj_vertex)

f = open('Y:/Student Handout/Input data/level0.json')
data = json.load(f)
rest = data["restaurants"]["r0"]["neighbourhood_distance"]
dist_matrix = []
for j in range(len(data['neighbourhoods'])):
    node = "n"+str(j)
    dist_s = []
    for i in data['neighbourhoods'][node]['distances']:
        dist = i
        dist_s.append(dist)
    dist_matrix.append(dist_s)
for index in range(len(dist_matrix)):
    dist_matrix[index] = [rest[index]] + dist_matrix[index]
    index = index + 1
rest.insert(0,0)
rest = [rest]
for i in dist_matrix:
    rest.append(i)
f.close()

n = len(rest)
cost = 0
path = []
visited = np.zeros(n, dtype=int)
tsp_g = np.array(rest)
print("Shortest Path: ")
travellingsalesman(0)
print(n)
print("Minimum Cost:", end=" ")
print(cost)

for p in range(1,len(path)-1):
    path[p] = path[p] - 1
    path[p] = "n" + str(path[p])



path[0] = "r0"
path[len(path)-1] = "r0"
print(path)


x = {"v0": {"path": ["r0", "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "r0"]}}
#y =['r0', 'n13', 'n8', 'n3', 'n16', 'n1', 'n18', 'n9', 'n14', 'n17', 'n4', 'n15', 'n10', 'n12', 'n6', 'n7', 'n19', 'n5', 'n0', 'n11', 'n2', 'r0']


final_path = []
for i in path:
    if i == 'r0':
        final_path.append("r0")
    else:
        string = 'n' + str(int(i[1:]))
        final_path.append(string)

converted_json = {"v0": {"path": final_path}}
converted_json_str = json.dumps(converted_json)
print(converted_json_str)


# Get the file path
file_path = os.path.abspath(__file__)

# Get the file name without extension
file_name = os.path.splitext(os.path.basename(file_path))[0]

# Create the output file name
output_file_name = file_name + ".json"

# Write the converted_json_str to the output file
with open(output_file_name, 'w') as f:
    f.write(converted_json_str)
    

