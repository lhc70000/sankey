import numpy as np
import json

def calculator(V, n, index):
    inputFile = "../input/robust/input"+"_"+str(V)+"_"+str(n)+"/"+str(index)+'.json'
    with open(inputFile, 'r') as f:
        data = json.load(f)
    layeredLinks = data['links']
    totalWeightedSum = 0
    for i in range(len(layeredLinks)):
        for j in range(len(layeredLinks[i])-1):
            for k in range(j+1, len(layeredLinks[i])):
                link1 = layeredLinks[i][j]['value']
                link2 = layeredLinks[i][k]['value']
                totalWeightedSum += link1 * link2
    return totalWeightedSum

result = {}
sumK = 6
for V in range(9, 13):
    result[V] = {}
    for n in range(1, sumK):
        result[V][n] = []
        for i in range(10):
            r = calculator(V, 2*n, str(i))
            result[V][n].append(r)
    sumK -= 1

with open("../output/robust_weighted_edge_sum.json", 'w') as outfile:
    json.dump(result, outfile)