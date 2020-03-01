from pulp import *
import json
import sys
import numpy as np

def runILP(V, n, index):

    inputFile = "../input/robust/input"+"_"+str(V)+"_"+str(n)+"/"+index+'.json'
    prob = LpProblem(inputFile, LpMinimize)

    with open(inputFile, 'r') as f:
        data = json.load(f)


    nodes = data['nodes']
    links = data['links']

    x = []
    for i in range(len(nodes)):
        x_i = []
        for j in range(len(nodes[i])):
            x_i_j = []
            for k in range(len(nodes[i])):
                if (j != k):
                    x_jk = LpVariable(nodes[i][j]['name'] + nodes[i][k]['name'], 0, 1, LpInteger)
                    x_i_j.append(x_jk)
                else:
                    x_i_j.append(0)
            x_i.append(x_i_j)
        x.append(x_i)

    c = []
    for i in range(len(links)):
        c_i = []
        for j in range(len(links[i])):
            c_i_j = []
            for k in range(len(links[i])):
                source1 = links[i][j]['sourceid']
                target1 = links[i][j]['targetid']
                source2 = links[i][k]['sourceid']
                target2 = links[i][k]['targetid']
                if (j != k) & (source1 != source2) & (target1 != target2):
                    c_jk = LpVariable(str(i)+"_"+str(links[i][j]['sourceid'])+"_"+str(links[i][j]['targetid'])+"_"+str(links[i][k]['sourceid'])+"_"+str(links[i][k]['targetid']), 0, 1, LpInteger)
                    c_i_j.append({
                            'var': c_jk,
                            'source1': links[i][j]['sourceid'],
                            'target1': links[i][j]['targetid'],
                            'source2': links[i][k]['sourceid'],
                            'target2': links[i][k]['targetid'],
                            'weight': links[i][k]['value'] * links[i][j]['value']
                        })
                else:
                    c_i_j.append(0)
            c_i.append(c_i_j)
        c.append(c_i)

    # obj
    obj = 0
    for i in range(len(c)):
        for j in range(len(c[i])):
            for k in range(len(c[i][j])):
                if c[i][j][k] != 0:
                    obj += c[i][j][k]['weight'] * c[i][j][k]['var']
    prob += obj, 'obj'

    # cond 1
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(j+1, len(x[i])):
                prob += x[i][j][k] + x[i][k][j] == 1

    # cond 2
    for i in range(len(x)):
        for a in range(len(x[i])):
            for b in range(a+1, len(x[i])):
                for d in range(b+1, len(x[i])):
                    prob += x[i][a][d] >= x[i][a][b] + x[i][b][d] - 1

    # cond 3
    for i in range(len(c)):
        for j in range(len(c[i])):
            for k in range(len(c[i][j])):
                if j != k:
                    cross = c[i][j][k]
                    if cross != 0:
                        prob += cross['var'] + x[i][cross['source2']][cross['source1']] + x[i+1][cross['target1']][cross['target2']] >= 1
                        prob += cross['var'] + x[i][cross['source1']][cross['source2']] + x[i+1][cross['target2']][cross['target1']] >= 1

    # additional cond 1
    for i in range(len(c)):
        for j in range(len(c[i])):
            for k in range(j+1, len(c[i])):
                if c[i][j][k] != 0:
                    prob += c[i][j][k]['var'] == c[i][k][j]['var']

    status = prob.solve()
    return {'value': value(prob.objective) / 2, 'status': LpStatus[prob.status]}

result = {}
sumK = 6
for V in range(9, 13):
    result[V] = {}
    for n in range(1, sumK):
        result[V][n] = []
        for i in range(10):
            r = runILP(V, 2*n, str(i))
            result[V][n].append(r)
    sumK -= 1

result = {}
sumK = 5
for V in range(10, 11):
    result[V] = {}
    for n in range(1, 2):
        result[V][n] = []
        for i in range(10):
            r = runILP(V, 2*n, str(i))
            result[V][n].append(r)
    sumK -= 1

with open("../output/robust_ilp_result_10.json", 'w') as outfile:
    json.dump(result, outfile)