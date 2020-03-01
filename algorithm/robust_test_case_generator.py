import numpy as np
import json
import os

def getRandCase(V, n, output):
    if not os.path.exists("../input/robust/input"+"_"+str(V)+"_"+str(n)):
        os.makedirs(os.path.dirname("../input/robust/input"+"_"+str(V)+"_"+str(n)+"/"+output+'.json'))
    levelSize = list(np.full(n, V))
    totalSize = 100
    levelNumber = n
    weights = []
    for i in range(len(levelSize)):
        weight = np.random.multinomial(totalSize - levelSize[i], [1/float(levelSize[i])]*levelSize[i], size=1)[0]
        newWeight = []
        for j in range(len(weight)):
            newWeight.append(int(weight[j]) + 1)
        weights.append(list(newWeight))
    nodes=[]
    for i in range(levelNumber):
        node = []
        for j in range(levelSize[i]):
            node.append({
                'name': str(i)+"_"+str(j),
                'id': j,
                'size': weights[i][j],
                'remainingSizeL': weights[i][j],
                'remainingSizeR': weights[i][j],
            })
        nodes.append(node)

    def addLink(level):
        link = []
        left = list(range(len(nodes[level])))
        right = list(range(len(nodes[level+1])))
        while not (len(left) == 0 & len(right) == 0):
            i = np.random.randint(len(left))
            j = np.random.randint(len(right))
            node1 = nodes[level][left[i]]
            node2 = nodes[level+1][right[j]]
            if node1['remainingSizeR'] > node2['remainingSizeL']:
                link.append({
                    'source': node1['name'],
                    'sourceid': left[i],
                    'target': node2['name'],
                    'targetid': right[j],
                    'value': node2['remainingSizeL']
                })
                node1['remainingSizeR'] -= node2['remainingSizeL']
                node2['remainingSizeL'] = 0
                right.remove(right[j])
            elif node1['remainingSizeR'] == node2['remainingSizeL']:
                link.append({
                    'source': node1['name'],
                    'sourceid': left[i],
                    'target': node2['name'],
                    'targetid': right[j],
                    'value': node2['remainingSizeL']
                })
                node1['remainingSizeR'] = 0
                node2['remainingSizeL'] = 0
                right.remove(right[j])
                left.remove(left[i])
            else:
                link.append({
                    'source': node1['name'],
                    'sourceid': left[i],
                    'target': node2['name'],
                    'targetid': right[j],
                    'value': node1['remainingSizeR']
                })
                node2['remainingSizeL'] -= node1['remainingSizeR']
                node1['remainingSizeR'] = 0
                left.remove(left[i])
        return link

    links = []
    completeLinks = 0
    for i in range(levelNumber-1):
        link = addLink(i)
        links.append(link)
        completeLinks += len(link)

    with open("../input/robust/input"+"_"+str(V)+"_"+str(n)+"/"+output+'.json', 'w') as f:
        json.dump({'nodes': nodes, 'links': links}, f)

    return completeLinks

sumK = 7

caseInfo = {}
np.random.seed(0)
for V in range(9, 13):
    caseInfo[V] = {}
    for n in range(1, sumK):
        totalLineNumber = 0
        for i in range(10):
            linkNumber = getRandCase(V, 2*n, str(i))
            totalLineNumber += linkNumber
        caseInfo[V][n] = totalLineNumber / 10
    sumK -= 1
with open('../input/robust/caseInfo.json', 'w') as f:
        json.dump(caseInfo, f)

 
