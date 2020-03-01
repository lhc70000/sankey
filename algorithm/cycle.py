import numpy as np
import json
from itertools import groupby
import algorithm
import helper
import matplotlib.pyplot as plt

# ****************************************************************
# two output pictures of this file are saved as ../output/dumbbell.png and ../output/box.pnd=g
# ****************************************************************


# Set fixed variables like stated in paper
alpha1 = 0.01
N = 100
alpha2 = 0.01
M = 50

# ****************************************************************
# Functions for generating the cycle case
# ****************************************************************
levelNumber = 4
levels = {
    '-1': 3,
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 0,
}
def getLevel(i):
    return levels[str(i)]

def getOptimalCase():
    levelSize = [5, 17, 29, 4]
    levelNumber = 4
    weights = []
    totalSize = 100
    for i in range(len(levelSize)):
        weight = np.random.multinomial(totalSize - levelSize[i], [1/float(levelSize[i])]*levelSize[i], size=1)[0]
        newWeight = []
        for j in range(len(weight)):
            newWeight.append(weight[j] + 1)
        weights.append(list(newWeight))
    # generate nodes
    nodes=[]
    level = []
    for i in range(levelNumber):
        node = []
        levelSub = []
        for j in range(levelSize[i]):
            levelSub.append(str(i)+"_"+str(j))
            node.append({
                'name': str(i)+"_"+str(j),
                'id': j,
                'size': weights[i][j],
                'remainingSizeL': weights[i][j],
                'remainingSizeR': weights[i][j],
            })
        nodes.append(node)
        level.append(levelSub)
    # genarate non-crossing links 
    links = []
    completeLinks = []
    for i in range(levelNumber):
        k = 0
        j = 0
        link = []
        while (not j == levelSize[i]) & (not k == levelSize[getLevel(i+1)]):
            node1 = nodes[getLevel(i)][j]
            node2 = nodes[getLevel(i+1)][k]
            if node1['remainingSizeR'] > node2['remainingSizeL']:
                link.append({
                    'source': node1['name'],
                    'sourceid': j,
                    'target': node2['name'],
                    'targetid': k,
                    'value': node2['remainingSizeL']
                })
                completeLinks.append({
                    'source': node1['name'],
                    'sourceid': j,
                    'target': node2['name'],
                    'targetid': k,
                    'value': node2['remainingSizeL']
                })
                node1['remainingSizeR'] -= node2['remainingSizeL']
                node2['remainingSizeL'] = 0
                k += 1
            elif node1['remainingSizeR'] == node2['remainingSizeL']:
                link.append({
                    'source': node1['name'],
                    'sourceid': j,
                    'target': node2['name'],
                    'targetid': k,
                    'value': node2['remainingSizeL']
                })
                completeLinks.append({
                    'source': node1['name'],
                    'sourceid': j,
                    'target': node2['name'],
                    'targetid': k,
                    'value': node2['remainingSizeL']
                })
                node1['remainingSizeR'] = 0
                node2['remainingSizeL'] = 0
                k += 1
                j += 1
            else:
                link.append({
                    'source': node1['name'],
                    'sourceid': j,
                    'target': node2['name'],
                    'targetid': k,
                    'value': node1['remainingSizeR']
                })
                completeLinks.append({
                    'source': node1['name'],
                    'sourceid': j,
                    'target': node2['name'],
                    'targetid': k,
                    'value': node1['remainingSizeR']
                })
                node2['remainingSizeL'] -= node1['remainingSizeR']
                node1['remainingSizeR'] = 0
                j += 1
        links.append(link)
    addedLinks = completeLinks
    totalWeightedSum = 0
    for i in range(len(links)):
        for j in range(len(links[i])-1):
            for k in range(j+1, len(links[i])):
                link1 = links[i][j]['value']
                link2 = links[i][k]['value']
                totalWeightedSum += link1 * link2
    return {'links':links, 'nodes': nodes, 'completeLinks': completeLinks, 'edge': totalWeightedSum}



# ****************************************************************
# Looping 20 test cases 
# ****************************************************************
# all weighted crossing are recorded here
stage1 = []
stage2 = []
np.random.seed(5)
for caseid in range(0, 20):
    # ****************************************************************
    # processing data
    # ****************************************************************
    data = getOptimalCase()
    links = data['links']
    nodes = data['nodes']
    completeLinks = data['completeLinks']
    edgeSum = data['edge']
    addedLinks = completeLinks
    addedLinks.sort(key=lambda content: content['source'])
    groups3 = groupby(addedLinks, lambda content: content['source'])
    groupedLinks = {}
    for source, linkss in groups3:
        groupedLinks[source] = list(linkss)



    # ****************************************************************
    # Generating matrices for Stage 1
    # ****************************************************************
    matrices = []
    for i in [0, 1, 2]:
        matrix = []
        for node in nodes[i]:
            row = []
            for j in range(len(nodes[getLevel(i+1)])):
                preNode = nodes[getLevel(i+1)][j]
                value = 0
                for link in groupedLinks[node['name']]:
                    if preNode['name'] == link['target']:
                        value = np.log10(float(link['value']) + 1)
                        # value = float(link['value'])
                row.append(value)
            matrix.append([float(j)/sum(row) for j in row])
        matrix = np.array(matrix)
        matrices.append(matrix)

    addedLinks.sort(key=lambda content: content['target'])
    groups3 = groupby(addedLinks, lambda content: content['target'])

    groupedLinks = {}
    for target, linkss in groups3:
        groupedLinks[target] = list(linkss)

    for i in [3,2,1]:
        matrix = []
        for node in nodes[getLevel(i)]:
            row = []
            for preNode in nodes[i-1]:
                value = 0
                for link in groupedLinks[node['name']]:
                    if preNode['name'] == link['source']:
                        # value = float(link['value'])
                        value = np.log10(float(link['value']) + 1)
                row.append(value)
            matrix.append([float(j)/sum(row) for j in row])
        matrix = np.array(matrix)
        matrices.append(matrix)

    addedLinks.sort(key=lambda content: content['source'])
    groups3 = groupby(addedLinks, lambda content: content['source'])
    groupedLinks = {}
    for source, linkss in groups3:
        groupedLinks[source] = list(linkss)




    # ****************************************************************
    # stage 1
    # ****************************************************************
    resultArray_N = [0] * N
    for index in range(0, N):
        resultObj = algorithm.parallel(matrices, alpha1)
        result = resultObj['result']
        # add the crossing between the last and the first layer
        result0 = result[0]
        result.append(result0)
        # calculating weighted crossing from obtained ordering
        weightedCrossing = 0
        crossing = 0
        for i in range(0, len(result)-1):
            order1 = result[i]
            order2 = result[getLevel(i+1)]
            nodes1 = nodes[i]
            nodes2 = nodes[getLevel(i+1)]
            m1 = np.empty([len(order1), len(order2)])
            m2 = np.empty([len(order1), len(order2)])
            for j in range(0, len(order1)):
                sourceName = nodes1[int(order1[j])-1]['name']
                for k in range(0, len(order2)):
                    targetName = nodes2[int(order2[k])-1]['name']
                    value1 = 0
                    value2 = 0
                    for link in groupedLinks[sourceName]:
                        if ((targetName == link['target']) & (sourceName ==link['source'])):
                            value1 = link['value']
                            value2 = 1
                    m1[j, k] = value1
                    m2[j, k] = value2
            weightedCrossing += helper.getCrossing(m1)
            crossing += helper.getCrossing(m2)
        resultArray_N[index] = {
            "weightedCrossing": weightedCrossing,
            "crossing": crossing,
            "order": result
        }
    resultArray_N.sort(key=lambda x: x['weightedCrossing'], reverse=False)
    result_N = resultArray_N[0]['order']
    toPrintResult_N = []
    for r in range(len(result_N)):
        toPrintResult_N.append(list([int(x) for x in result_N[r]]))
    ratio = (resultArray_N[0]['weightedCrossing']) / edgeSum
    stage1.append(ratio)




    # ****************************************************************
    # preparing data for Stage 2, not part of the algorithm
    # ****************************************************************
    preNodes = nodes
    # initialOrdering: result from Stage 1 (Best-in-N ordering with N = 50)
    orders = toPrintResult_N
    nodes = []
    for i in range(0, len(preNodes)):
        newLevel = {}
        for j in range(0, len(orders[i])):
            newLevel[preNodes[i][int(orders[i][j])-1]["name"]] = {
                'order': j,
                'size': preNodes[i][int(orders[i][j])-1]["size"],
                'left_edge_number': 0,
                'right_edge_number': 0,
                'calculatedPos': 0,
                'id': int(orders[i][j]),
            }
        nodes.append(newLevel)
    for i in range(len(links)):
        for j in range(len(links[i])):
            link = links[i][j]
            nodes[levels[str(i)]][link['source']]['right_edge_number'] += 1
            nodes[levels[str(i+1)]][link['target']]['left_edge_number'] += 1



    # ****************************************************************
    # calculation functions for Stage 2
    # ****************************************************************
    def getNodePos(layer):
        keys = list(nodes[layer].keys())
        for i in range(len(keys)):
            nodes[layer][keys[i]]['pos'] = (len(keys) - nodes[layer][keys[i]]['order'] - 1) / len(keys)

    def getAllNodePos():
        for i in range(levelNumber):
            getNodePos(i)

    def updateLinkOrder(level, orientation):
        priority = ['source', 'target']
        indexes = [levels[str(level)], levels[str(level+1)]]
        if orientation == 'right':
            priority.reverse()
            indexes.reverse()
        links[level].sort(key=lambda x: (nodes[indexes[0]][x[priority[0]]]['order'], nodes[indexes[1]][x[priority[1]]]['order']))

    def initPos():
        getAllNodePos()
        for level in range(levelNumber):
            updateLinkOrder(level, 'left')
            for i in range(len(links[level])):
                link = links[level][i]
                if (i == 0) | ((i != 0) & (links[level][i-1]['source'] != link['source'])):
                    j = nodes[levels[str(level)]][link['source']]['right_edge_number']
                link['sourcepos'] = nodes[levels[str(level)]][link['source']]['pos'] + (j / (nodes[levels[str(level)]][link['source']]['right_edge_number'] + 1)) / len(nodes[levels[str(level)]].keys())
                j = j - 1
            updateLinkOrder(level, 'right')
            for i in range(len(links[level])):
                link = links[level][i]
                if (i == 0) | ((i != 0) & (links[level][i-1]['target'] != link['target'])):
                    j = nodes[levels[str(level+1)]][link['target']]['left_edge_number']
                link['targetpos'] = nodes[levels[str(level+1)]][link['target']]['pos'] + (j / (nodes[levels[str(level+1)]][link['target']]['left_edge_number'] + 1)) / len(nodes[levels[str(level+1)]].keys())

                j = j - 1

    def posCalculation(level, leftPoses, rightPoses):
        keys = list(nodes[level].keys())
        isLeft = len(leftPoses) == 0
        isRight = len(rightPoses) == 0
        for i in range(len(keys)):
            node = nodes[level][keys[i]]
            # left
            leftPos = 0
            if not isLeft:
                leftRands = np.random.rand(len(links[levels[str(level-1)]]))
                for i in range(len(links[levels[str(level-1)]])):
                    leftPos += alpha2 * leftRands[i] * leftPoses[i] / sum(leftRands)
                for l in range(len(node['left_pos'])):
                    leftPos += (1-alpha2) * node['left_pos'][l] * node['left_weight'][l] / sum(node['left_weight'])
            # right
            rightPos = 0
            if not isRight:
                rightRands = np.random.rand(len(links[level]))
                for i in range(len(links[level])):
                    rightPos += alpha2 * rightRands[i] * rightPoses[i] / sum(rightRands)
                for r in range(len(node['right_pos'])):
                    rightPos += (1-alpha2) * node['right_pos'][r] * node['right_weight'][r] / sum(node['right_weight'])
            if (not isRight) & (not isLeft):
                node['calculatedPos'] = (leftPos + rightPos) / 2
            else:
                node['calculatedPos'] = leftPos + rightPos

    def calculateNodePos(level, isLeft = False, isRight = False):
        keys = list(nodes[level].keys())
        for i in range(len(keys)):
            nodes[level][keys[i]]['left_pos'] = []
            nodes[level][keys[i]]['right_pos'] = []
            nodes[level][keys[i]]['left_weight'] = []
            nodes[level][keys[i]]['right_weight'] = []
        leftPoses = []
        if not isLeft:
            for i in range(len(links[levels[str(level-1)]])):
                link = links[levels[str(level-1)]][i]
                nodes[level][link['target']]['left_pos'].append(link['sourcepos'])
                nodes[level][link['target']]['left_weight'].append(np.log10(link['value'] + 1))
                leftPoses.append(link['sourcepos'])
        rightPoses = []
        if not isRight:
            for i in range(len(links[level])):
                link = links[level][i]
                nodes[level][link['source']]['right_pos'].append(link['targetpos'])
                nodes[level][link['source']]['right_weight'].append(np.log10(link['value'] + 1))
                rightPoses.append(link['targetpos'])
        posCalculation(level, leftPoses, rightPoses)

    def updateNodeOrder(level):
        keys = list(nodes[level].keys())
        keys.sort(key = lambda k:nodes[level][k]['calculatedPos'], reverse=True)
        for i in range(len(keys)):
            nodes[level][keys[i]]['order'] = i

    def updateLinkPos(level, orientation):
        index = levels[str(level + 1)]
        otherOrientation = 'source'
        if orientation == 'source':
            otherOrientation = 'target'
            index = level
        for i in range(len(links[level])):
            link = links[level][i]
            link[orientation+"pos"] = nodes[index][link[orientation]]['pos'] + link[otherOrientation+"pos"] / len(nodes[index].keys())

    def getOrdering():
        orders = []
        for i in range(levelNumber):
            keys = list(nodes[i].keys())
            keys.sort(key = lambda k:nodes[i][k]['order'])
            order = []
            for j in range(len(keys)):
                order.append(nodes[i][keys[j]]['id'])
            orders.append(order)
        return orders



    # ****************************************************************
    # Stage 2
    # ****************************************************************
    initPos()
    minWeightedCrossing = 10000000
    correspondingCrossing = 0
    correspondingOrdering = []
    minAchievedIteration = 0

    for index in range(M):
        orgOrdering = getOrdering()

        for i in [0,1,2,3]:
            calculateNodePos(i)
            updateNodeOrder(i)
            getNodePos(i)
            updateLinkPos(levels[str(i-1)], 'target')
            updateLinkPos(i, 'source')

        weightedCrossing = 0
        crossing = 0
        result = getOrdering()
        for i in range(0, len(result)):
            order1 = result[getLevel(i)]
            order2 = result[getLevel(i+1)]
            nodes1 = preNodes[getLevel(i)]
            nodes2 = preNodes[getLevel(i+1)]
            m1 = np.empty([len(order1), len(order2)])
            m2 = np.empty([len(order1), len(order2)])
            for j in range(0, len(order1)):
                sourceName = nodes1[int(order1[j])-1]['name']
                for k in range(0, len(order2)):
                    targetName = nodes2[int(order2[k])-1]['name']
                    value1 = 0
                    value2 = 0
                    for link in groupedLinks[sourceName]:
                        if ((targetName == link['target']) & (sourceName ==link['source'])):
                            value1 = link['value']
                            value2 = 1
                    m1[j, k] = value1
                    m2[j, k] = value2
            weightedCrossing += helper.getCrossing(m1)
            crossing += helper.getCrossing(m2)
        if weightedCrossing < minWeightedCrossing:
            minWeightedCrossing = weightedCrossing
            correspondingCrossing = crossing
            correspondingOrdering = result
            minAchievedIteration = index
    stage2.append(minWeightedCrossing/edgeSum)



# ****************************************************************
# Generating pictures from recoreded results
# ****************************************************************
n = 20
X = stage1
Y = stage2
stage1 = [a for _,a in sorted(zip(Y,X))]
labels = [b for _,b in sorted(zip(Y,range(1,n+1)))]
Y.sort()
stage2 = Y
# dumbbell
fig, ax = plt.subplots(figsize=(7,3.8))
ax.set_xticklabels(labels)
ax.set_facecolor('#EFEFEF')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.vlines(x=range(n), ymin=stage2, ymax=stage1, color='black', alpha=0.4)
plt.scatter(range(n), stage1, color='#7FBFE8', alpha=0.8 , label='Markov stage')
plt.scatter(range(n), stage2, color='#6AF588', alpha=1, label='Refinement stage')
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), loc='upper left')
plt.legend()
plt.xticks(range(n))
plt.xlabel('Case ID')
plt.savefig("../output/dumbbell")
plt.close()
# box
boxData = [stage1, stage2]
fig, ax = plt.subplots(figsize=(5,3.8))
ax.set_facecolor('#EFEFEF')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticklabels(["Markov stage", "Refinement stage"])
plt.boxplot(boxData)
plt.savefig("../output/box")
