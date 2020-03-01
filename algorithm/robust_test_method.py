from __future__ import division
import numpy as np
import json
from itertools import groupby
import algorithm
import helper


def runMethod(V, n, index):
    # Set fixed variables like stated in paper
    inputFile = "../input/robust/input"+"_"+str(V)+"_"+str(n)+"/"+str(index)+'.json'
    alpha1 = 0.01
    N = 100
    alpha2 = 0.1
    M = 100

    # ****************************************************************
    # processing data, adding dummny nodes, not part of the algorithm
    # ****************************************************************
    with open(inputFile, 'r') as f:
        data = json.load(f)
    layeredLinks = data['links']
    nodes = data['nodes']
    level = []
    for i in range(len(nodes)):
        l = []
        for j in range(len(nodes[i])):
            l.append(nodes[i][j])
        level.append(l)
    levelNumber = n
    addedLinks = []
    for i in range(len(layeredLinks)):
        for j in range(len(layeredLinks[i])):
            addedLinks.append(layeredLinks[i][j])
    link1 = addedLinks
    link2 = addedLinks
    link1.sort(key=lambda content: content['source'])
    groups1 = groupby(link1, lambda content: content['source'])
    for source, links in groups1:
        size = 0
        for link in links:
            size += float(link['value'])
        for i in range(n-1):
            if source in level[i]:
                nodes[i].append({
                    'name': source,
                    'size': size,
                })
    link2.sort(key=lambda content: content['target'])
    groups2 = groupby(link2, lambda content: content['target'])
    for target, links in groups2:
        if target in level[levelNumber - 1]:
            size = 0
            for link in links:
                size += float(link['value'])
            nodes[levelNumber - 1].append({
                'name': target,
                'size': size,
            })
    addedLinks.sort(key=lambda content: content['source'])
    groups3 = groupby(addedLinks, lambda content: content['source'])
    groupedLinks = {}
    for source, linksss in groups3:
        groupedLinks[source] = list(linksss)



    # ****************************************************************
    # Generating matrices for Stage 1
    # ****************************************************************
    matrices = []
    for i in range(n-1):
        matrix = []
        for node in nodes[i]:
            row = []
            for preNode in nodes[i+1]:
                value = 0
                for link in groupedLinks[node['name']]:
                    if preNode['name'] == link['target']:
                        value = float(link['value'])
                row.append(value)
            matrix.append([float(j)/sum(row) for j in row])
        matrix = np.array(matrix)
        matrices.append(matrix)

    addedLinks.sort(key=lambda content: content['target'])
    groups3 = groupby(addedLinks, lambda content: content['target'])

    groupedLinks = {}
    for target, links in groups3:
        groupedLinks[target] = list(links)

    for i in range(n-1, 0, -1):
        matrix = []
        for node in nodes[i]:
            row = []
            for preNode in nodes[i-1]:
                value = 0
                for link in groupedLinks[node['name']]:
                    if preNode['name'] == link['source']:
                        value = float(link['value'])
                row.append(value)
            matrix.append([float(j)/sum(row) for j in row])
        matrix = np.array(matrix)
        matrices.append(matrix)

    addedLinks.sort(key=lambda content: content['source'])
    groups3 = groupby(addedLinks, lambda content: content['source'])
    groupedLinks = {}
    for source, linksss in groups3:
        groupedLinks[source] = list(linksss)



    # ****************************************************************
    # stage 1
    # ****************************************************************
    resultArray = [0] * N
    for index in range(0, N):
        resultObj = algorithm.parallel(matrices, alpha1)
        result = resultObj['result']
        weightedCrossing = 0
        crossing = 0
        for i in range(0, len(result) - 1):
            order1 = result[i]
            order2 = result[i+1]
            nodes1 = nodes[i]
            nodes2 = nodes[i+1]
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
        resultArray[index] = {
            "weightedCrossing": weightedCrossing,
            "crossing": crossing,
            "order": result
        }
    resultArray.sort(key=lambda x: x['weightedCrossing'], reverse=False)
    stage1Result = resultArray[0]['weightedCrossing']
    result = resultArray[0]['order']
    toPrintResult = []
    for r in range(len(result)):
        toPrintResult.append(list(result[r]))

    # ****************************************************************
    # preparing data for Stage 2, not part of the algorithm
    # ****************************************************************
    numLink = len(data['links'])
    links = data['links']
    x = np.arange(M+1)
    yWeighted = []
    yNonWeighted = []
    preNodes = nodes
    orders = result
    nodes = []
    for i in range(0, len(preNodes)):
        newLevel = {}
        for j in range(0, len(orders[i])):
            newLevel[preNodes[i][int(orders[i][j])-1]["name"]] = {
                'order': j,
                # 'size': preNodes[i][int(orders[i][j])-1]["size"],
                'left_edge_number': 0,
                'right_edge_number': 0,
                'calculatedPos': 0,
                'id': int(orders[i][j]),
            }
        nodes.append(newLevel)
    for i in range(len(links)):
        for j in range(len(links[i])):
            link = links[i][j]
            nodes[i][link['source']]['right_edge_number'] += 1
            nodes[i+1][link['target']]['left_edge_number'] += 1




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
        indexes = [level, level+1]
        if orientation == 'right':
            priority.reverse()
            indexes.reverse()
        links[level].sort(key=lambda x: (nodes[indexes[0]][x[priority[0]]]['order'], nodes[indexes[1]][x[priority[1]]]['order']))

    def assignLinkPos(level, orientation):
        updateLinkOrder(level, orientation)

    def initPos():
        getAllNodePos()
        for level in range(levelNumber-1):
            updateLinkOrder(level, 'left')
            for i in range(len(links[level])):
                link = links[level][i]
                if (i == 0) | ((i != 0) & (links[level][i-1]['source'] != link['source'])):
                    j = nodes[level][link['source']]['right_edge_number']
                link['sourcepos'] = nodes[level][link['source']]['pos'] + (j / (nodes[level][link['source']]['right_edge_number'] + 1)) / len(nodes[level].keys())
                j = j - 1
            updateLinkOrder(level, 'right')
            for i in range(len(links[level])):
                link = links[level][i]
                if (i == 0) | ((i != 0) & (links[level][i-1]['target'] != link['target'])):
                    j = nodes[level+1][link['target']]['left_edge_number']
                link['targetpos'] = nodes[level+1][link['target']]['pos'] + (j / (nodes[level+1][link['target']]['left_edge_number'] + 1)) / len(nodes[level+1].keys())

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
                leftRands = np.random.rand(len(links[level-1]))
                for i in range(len(links[level-1])):
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
            for i in range(len(links[level-1])):
                link = links[level-1][i]
                nodes[level][link['target']]['left_pos'].append(link['sourcepos'])
                nodes[level][link['target']]['left_weight'].append(link['value'])
                leftPoses.append(link['sourcepos'])
        rightPoses = []
        if not isRight:
            for i in range(len(links[level])):
                link = links[level][i]
                nodes[level][link['source']]['right_pos'].append(link['targetpos'])
                nodes[level][link['source']]['right_weight'].append(link['value'])
                rightPoses.append(link['targetpos'])
        posCalculation(level, leftPoses, rightPoses)

    def updateNodeOrder(level):
        keys = list(nodes[level].keys())
        keys.sort(key = lambda k:nodes[level][k]['calculatedPos'], reverse=True)
        for i in range(len(keys)):
            nodes[level][keys[i]]['order'] = i

    def updateLinkPos(level, orientation):
        index = level+1
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
    minWeightedCrossing = stage1Result
    correspondingCrossing = 0
    correspondingOrdering = []
    minAchievedIteration = 0
    for index in range(M):

        for i in range(1, n-1):
            calculateNodePos(i)
            updateNodeOrder(i)
            getNodePos(i)
            updateLinkPos(i-1, 'target')
            updateLinkPos(i, 'source')
            
        for i in [n-1]:
            calculateNodePos(i, isRight=True)
            updateNodeOrder(i)
            getNodePos(i)
            updateLinkPos(i-1, 'target')


        for i in range(n-2, 0, -1):
            calculateNodePos(i)
            updateNodeOrder(i)
            getNodePos(i)
            updateLinkPos(i-1, 'target')
            updateLinkPos(i, 'source')


        for i in [0]:
            calculateNodePos(i, isLeft=True)
            updateNodeOrder(i)
            getNodePos(i)    
            updateLinkPos(i, 'source')


        result = getOrdering()
        checknodes = data['nodes']
        weightedCrossing = 0
        crossing = 0
        for i in range(0, len(result) - 1):
            order1 = result[i]
            order2 = result[i+1]
            nodes1 = checknodes[i]
            nodes2 = checknodes[i+1]
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
    # return result for this case
    return {'stage1': stage1Result, 'stage2': minWeightedCrossing}

np.random.seed(0)

result = {}
sumK = 6
for V in range(9, 13):
    result[V] = {}
    for n in range(1, sumK):
        result[V][n] = []
        for i in range(10):
            r = runMethod(V, 2*n, str(i))
            result[V][n].append(r)
    sumK -= 1

with open("../output/robust_method_result.json", 'w') as outfile:
    json.dump(result, outfile)

