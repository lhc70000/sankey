import numpy as np
import json
from itertools import groupby
import random

# our file: contains matrix calculation
import algorithm
# out file: contains eigen calculation
import helper

# ****************************************************************
# All outputs of this file are save in ../output/heur_case.txt
# ****************************************************************



# Set fixed variables like stated in paper
alpha1 = 0.1
N = 100
alpha2 = 0.1
M = 50



# ****************************************************************
# Processing data, adding dummny nodes, not part of the algorithm
# ****************************************************************
with open('../input/heur_case.json', 'r') as f:
    data = json.load(f)
# specify which layer each node belongs to
level = {
    0: ['Primary Uranium', 'Imports', 'Fuel for Energy (in)', 'Primary Oil', 'Primary Natural Gas', 'Primary Coal', 'Primary Biomass', 'Primary Hydroelectricity'],
    1: ['Uranium Production', 'Oil Production', 'Natural Gas Production', 'Coal Production', 'Biofuel Production'],
    2: ['Oil Domestic Use', 'Natural Gas Domestic Use', 'Coal Domestic Use', 'Biofuel Domestic'],
    3: ['Electricity Generation'],
    4: ['Electricity Domestic Use'],
    5: ['Non-Energy', 'Commercial & Institutional', 'Personal Transport', 'Residential', 'Industrial', 'Freight Transportation'],
    6: ['Exports', 'Conversion Losses', 'Useful Energy', 'Fuel for Energy (out)', 'Non-Energy Dummy End'],
}
levelNumber = 7
# add dummy nodes and links
addedLinks = []
numLink = len(data['links'])
for index in range(numLink):
    isLong = False
    source = data['links'][index]['source']
    target = data['links'][index]['target']
    for j in range(levelNumber - 1):
        if (source in level[j]) & (target not in level[j+1]):
            isLong = True
            addedLinks.append({
                'source': source,
                'target': 'dummy '+target+str(index)+str(j+1),
                'value': data['links'][index]['value'],
            })
            level[j+1].append('dummy '+target + str(index)+str(j+1))
            isFound = False
            for l in range(j+2, levelNumber):
                if ((target not in level[l]) & (not isFound)):
                    level[l].append('dummy '+target + str(index)+str(l))
                    addedLinks.append({
                        'source': 'dummy '+target+str(index)+str(l-1),
                        'target': 'dummy '+target+str(index)+str(l),
                        'value': data['links'][index]['value'],
                    })
                elif (target in level[l]):
                    isFound = True
                    addedLinks.append({
                        'source': 'dummy '+target+str(index)+str(l-1),
                        'target': target,
                        'value': data['links'][index]['value'],
                    }) 
    if (not isLong):
        addedLinks.append(data['links'][index])
# change data structure of links and nodes for better access
link1 = addedLinks
link2 = addedLinks
link1.sort(key=lambda content: content['source'])
groups1 = groupby(link1, lambda content: content['source'])
node0 = []
node1 = []
node2 = []
node3 = []
node4 = []
node5 = []
node6 = []
nodes = [node0, node1, node2, node3, node4, node5, node6]
np.random.seed(26)
for source, links in groups1:
    size = 0
    for link in links:
        size += float(link['value'])
    for i in [0, 1, 2, 3, 4, 5]:
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
for source, links in groups3:
    groupedLinks[source] = list(links)



# ****************************************************************
# Result of the combined method on this case obtained from the output graph
# ****************************************************************
heurOrder = [
    [8,2,1,7,6,4,3,5],
    [6,5,4,3,2,1,7],
    [10,13,12,5,4,14,6,3,15,7,2,16,8,1,9,11],
    [9,4,10,14,20,2,22,24,16,13,5,11,23,17,25,3,15,21,6,12,18,7,1,26,19,8],
    [10,4,11,16,22,2,24,26,18,15,5,12,25,19,27,3,17,23,6,13,20,7,1,14,9,28,21,8],
    [13,7,14,18,8,15,19,9,4,1,16,10,5,6,3,2,17,20,12,11],
    [2,5,1,4,3]
]
weightedCrossing = 0
crossing = 0
for i in range(0, len(heurOrder) - 1):
    order1 = heurOrder[i]
    order2 = heurOrder[i+1]
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
print("***************************")
print("Combined method result as a baseline:")
print("Order from the graph produced by the combined method:")
print(heurOrder)
print("Weighted crossing: ", weightedCrossing)
print("Crossing: ", crossing)



# ****************************************************************
# Generating matrices for Stage 1
# ****************************************************************
matrices = []
for i in [0,1,2,3,4,5]:
    matrix = []
    for node in nodes[i]:
        row = []
        for preNode in nodes[i+1]:
            value = 0
            for link in groupedLinks[node['name']]:
                if preNode['name'] == link['target']:
                    value = np.log10(float(link['value']) + 1)
            row.append(value)
        matrix.append([float(j)/sum(row) for j in row])
    matrix = np.array(matrix)
    matrices.append(matrix)

addedLinks.sort(key=lambda content: content['target'])
groups3 = groupby(addedLinks, lambda content: content['target'])

groupedLinks = {}
for target, links in groups3:
    groupedLinks[target] = list(links)

for i in [6,5,4,3,2,1]:
    matrix = []
    for node in nodes[i]:
        row = []
        for preNode in nodes[i-1]:
            value = 0
            for link in groupedLinks[node['name']]:
                if preNode['name'] == link['source']:
                    # value = float(link['value'])
                    value = np.log10(float(link['value']) + 1)
                    # value = 1
            row.append(value)
        matrix.append([float(j)/sum(row) for j in row])
    matrix = np.array(matrix)
    matrices.append(matrix)

addedLinks.sort(key=lambda content: content['source'])
groups3 = groupby(addedLinks, lambda content: content['source'])
groupedLinks = {}
for source, links in groups3:
    groupedLinks[source] = list(links)



# ****************************************************************
# Stage 1
# ****************************************************************
resultArray = [0] * N
for index in range(0, N):
    resultObj = algorithm.parallel(matrices, alpha1)
    result = resultObj['result']
    # calculating weighted crossing from obtained ordering
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
result = resultArray[0]['order']
toPrintResult = []
for r in range(len(result)):
    toPrintResult.append(list([int(x) for x in result[r]]))
print("***************************")
print("Stage 1 result:")
print("Order produced:")
print(toPrintResult)
print("Weighted crossing: ", resultArray[0]['weightedCrossing'])
print("Crossing: ", resultArray[0]['crossing'])



# ****************************************************************
# preparing data for Stage 2, not part of the algorithm
# ****************************************************************
# initialOrdering: result from Stage 1 (Best-in-N ordering)
orders = toPrintResult
preNodes = [
    [{'name': 'Fuel for Energy (in)', 'size': 2.0}, {'name': 'Imports', 'size': 5.0}, {'name': 'Primary Biomass', 'size': 0.5}, {'name': 'Primary Coal', 'size': 1.5}, {'name': 'Primary Hydroelectricity', 'size': 2.0}, {'name': 'Primary Natural Gas', 'size': 5.0},{'name': 'Primary Oil', 'size': 5.0}, {'name': 'Primary Uranium', 'size': 10.0}],
    [{'name': 'Biofuel Production', 'size': 0.5}, {'name': 'Coal Production', 'size': 3.0}, {'name': 'Natural Gas Production', 'size': 5.9}, {'name': 'Oil Production', 'size': 8.5}, {'name': 'Uranium Production', 'size': 10.9}, {'name': 'dummy Electricity Generation11', 'size': 0.2}, {'name': 'dummy Electricity Generation141', 'size': 2.0}],
    [{'name': 'Biofuel Domestic', 'size': 0.4}, {'name': 'Coal Domestic Use', 'size': 2.0}, {'name': 'Natural Gas Domestic Use', 'size': 3.25}, {'name': 'Oil Domestic Use', 'size': 4.0}, {'name': 'dummy Conversion Losses172', 'size': 0.1}, {'name': 'dummy Conversion Losses202', 'size': 3.5}, {'name': 'dummy Conversion Losses232', 'size': 0.9}, {'name': 'dummy Conversion Losses262', 'size': 0.5}, {'name': 'dummy Conversion Losses282', 'size': 0.1}, {'name': 'dummy Electricity Generation12', 'size': 0.2}, {'name': 'dummy Electricity Generation142', 'size':2.0}, {'name': 'dummy Electricity Generation162', 'size': 0.8}, {'name': 'dummy Exports152', 'size': 10.0}, {'name': 'dummy Exports192', 'size': 2.0}, {'name': 'dummy Exports222', 'size': 2.0}, {'name': 'dummy Exports252', 'size': 0.5}],
    [{'name': 'Electricity Generation', 'size': 6.3}, {'name': 'dummy Commercial & Institutional313', 'size': 0.5}, {'name': 'dummy Commercial & Institutional393', 'size': 0.5}, {'name': 'dummy Conversion Losses173', 'size': 0.1}, {'name': 'dummy Conversion Losses203', 'size': 3.5}, {'name': 'dummy Conversion Losses233', 'size': 0.9}, {'name': 'dummy Conversion Losses263', 'size': 0.5}, {'name': 'dummy Conversion Losses283', 'size': 0.1}, {'name': 'dummy Exports153', 'size': 10.0}, {'name': 'dummy Exports193', 'size': 2.0}, {'name': 'dummy Exports223', 'size': 2.0}, {'name': 'dummy Exports253', 'size': 0.5}, {'name': 'dummy Freight Transportation353', 'size': 0.5}, {'name': 'dummy Fuel for Energy (out)293', 'size': 0.5}, {'name': 'dummy Fuel for Energy (out)373', 'size': 0.25}, {'name': 'dummy Industrial343', 'size': 0.5}, {'name': 'dummy Industrial423', 'size': 1.0}, {'name': 'dummy Industrial443', 'size': 0.5}, {'name': 'dummy Industrial483', 'size': 0.1}, {'name': 'dummy Non-Energy303', 'size': 0.5}, {'name': 'dummy Non-Energy383', 'size': 0.25}, {'name': 'dummy Personal Transport323', 'size': 0.9}, {'name': 'dummy Personal Transport403', 'size': 0.25}, {'name': 'dummy Residential333', 'size': 0.5}, {'name': 'dummy Residential413', 'size':0.5}, {'name': 'dummy Residential473', 'size': 0.1}, ],
    [{'name': 'Electricity Domestic Use', 'size': 2.5}, {'name': 'dummy Commercial & Institutional314', 'size': 0.5}, {'name': 'dummy Commercial & Institutional394', 'size': 0.5}, {'name': 'dummy Conversion Losses174', 'size': 0.1}, {'name': 'dummy Conversion Losses204', 'size': 3.5}, {'name': 'dummy Conversion Losses234', 'size': 0.9}, {'name': 'dummy Conversion Losses264', 'size': 0.5}, {'name': 'dummy Conversion Losses284', 'size': 0.1}, {'name': 'dummy Conversion Losses514', 'size': 3.0}, {'name': 'dummy Exports154', 'size': 10.0}, {'name': 'dummy Exports194', 'size': 2.0}, {'name': 'dummy Exports224', 'size': 2.0}, {'name': 'dummy Exports254', 'size': 0.5}, {'name': 'dummy Exports504', 'size': 0.3}, {'name': 'dummy Freight Transportation354', 'size': 0.5}, {'name': 'dummy Fuel for Energy (out)294', 'size': 0.5}, {'name': 'dummy Fuel for Energy (out)374', 'size': 0.25}, {'name': 'dummy Industrial344', 'size':0.5}, {'name': 'dummy Industrial424', 'size': 1.0}, {'name': 'dummy Industrial444', 'size': 0.5}, {'name': 'dummy Industrial484','size': 0.1}, {'name': 'dummy Non-Energy304', 'size': 0.5}, {'name': 'dummy Non-Energy384', 'size': 0.25}, {'name': 'dummy Personal Transport324', 'size': 0.9}, {'name': 'dummy Personal Transport404', 'size': 0.25}, {'name': 'dummy Residential334', 'size': 0.5}, {'name': 'dummy Residential414', 'size': 0.5}, {'name': 'dummy Residential474', 'size': 0.1}],
    [{'name': 'Commercial & Institutional', 'size': 1.3}, {'name': 'Freight Transportation', 'size': 0.5}, {'name': 'Industrial', 'size': 3.6}, {'name': 'Non-Energy', 'size': 0.25}, {'name': 'Personal Transport', 'size': 1.15}, {'name': 'Residential', 'size': 1.1}, {'name': 'dummy Conversion Losses175', 'size': 0.1}, {'name': 'dummy Conversion Losses205', 'size': 3.5}, {'name': 'dummy Conversion Losses235', 'size': 0.9}, {'name': 'dummy Conversion Losses265', 'size': 0.5}, {'name': 'dummy Conversion Losses285', 'size': 0.1}, {'name': 'dummy Conversion Losses515', 'size': 3.0}, {'name': 'dummy Exports155', 'size': 10.0}, {'name': 'dummy Exports195', 'size': 2.0}, {'name': 'dummy Exports225', 'size': 2.0}, {'name': 'dummy Exports255', 'size': 0.5}, {'name': 'dummy Exports505', 'size': 0.3}, {'name': 'dummy Fuel for Energy (out)295', 'size': 0.5}, {'name': 'dummy Fuel for Energy (out)375', 'size': 0.25}, {'name': 'dummy Fuel for Energy (out)555', 'size': 0.2}],
    [{'name': 'Conversion Losses', 'size': 11.9}, {'name': 'Exports', 'size': 14.8}, {'name': 'Fuel for Energy (out)', 'size': 0.95}, {'name': 'Useful Energy', 'size': 3.85}, {'name': 'dummy Non-Energy Dummy End 566', 'size': 0.25}],   
    ]
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
addedLinks1 = addedLinks
links = []
for j in range(0, len(nodes)-1):
    linkLevel = []
    i = 0
    for link in addedLinks:
        if link['source'] in nodes[j].keys():
            link["id"] = i
            link['value'] = float(link['value'])
            i += 1
            linkLevel.append(link)
    links.append(linkLevel)
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
            # link['sourcepos'] = nodes[level][link['source']]['pos'] + (j / (nodes[level][link['source']]['right_edge_number'] + 1)) / maxNodes
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
minWeightedCrossing = 10000000
correspondingCrossing = 0
correspondingOrdering = []
minAchievedIteration = 0
for index in range(M):

    for i in [1, 2, 3, 4, 5]:
        calculateNodePos(i)
        updateNodeOrder(i)
        getNodePos(i)
        updateLinkPos(i-1, 'target')
        updateLinkPos(i, 'source')
        
    for i in [6]:
        calculateNodePos(i, isRight=True)
        updateNodeOrder(i)
        getNodePos(i)
        updateLinkPos(i-1, 'target')

    for i in [5, 4, 3, 2, 1]:
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
    
    weightedCrossing = 0
    crossing = 0
    result = getOrdering()
    for i in range(0, len(result) - 1):
        order1 = result[i]
        order2 = result[i+1]
        nodes1 = preNodes[i]
        nodes2 = preNodes[i+1]
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

print("***************************")
print("Stage 2 result:")
print("Order produced:")
print(correspondingOrdering)
print("Weighted crossing: ", minWeightedCrossing)
print("Crossing: ", correspondingCrossing)
print("Min achievd at the "+str(minAchievedIteration)+"-th iteration.")
print("***************************")