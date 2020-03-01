import numpy as np
import json
from itertools import groupby

# ****************************************************************
# all outputs of this file are save in ../output/ilp_case.txt
# ****************************************************************

# our file: contains matrix calculation
import algorithm
# out file: contains eigen calculation
import helper
 
# Set fixed variables like stated in paper
alpha1 = 0.01
N = 100
alpha2 = 0.1
M = 50

# ****************************************************************
# processing data, adding dummny nodes, not part of the algorithm
# ****************************************************************
with open('../input/ilp_case.json', 'r') as f:
    data = json.load(f)
# specify which layer each node belongs to
level = {
    0: ['Agriculture', 'Waste', 'Energy', 'Industrial Processes', 'Land Use Change'],
    1: ['Harvest / Management', 'Deforestation', 'Landfills', 'Waste water - Other Waste', 'Agriculture Soils', 'Rice Cultivation', 'Other Agriculture', 'Livestock and Manure', 'Electricity and heat', 'Fugitive Emissions', 'Transportation', 'Industry', 'Other Fuel Combustion'],
    2: ['Coal Mining', 'Machinery', 'Pulp - Paper and Printing', 'Air', 'Unallocated Fuel Combustion', 'Commercial Buildings', 'T and D Losses', 'Residential Buildings', 'Food and Tobacco', 'Iron and Steel', 'Oil and Gas Processing', 'Agricultural Energy Use', 'Rail - Ship and Other Transport', 'Road', 'Aluminium Non-Ferrous Metals', 'Other Industry', 'Chemicals', 'Cement'],
    3: ['Carbon Dioxide', 'HFCs - PFCs', 'Methane', 'Nitrous Oxide'],
}
levelNumber = 4
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
nodes = [node0, node1, node2, node3]
np.random.seed(38)
for source, links in groups1:
    size = 0
    for link in links:
        size += float(link['value'])
    for i in [0, 1, 2]:
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
# result of the BC method on the case obtained from the output graph
# ****************************************************************
bcOrder = [[2, 4, 3, 1, 5], [3, 12, 10, 6, 4, 2, 5, 17, 8, 16, 11, 14, 9, 15, 1, 7, 13],
[16, 15, 7, 17, 10, 14, 2, 1, 19, 13, 18, 9, 12, 20, 8, 11, 22, 24, 5, 23, 27, 6, 28, 26, 21, 3, 25, 4, 29],
[1, 2, 3, 4]]
weightedCrossing = 0
crossing = 0
for i in range(0, len(bcOrder) - 1):
    order1 = bcOrder[i]
    order2 = bcOrder[i+1]
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
print("BC method result as a baseline:")
print("Order from the graph produced by the BC method:")
print(bcOrder)
print("Weighted crossing: ", weightedCrossing)
print("Crossing: ", crossing)



# ****************************************************************
# result of the BC method on the case obtained from the output graph
# ****************************************************************
ilpOrder = [
    [1, 5, 3, 2, 4],
    [1, 9, 8, 11, 13, 7, 4, 15, 16, 14, 17, 6, 3, 10, 12, 2, 5],
    [26, 28, 27, 29, 23, 22, 24, 25, 21, 6, 11, 4, 5, 3, 9, 12, 8, 13, 17, 10, 7, 18, 15, 1, 16, 14, 2, 19, 20],
    [4, 3, 2, 1],
]
weightedCrossing = 0
crossing = 0
for i in range(0, len(ilpOrder) - 1):
    order1 = ilpOrder[i]
    order2 = ilpOrder[i+1]
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
print("ILP method result as a baseline:")
print("Order from the graph produced by the ILP method:")
print(ilpOrder)
print("Weighted crossing: ", weightedCrossing)
print("Crossing: ", crossing)



# ****************************************************************
# Generating matrices for Stage 1
# ****************************************************************
matrices = []
for i in [0, 1, 2,]:
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

for i in [3,2,1]:
    matrix = []
    for node in nodes[i]:
        row = []
        for preNode in nodes[i-1]:
            value = 0
            for link in groupedLinks[node['name']]:
                if preNode['name'] == link['source']:
                    value = np.log10(float(link['value']) + 1)
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
# stage 1
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
    [{'name': 'Agriculture', 'size': 13.8},{'name': 'Energy', 'size': 65.6}, {'name': 'Industrial Processes', 'size': 5.1}, {'name': 'Land Use Change', 'size': 12.200000000000001}, {'name': 'Waste', 'size': 3.2}],
    [{'name': 'Agriculture Soils','size': 5.2}, {'name': 'Deforestation','size': 10.9}, {'name': 'Electricity and heat', 'size': 23.0}, {'name': 'Fugitive Emissions', 'size': 4.5}, {'name': 'Harvest / Management', 'size': 1.3}, {'name': 'Industry', 'size': 14.3}, {'name':'Landfills', 'size': 1.7}, {'name': 'Livestock and Manure', 'size': 5.3999999999999995}, {'name': 'Other Agriculture', 'size': 1.7}, {'name': 'Other Fuel Combustion', 'size': 9.1}, {'name': 'Rice Cultivation', 'size': 1.5}, {'name': 'Transportation', 'size': 14.7}, {'name': 'Waste water - Other Waste', 'size': 1.5}, {'name': 'dummy Aluminium Non-Ferrous Metals401', 'size': 0.4}, {'name': 'dummy Cement411', 'size': 2.8}, {'name': 'dummy Chemicals421', 'size': 1.4}, {'name': 'dummy Other Industry431', 'size': 0.5}],
    [{'name': 'Agricultural Energy Use', 'size': 1.4}, {'name': 'Air', 'size': 1.7}, {'name': 'Aluminium Non-Ferrous Metals','size': 1.2}, {'name': 'Cement', 'size': 5.0}, {'name': 'Chemicals', 'size': 4.1}, {'name': 'Coal Mining', 'size': 1.3}, {'name': 'Commercial Buildings', 'size': 6.3}, {'name': 'Food and Tobacco', 'size': 1.0}, {'name': 'Iron and Steel', 'size': 4.0}, {'name': 'Machinery', 'size': 1.0}, {'name': 'Oil and Gas Processing', 'size': 6.4}, {'name': 'Other Industry', 'size': 7.0}, {'name': 'Pulp - Paper and Printing', 'size': 1.1}, {'name': 'Rail - Ship and Other Transport', 'size': 2.5}, {'name': 'Residential Buildings', 'size': 10.2}, {'name': 'Road', 'size': 10.5}, {'name': 'T and D Losses', 'size': 2.2}, {'name': 'Unallocated Fuel Combustion', 'size': 3.8}, {'name': 'dummy Carbon Dioxide162', 'size': 10.9}, {'name': 'dummy Carbon Dioxide392', 'size': 1.3}, {'name': 'dummy Methane552', 'size': 1.7}, {'name': 'dummy Methane562', 'size': 5.1}, {'name': 'dummy Methane612','size': 1.4}, {'name': 'dummy Methane722', 'size': 1.5}, {'name': 'dummy Methane832', 'size': 1.2}, {'name': 'dummy Nitrous Oxide52', 'size': 5.2}, {'name': 'dummy Nitrous Oxide572', 'size': 0.3}, {'name': 'dummy Nitrous Oxide622', 'size':0.3}, {'name': 'dummy Nitrous Oxide842', 'size': 0.3}], 
    [{'name': 'Carbon Dioxide', 'size': 76.80000000000001}, {'name': 'HFCs - PFCs', 'size': 1.1}, {'name': 'Methane', 'size': 15.299999999999999}, {'name': 'Nitrous Oxide', 'size': 6.7}]
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

    for i in [1, 2]:
        calculateNodePos(i)
        updateNodeOrder(i)
        getNodePos(i)
        updateLinkPos(i-1, 'target')
        updateLinkPos(i, 'source')
        
    for i in [3]:
        calculateNodePos(i, isRight=True)
        updateNodeOrder(i)
        getNodePos(i)
        updateLinkPos(i-1, 'target')

    for i in [2,1]:
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