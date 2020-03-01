# Crossing Reduction of Sankey Diagram with Barycentre Ordering via Markov Chain

by He Chen LI, Shi Ying LI, Bo Wen TAN and Shuai Cheng LI.

This project contains the source code for the above paper. The paper has been submitted to the [46th International Workshop on Graph-Theoretic Concepts in Computer Science](https://algorithms.leeds.ac.uk/wg2020).

## Requirements

This project uses **Python 3.7**. We uses the following packages (click for installation guide):

* [numpy and matplotlib](https://scipy.org/install.html);

* [pulp](https://pythonhosted.org/PuLP/main/installing_pulp_at_home.html) (for implementing the Integer Linear Programming method in paper ["Optimal Sankey Diagrams Via Integer Programming"](https://ieeexplore.ieee.org/abstract/document/8365985)).

## Test Cases Reproduction

### **Test against the state-of-art heuristic combined method**

#### Dataset

The dataset used in this test is the same one used in the paper ["EnergyViz: an interactive system for visualization of energy systems"](https://link.springer.com/article/10.1007/s00371-015-1186-8) where the combined method is introduced. It is the the Canada energy flow data in 1978 from [Canadian Energy Systems Analysis Research](https://www.cesarnet.ca/) website. However, the dataset is no longer available as shown in [this page](https://www.cesarnet.ca/visualization/sankey-diagrams-canadas-energy-systems?scope=Canada&year=1978&hide=all&modifier=none#chart-form). As a result, we build an artificial dataset based on the combined method's result figure. The dataset is a json file available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/input/heur_case.json) in the **input** folder.

#### Source code

The source code of this test is available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/algorithm/heur_case.py) in the **algorithm** folder.

#### Output file

The output file of this test is available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/output/heur_case.txt) in the **output** folder. It contains the ordering as well as the non-weighted and weighted crossing of results from the combined method and both stages of our method.

#### Visualization

Visualizations of these output orderings from the [combined method](https://gitlab.deepomics.org/lhc/sankey/blob/master/visualization/heur_case_comb_result.html), the [Markov stage](https://gitlab.deepomics.org/lhc/sankey/blob/master/visualization/heur_case_stage1_result.html) and the [Refinement stage](https://gitlab.deepomics.org/lhc/sankey/blob/master/visualization/heur_case_stage2_result.html) are available in the **visualization** folder.

### **Test against the ILP and BC method**

#### Dataset

The dataset used in this test is the same one used in the paper ["Optimal Sankey Diagrams Via Integer Programming"](https://ieeexplore.ieee.org/document/8365985) where the ILP method is introduced. It is the ["World Greenhouse Gas Emissions"](http://pdf.wri.org/working_papers/world_greenhouse_gas_emissions_2005.pdf) data from the World Resource Institute. This dataset is a json file available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/input/ilp_case.json) in the **input** folder.

#### Source code

The source code of this test is available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/algorithm/ilp_case.py) in the **algorithm** folder.

#### Output file

The output file of this test is available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/output/ilp_case.txt) in the **output** folder. It contains the ordering as well as the non-weighted and weighted crossing of results from the BC method, the ILP method and both stages of our method.

#### Visualization

Visualizations of these output orderings from the [BC method](https://gitlab.deepomics.org/lhc/sankey/blob/master/visualization/ilp_case_BC_result.html), the [ILP method](https://gitlab.deepomics.org/lhc/sankey/blob/master/visualization/ilp_case_ILP_result.html), the [Markov stage](https://gitlab.deepomics.org/lhc/sankey/blob/master/visualization/ilp_case_Stage1_result.html) and the [Refinement stage](https://gitlab.deepomics.org/lhc/sankey/blob/master/visualization/ilp_case_Stage2_result.html) are available in the **visualization** folder.

### **Test on the circular form of the Sankey diagram**

#### Source code

The source code of this test is available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/algorithm/cycle.py) in the **algorithm** folder. It also contains code that generated the 20 test cases on circular Sankey diagram.

#### Output files

The output files of this test are two figures, respectively the [dumbbell plot](https://gitlab.deepomics.org/lhc/sankey/blob/master/output/dumbbell.png) and the [boxplot](https://gitlab.deepomics.org/lhc/sankey/blob/master/output/box.png), available in the **output** folder.

### **Robust Test**

#### Dataset

All test cases are generated by a python script available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/algorithm/robust_test_case_generator.py) in the **algorithm** folder. The generated test cases are summarized in the [robust](https://gitlab.deepomics.org/lhc/sankey/tree/master/input/robust) folder within the **input** folder. The script also returns a [json file](https://gitlab.deepomics.org/lhc/sankey/blob/master/input/robust/caseInfo.json) containing the average edge number in each 10 test cases associated with a pair of (n, V).

#### Source code

We implemented the ILP method as a python script using the pulp package [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/algorithm/robust_test_ilp.py) in the **algorithm** folder. And the source code of our method is available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/algorithm/robust_test_method.py) in the **algorithm** folder.

#### Output files

The output files of the ILP script and our method's script are json files available [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/output/robust_ilp_result.json) and [here](https://gitlab.deepomics.org/lhc/sankey/blob/master/output/robust_method_result.json) in the **output** folder. The figure given in our paper is written in JavaScript based on this [summarized result file](https://gitlab.deepomics.org/lhc/sankey/blob/master/output/robust_test_result_summarized.json).
