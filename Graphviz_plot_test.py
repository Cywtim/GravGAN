"""

This is the file using graphviz to plot the flow chart of the neural network

"""

import numpy as np
import matplotlib.pyplot as plt
import graphviz

dot = graphviz.Digraph('round-table', comment='The Round Table') # round-table is the file name

dot.node('A', 'King Arthur') # create a node named 'King Arthur' labelled as 'A'
dot.node('B', 'Sir Bedevere') #
dot.node('L', 'Sir Lancelot')

dot.edges(['AB', 'AL']) # arrows from A point to B and L
dot.edge('B', 'L', constraint='false')

print(dot.source) # print the dot file of diagram

dot.render(directory='doctest-output', view=True) # output pdf figure