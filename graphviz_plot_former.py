
import graphviz

dot = graphviz.Digraph('former', comment='The Round Table') # round-table is the file name

dot.node('l', 'Lensed Image')
dot.node('s', 'Source Image')
dot.node('g', 'Generator')
dot.node('d', 'Discriminator')
dot.node('f', 'Fake Image')
dot.node('o', 'Output')
dot.node('c','concatenate')

dot.edge('l','g')
dot.edge('g','f')
dot.edges(['sc','fc'])
dot.edges(['cd','ds','dl','do'])

dot.render(directory='doctest-output', view=True)

