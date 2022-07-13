import graphviz

dot = graphviz.Digraph('G', filename='Latter.gv') # round-table is the file name
dot.attr(compound='true')

dot.node('l', 'Lensed Image')
dot.node('s', 'Source Image')
dot.node('g', 'Generator')
dot.node('d', 'Discriminator')
dot.node('f', 'Fake Image')
dot.node('w','Fake parameter')
dot.node('o', 'Output')
dot.node('c','concatenate')
dot.node('x','Concatenate image')
dot.node('z','concatenate parameter')
dot.node('p', 'Parameters')

dot.edges(['fx', 'sx'])
dot.edges(['wz', 'pz'])

with dot.subgraph(name='Generator') as generator:
    generator.edge('l','g')
    generator.edges(['gf', 'gw'])

with dot.subgraph(name="Discriminator") as discriminator:
    discriminator.edges(['xc', 'wc', 'zc'])
    discriminator.edge('c', 'd')
    discriminator.edges(['dl', 'ds', 'dp'])


dot.edge('d', 'o')

dot.render(directory='doctest-output', view=True)
