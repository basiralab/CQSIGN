import networkx as nx
import metis
metis.test()
G = metis.example_networkx()
(edgecuts, parts) = metis.part_graph(G, 3)