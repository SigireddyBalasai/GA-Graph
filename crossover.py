import random
import networkx as nx

def crossover(parent1,parent2):
    child1 = nx.DiGraph()
    child2 = nx.DiGraph()

    crosspoint1 = random.choice(list(parent1.nodes()))
    crosspoint2 = random.choice(list(parent2.nodes()))

    child1.add_nodes_from(parent1.nodes(data=True))
    child1.add_edges_from(parent1.edges(data=True))
    child2.add_nodes_from(parent2.nodes(data=True))
    child2.add_edges_from(parent2.edges(data=True))

    child1.remove_node(crosspoint1)
    child2.remove_node(crosspoint2)

    child1.add_node(crosspoint2, **parent2.nodes[crosspoint2])
    child2.add_node(crosspoint1, **parent1.nodes[crosspoint1])

    child1.add_edges_from(parent2.edges([crosspoint2], data=True))
    child2.add_edges_from(parent1.edges([crosspoint1], data=True))
    print(type(child1))
    print(type(child2))
    return child1, child2

