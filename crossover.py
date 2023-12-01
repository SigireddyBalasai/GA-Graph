import random
import networkx as nx

def crossover(parent1, parent2):
    """Return a new chromosome created via crossover."""
    crossover_node = random.choice(list(parent1.nodes))
    child = nx.DiGraph()
    for node in nx.topological_sort(parent1):
        child.add_node(node, **parent1.nodes[node])
        if node == crossover_node:
            break
        child.add_edges_from(parent1.out_edges(node, data=True))
    for node in nx.topological_sort(parent2):
        if node == crossover_node:
            continue
        child.add_node(node, **parent2.nodes[node])
        child.add_edges_from(parent2.out_edges(node, data=True))

    return child