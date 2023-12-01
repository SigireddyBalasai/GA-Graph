import networkx as nx
import random

def mutate_dag(dag):
    try:
        if random.random() < 0.5:
            start_node = random.choice(list(dag.nodes))
            end_node = random.choice(list(dag.nodes - set(nx.descendants(dag, start_node))))
            path = nx.shortest_path(dag, start_node, end_node)
            dag.add_edges_from(zip(path[:-1], path[1:]))

        else:
            paths = list(nx.all_simple_paths(dag, source=random.choice(list(dag.nodes)), target=random.choice(list(dag.nodes))))
            if paths:
                path_to_remove = random.choice(paths)
                dag.remove_edges_from(zip(path_to_remove[:-1], path_to_remove[1:]))

        return dag
    except:
        return dag