import networkx as nx
import random

def mutate_dag(dag):
    choices = random.choices(['add_node', 'add_edge', 'remove_node', 'remove_edge'], weights=[0.25, 0.25, 0.25, 0.25])[0]
    print(choices)
    if choices == 'add_node':
        edge = random.choice(list(dag.edges))
        node1 , node2 = edge
        node_ = random.choices(['Dense', 'DropOut', 'Conv2D', 'MaxPooling2D', 'AveragePooling2D'],weights=[0.99, 0.01, 0.3, 0.2, 0.2])[0]
        dag.add_node(dag.number_of_nodes())
        dag.nodes[dag.number_of_nodes() - 1]['state'] = 'hidden'
        if node_ == 'Dense':
            dag.nodes[dag.number_of_nodes() - 1]['type'] = 'Dense'
            dag.nodes[dag.number_of_nodes() - 1]['units'] = random.randint(1, 128)
            dag.nodes[dag.number_of_nodes() - 1]['activation'] = random.choice(['relu', 'sigmoid', 'tanh'])
        elif node_ == 'DropOut':
            dag.nodes[dag.number_of_nodes() - 1]['type'] = 'DropOut'
            dag.nodes[dag.number_of_nodes() - 1]['rate'] = random.uniform(0.0, 1.0)
        elif node_ == 'Conv2D':
            dag.nodes[dag.number_of_nodes() - 1]['type'] = 'Conv2D'
            dag.nodes[dag.number_of_nodes() - 1]['filters'] = random.randint(1, 128)
            dag.nodes[dag.number_of_nodes() - 1]['kernel_size'] = random.randint(1, 10)
            dag.nodes[dag.number_of_nodes() - 1]['activation'] = random.choice(['relu', 'sigmoid', 'tanh'])
        elif node_ == 'MaxPooling2D':
            dag.nodes[dag.number_of_nodes() - 1]['type'] = 'MaxPooling2D'
            dag.nodes[dag.number_of_nodes() - 1]['pool_size'] = random.randint(1, 10)
        elif node_ == 'AveragePooling2D':
            dag.nodes[dag.number_of_nodes() - 1]['type'] = 'AveragePooling2D'
            dag.nodes[dag.number_of_nodes() - 1]['pool_size'] = random.randint(1, 10)
        #print(node_)
        #print(dag.number_of_nodes() - 1)
        #print(node1, dag.number_of_nodes() - 1)
        #print(dag.number_of_nodes() - 1, node2)
        dag.add_edge(node1, dag.number_of_nodes() - 1)
        dag.add_edge(dag.number_of_nodes() - 1, node2)
    elif choices == 'add_edge':
        #print('add_edge')
        node1, node2 = random.choices(list(dag.nodes), k=2)
        #print(node1, node2)
        dag.add_edge(node1, node2)
    elif choices == 'remove_node':
        node = random.choice(list(dag.nodes))
        #print(node)
        dag.remove_node(node)
    elif choices == 'remove_edge':
        edge = random.choice(list(dag.edges))
        #print(edge)
        dag.remove_edge(edge[0], edge[1])
    return dag