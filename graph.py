import networkx as nx
import igraph as ig
import random


def create_random_graph(nodes,edges):
    graph = ig.Graph.Erdos_Renyi(n=nodes, p=edges, directed=False, loops=False)
    graph.to_directed(mode='acyclic')
    graph = graph.to_networkx()
    return graph


def assign_states(graph, states):
    # assign input to states[0] nodes and hidden to states[1] nodes and output to states[2] nodes
    # states: [input, hidden, output]
    # graph: networkx graph
    # return: networkx graph
    for i in range(states[0]):
        graph.nodes[i]['state'] = 'input'
    for i in range(states[0], states[0]+states[1]):
        graph.nodes[i]['state'] = 'hidden'
        choice = random.choices(['Conv2D','MaxPooling2D','AveragePooling2D','GlobalAveragePooling2D',
                                 #'LocallyConnected2D',
                                 'DropOut','Dense']
                                ,k=1)[0]
        graph.nodes[i]['layer'] = choice
        if choice == 'DropOut':
            graph.nodes[i]['rate'] = random.uniform(0, 1)
        elif choice == 'Dense':
            graph.nodes[i]['units'] = random.randint(1, 16)
            graph.nodes[i]['activation'] = random.choice(['relu','sigmoid','softmax','tanh'])
        elif choice == 'Conv2D':
            graph.nodes[i]['filters'] = random.randint(1, 16)
            graph.nodes[i]['kernel_size'] = (random.randint(1, 16),)*2
            graph.nodes[i]['activation'] = random.choice(['relu','sigmoid','softmax','tanh'])
        elif choice == 'MaxPooling2D':
            graph.nodes[i]['pool_size'] = random.randint(1, 16)
        elif choice == 'AveragePooling2D':
            graph.nodes[i]['pool_size'] = random.randint(1, 16)
        elif choice == 'GlobalAveragePooling2D':
            pass
        elif choice == 'DropOut':
            graph.nodes[i]['rate'] = random.uniform(0, 1)
        elif choice == 'AveragePooling2D':
            graph.nodes[i]['pool_size'] = random.randint(1, 16)
        '''elif choice == 'LocallyConnected2D':
            graph.nodes[i]['filters'] = random.randint(1, 16)
            graph.nodes[i]['kernel_size'] = (random.randint(1, 16),)*2
            graph.nodes[i]['activation'] = random.choice(['relu','sigmoid','softmax','tanh'])'''
        
    for i in range(states[0]+states[1], states[0]+states[1]+states[2]):
        graph.nodes[i]['state'] = 'output'
    return graph

def to_useful(graph):
    # remove a node if no path to input or final node
    D = graph.copy()
    node_list = set(D.nodes())
    print(node_list)
    node_remove = set()
    inputs = [node for node in D.nodes() if D.nodes[node]['state'] == 'input']
    outputs = [node for node in D.nodes() if D.nodes[node]['state'] == 'output']
    hidden = [node for node in D.nodes() if D.nodes[node]['state'] == 'hidden']
    for node_hidden in hidden:
        for node_input in inputs:
            if not nx.has_path(D, node_input , node_hidden):
                node_remove.add(node_hidden)
        for node_output in outputs:
            if not nx.has_path(D, node_hidden, node_output):
                node_remove.add(node_hidden)
    D.remove_nodes_from(node_remove)
    return D


