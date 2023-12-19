import networkx as nx
import igraph as ig
import random
import numpy as np
import numpy as np
import copy
from model import create_model


def create_random_graph(nodes:int,edges:int)->nx.DiGraph:
    graph = ig.Graph.Erdos_Renyi(n=nodes, p=edges, directed=False, loops=False)
    graph.to_directed(mode='acyclic')
    graph = graph.to_networkx()
    return graph

def create_final_graph(graph1, graph2):
    final_graph = nx.union(graph1, graph2, rename=('1-', '2-'))
    nodes_1 = [node for node in final_graph.nodes() if node.startswith('1-')]
    nodes_2 = [node for node in final_graph.nodes() if node.startswith('2-')]
    final_1 = max(nodes_1)
    final_2 = min(nodes_2)
   #print(final_1, final_2)
    final_graph.add_edge(final_1, final_2)
   #print(final_graph)
    return final_graph

def assign_states(graph:nx.DiGraph, states: tuple[int, int,int,int, int, int]):
    # assign input to states[0] nodes and hidden to states[1] nodes and output to states[2] nodes
    # states: (input, hidden, output, other)
    # graph: networkx graph
    # return: networkx graph
    states = np.cumsum(states).tolist()
   #print(states)
   #print(graph.nodes())
    for i in range(states[0]):
        graph.nodes['1-' + str(i)]['state'] = 'input'
    for i in range(states[0],states[1]):
        graph.nodes['1-' + str(i)]['state'] = 'hidden_1'
        choice = random.choices(['Conv2D', 'MaxPooling2D', 'AveragePooling2D', 'GlobalAveragePooling2D', 'DropOut','Conv2DTranspose'], weights=[0.9, 0.03, 0.03, 0.005, 0.03, 0.005], k=1)[0]
        graph.nodes['1-' + str(i)]['layer'] = choice
        if choice == 'Conv2D':
            graph.nodes['1-' + str(i)]['filters'] = random.randint(1, 16)
            kernel_size = random.randint(1, 64)
            graph.nodes['1-' + str(i)]['kernel_size'] = (kernel_size, kernel_size)  # Ensure both dimensions are the same
            graph.nodes['1-' + str(i)]['activation'] = random.choice(['relu','sigmoid','softmax','tanh'])
        elif choice == 'MaxPooling2D':
            graph.nodes['1-' + str(i)]['pool_size'] = random.randint(1, 16)
        elif choice == 'AveragePooling2D':
            graph.nodes['1-' + str(i)]['pool_size'] = random.randint(1, 16)
        elif choice == 'GlobalAveragePooling2D':
            pass
        elif choice == 'DropOut':
            graph.nodes['1-' + str(i)]['rate'] = random.uniform(0, 1)
        elif choice == 'Conv2DTranspose':
            graph.nodes['1-' + str(i)]['filters'] = random.randint(1, 4)
            kernel_size = random.randint(1, 64)
            graph.nodes['1-' + str(i)]['kernel_size'] = (kernel_size, kernel_size)  # Ensure both dimensions are the same
            graph.nodes['1-' + str(i)]['activation'] = random.choice(['relu','sigmoid','softmax','tanh'])
    for i in range(states[1],states[2]):
        graph.nodes['1-' + str(i)]['state'] = 'transition'
    for i in range(states[2],states[3]):
        graph.nodes['2-' + str(i - states[2])]['state'] = 'transition'
    for i in range(states[3],states[4]):
        graph.nodes['2-' + str(i - states[2])]['state'] = 'hidden_2'
        choice = random.choices(['Dense', 'LSTM'], weights=[0.75, 0.25], k=1)[0]
        graph.nodes['2-' + str(i - states[2])]['layer'] = choice
        if choice == 'Dense':
            graph.nodes['2-' + str(i - states[2])]['units'] = random.randint(1, 64)
            graph.nodes['2-' + str(i - states[2])]['activation'] = random.choice(['relu','sigmoid','softmax','tanh'])
        elif choice == 'LSTM':
            graph.nodes['2-' + str(i - states[2])]['units'] = random.randint(1, 64)
            graph.nodes['2-' + str(i - states[2])]['activation'] = random.choice(['relu','sigmoid','softmax','tanh'])
    for i in range(states[4],states[5]):
       #print(i - states[3],states[3],states[4],states[3] - states[4])
        graph.nodes['2-' + str(i - states[2])]['state'] = 'output'
    return graph


def to_useful(graph):
    # remove a node if no path to input or final node
    D = copy.deepcopy(graph)
    node_list = set(D.nodes())
    #print(node_list)
    for i in D.nodes():
        #print(i, D.nodes[i]['state'])
        pass
    node_remove = set()
    inputs = [node for node in D.nodes() if D.nodes[node]['state'] == 'input']
    transition = [node for node in D.nodes() if D.nodes[node]['state'] == 'transition']
    outputs = [node for node in D.nodes() if D.nodes[node]['state'] == 'output']
    hidden1 = [node for node in D.nodes() if D.nodes[node]['state'] == 'hidden_1']
    hidden2 = [node for node in D.nodes() if D.nodes[node]['state'] == 'hidden_2']
    for node_hidden in hidden1:
        for node_input in inputs:
            if not nx.has_path(D, node_input , node_hidden):
                node_remove.add(node_hidden)
        node_output = transition[0]
        if not nx.has_path(D, node_hidden, node_output):
            node_remove.add(node_hidden)
    D.remove_nodes_from(node_remove)
    node_remove = set()
    for node_hidden in hidden2:
        node_input = transition[1]
        if not nx.has_path(D, node_input , node_hidden):
            node_remove.add(node_hidden)
        for node_output in outputs:
            if not nx.has_path(D, node_hidden, node_output):
                node_remove.add(node_hidden)
    D.remove_nodes_from(node_remove)
    #print(D)
    return D


if __name__ == '__main__':
    g = create_random_graph(10, 0.5)
    g2 = create_random_graph(10, 0.5)
    g = create_final_graph(g, g2)
    g = assign_states(g, (1, 8, 1,1, 8, 1))
    for i in g.nodes():
        print(i, g.nodes[i]['state'])
    print(g.edges)
    g = to_useful(g)
    for i in g.nodes():
        print(i, g.nodes[i]['state'])
    print(g.edges)
    model = create_model(g, (28, 28, 1), 10)
