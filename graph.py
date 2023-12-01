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
        choice = random.choice(['Dense','DropOut',])
        graph.nodes[i]['layer'] = choice
        if choice == 'Dense':
            graph.nodes[i]['activation'] = random.choice(['relu','sigmoid','tanh','softmax'])
            graph.nodes[i]['units'] = random.randint(1, 100)
        elif choice == 'DropOut':
            graph.nodes[i]['rate'] = random.uniform(0, 1)
        
    for i in range(states[0]+states[1], states[0]+states[1]+states[2]):
        graph.nodes[i]['state'] = 'output'
    return graph

def to_useful(graph, states):
    # remove a node if no path to input or final node
    D = graph.copy()
    for i in range(states[0]):
        for j in range(states[0], states[0]+states[1]):
            print(i,j)
            try:
                print(nx.has_path(D, i, j))
                if nx.has_path(D, i, j):
                    continue
                else:
                    print('removing j')
                    D.remove_node(j)
            except:
                pass
    for i in range(states[0]+states[1], states[0]+states[1]+states[2]):
        for j in range(states[0], states[0]+states[1]):
            print(j,i)
            try:
                print(nx.has_path(D, j, i))
                if nx.has_path(D, j, i):
                    continue
                else:
                    print('removing j')
                    D.remove_node(j)
            except:
                pass
    return D


