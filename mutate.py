import networkx as nx
import random

def mutate_dag(dag):
    choices = random.choices(['add_node', 'add_edge', 'remove_node', 'remove_edge'], weights=[0.25, 0.25, 0.25, 0.25])
    if choices == 'add_node':
        edge = random.choice(list(dag.edges))
        node1 , node2 = edge
        node_ = random.choice(['Dense', 'DropOut', 'Conv2D', 'MaxPooling2D', 'AveragePooling2D'])
        if(node_ == 'Dense'):
            dag.add_node(dag.number_of_nodes(), state='hidden', layer=node_, units=random.randint(1, 100), activation=random.choice(['relu', 'sigmoid', 'softmax', 'tanh']))
        elif(node_ == 'DropOut'):
            dag.add_node(dag.number_of_nodes(), state='hidden', layer=node_, rate=random.uniform(0, 1))
        elif(node_ == 'Conv2D'):
            dag.add_node(dag.number_of_nodes(), state='hidden', layer=node_, filters=random.randint(1, 100), kernel_size=random.randint(1, 10), activation=random.choice(['relu', 'sigmoid', 'softmax', 'tanh']))
        elif(node_ == 'MaxPooling2D'):
            dag.add_node(dag.number_of_nodes(), state='hidden', layer=node_, pool_size=random.randint(1, 10))
        elif(node_ == 'AveragePooling2D'):
            dag.add_node(dag.number_of_nodes(), state='hidden', layer=node_, pool_size=random.randint(1, 10))
        #add edge between node1 and node2 and node
        dag.add_edge(node1, node_)
        dag.add_edge(node_, node2)
    elif choices == 'add_edge':
        node1 = random.choice(list(dag.nodes))
        node2 = random.choice(list(dag.nodes))
        if node1 != node2:
            dag.add_edge(node1, node2)
    elif choices == 'remove_node':
        node = random.choice(list(dag.nodes))
        dag.remove_node(node)
    elif choices == 'remove_edge':
        edge = random.choice(list(dag.edges))
        dag.remove_edge(edge[0], edge[1])
    return dag