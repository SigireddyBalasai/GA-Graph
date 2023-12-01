from graph import create_random_graph, assign_states, to_useful
from model import create_model
from crossover import crossover
from mutate import mutate_dag
import networkx as nx
from matplotlib import pyplot as plt

class Individual:
    def __init__(self, input_size, output_size, states, nodes, edges):
        self.input_size = input_size
        self.output_size = output_size
        self.states = states
        self.nodes = nodes
        self.edges = edges
        self.score_ = 0
        self.loss = 0
        self.accuracy = 0
        self.num_parameters = 0
        self.history = 0
        self.graph = assign_states(self.create_random_graph(),self.states)
        nx.draw(self.graph, with_labels=True)
        plt.show()
        self.normalized = to_useful(self.graph, self.states)
        nx.draw(self.normalized, with_labels=True)
        plt.show()
        self.model = create_model(self.normalized, self.input_size, self.output_size)


    def create_random_graph(self):
        graph = nx.DiGraph()
        graph = create_random_graph(self.nodes, self.edges)
        return graph

    def score(self, X, y):
        from sklearn import model_selection
        from sklearn import preprocessing
        from sklearn import metrics
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        num_parameters = self.model.count_params()
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(x_train, y_train, epochs=5, batch_size=10)
        loss,accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        score = metrics.roc_auc_score(y_test, self.model.predict(x_test))
        self.score_ = score
        self.loss = loss
        self.accuracy = accuracy
        self.num_parameters = num_parameters
        self.history = history
        return score,loss,accuracy,num_parameters,history

    def mutate(self):
        self.graph = mutate_dag(self.graph)
        self.normalized = to_useful(self.graph, self.states)
        self.model = create_model(self.normalized, self.input_size, self.output_size)
        return self

    def crossover(self, other):
        self.graph = crossover(self.graph, other.graph)
        self.normalized = to_useful(self.graph, self.states)
        self.model = create_model(self.normalized, self.input_size, self.output_size)
        return self
    
    def get_score(self):
        return self.score_

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy

    def get_num_parameters(self):
        return self.num_parameters

    def get_history(self):
        return self.history

    def get_model(self):
        return self.model

    def get_graph(self):
        plt.figure(figsize=(10,10))
        nx.draw(self.graph, with_labels=True)
        plt.show()
    

    


    
    