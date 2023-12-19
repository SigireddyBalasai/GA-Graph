from graph import create_random_graph, assign_states, to_useful,create_final_graph
from model import create_model
from crossover import crossover
from mutate import mutate_dag
import networkx as nx
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

class Individual:
    def __init__(self, input_size, output_size, states, nodes, edges):
        self.input_size = input_size
        self.output_size = output_size
        self.states = states
        self.nodes = nodes
        self.edges = edges
        self.graph = assign_states(self.create_random_graph(), self.states)
        self.normalized = to_useful(self.graph)
        self.model = create_model(self.normalized, self.input_size, self.output_size)
        self.score = 0

    def create_random_graph(self):
        graph = nx.DiGraph()
        graph1 = create_random_graph(self.nodes, self.edges)
        graph2 = create_random_graph(self.nodes, self.edges)
        graph = create_final_graph(graph1,graph2)
        return graph

    def evaluate(self, train_ds):
        accuracies = []
        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=tf.keras.metrics.CategoricalAccuracy())
        for i in train_ds:
            #x = tf.expand_dims(i[0], axis=0)
            #y = tf.expand_dims(i[1], axis=0)
            #print(x.shape, y.shape, "x.shape, y.shape")
            loss,accuracy = self.model.evaluate(i[0], i[1], verbose=0)
            accuracies.append(accuracy)

        accuracies = np.array(accuracies)
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        score = mean_accuracy / std_accuracy
        self.score = score

    def mutate(self):
        self.graph = mutate_dag(self.graph)
        self.normalized = to_useful(self.graph)
        self.model = create_model(self.normalized, self.input_size, self.output_size)
        return self

    def crossover(self, other):
        self.graph = crossover(self.graph, other.graph)
        self.normalized = to_useful(self.graph)
        self.model = create_model(self.normalized, self.input_size, self.output_size)
        return self

    def get_model(self):
        return self.model

    def get_graph(self):
        plt.figure(figsize=(10, 10))
        nx.draw(self.graph, with_labels=True)
        plt.draw()
        plt.show()


    def save_model(self, folder):
        model = tf.keras.models.clone_model(self.model)
        # save architecture as image and model as h5 file'
        print("saving model")
        score = self.score
        print(score, "score")
        tf.keras.utils.plot_model(self.model, to_file=f'{folder}/{self.score}.png', show_shapes=True)
        print(f"image saved in {folder}/{self.score}.png")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.save(f'{folder}/{self.score}')
        self.model = model
    
    def get_score(self):
        return self.score