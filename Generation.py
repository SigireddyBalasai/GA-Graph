from Individual import Individual
import random

class Generation:
    def __init__(self,input_size,output_size,states,nodes,edges,population,limit,X,y):
        self.input_size = input_size
        self.output_size = output_size
        self.states = states
        self.nodes = nodes
        self.edges = edges
        self.population = population
        self.limit = limit
        self.generation = 0
        self.create_population(self.input_size, self.output_size, self.states, self.nodes, self.edges)
        self.score_population(X, y)

    def get_generation(self):
        return self.generation
    
    def get_population(self):
        return self.population

    def set_generation(self, generation):
        self.generation = generation

    def set_population(self, population):
        self.population = population

    def create_population(self, input_size, output_size, states, nodes, edges):
        population = []
        for i in range(self.population):
            population.append(Individual(input_size, output_size, states, nodes, edges))
        self.population = population
    
    def score_population(self, X, y):
        for individual in self.population:
            individual.score(X, y)
        self.population.sort(key=lambda x: x.get_score(), reverse=True)
        return self.population
    
    def get_best_individual(self):
        return self.population[0]

    def mutate_population(self,mutation_rate):
        mutated = []
        for individual in self.population:
            if random.random() < mutation_rate:
                mutated.append(individual.mutate())
        return self.population + mutated
    
    def crossover_population(self, crossover_rate):
        crossed = []
        for i in range(len(self.population)):
            if random.random() < crossover_rate:
                crossed.append(self.population[i].crossover(self.population[i+1]))
        self.population += crossed
        return self.population

    def next_generation(self, X, y, mutation_rate, crossover_rate):
        self.score_population(X, y)
        self.mutate_population(mutation_rate)
        self.crossover_population(crossover_rate)
        self.generation += 1
        self.population = self.population[:self.limit]
        return self.population

    def run(self,n,X,y,mutation_rate,crossover_rate):
        for i in range(n):

            print(f'Generation: {self.generation}')
            print(f'Best score: {self.get_best_individual().get_score()}')
            print(f'Best parameters: {self.get_best_individual().get_num_parameters()}')
            print(f'Best loss: {self.get_best_individual().get_loss()}')
            print(f'Best accuracy: {self.get_best_individual().get_accuracy()}')
            print(f'Best history: {self.get_best_individual().get_history()}')
            print(f'Best model: {self.get_best_individual().get_model()}')
            print(f'Best graph: {self.get_best_individual().get_graph()}')
            self.next_generation(X, y, mutation_rate, crossover_rate)
        return self.population