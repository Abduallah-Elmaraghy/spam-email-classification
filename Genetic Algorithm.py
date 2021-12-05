import pandas as pd
import random

from Levenshtein import distance

SEED = 2021

random.seed = SEED

dataset= pd.read_csv('spambase.csv')  

#df = pd.read_csv('bach_choral_set_dataset.csv')
dataset

TEST_POP_SIZE = 100
INIT_POP_SIZE = 500
MUTATIONS = 50
GENERATIONS = 100

total_num_of_instances = len(dataset)
print(f"total_num_of_instances: {total_num_of_instances}")

# Sample an initial population from the complete training set.
initial_population = dataset["class"].sample(INIT_POP_SIZE, random_state=SEED).reset_index(drop=True)

# Simplest possible Genetic Algorithm helper methods.

def cross_over(parent1, parent2):
    """ Splice two parents together at a random point to generate a child. """
    min_len = min(len(parent1), len(parent2))
    splice_idx = random.randint(0, min_len)
    child = parent1[:splice_idx] + parent2[splice_idx:]
    return child
    
def mutate(individual):
    """ Mutate an individual by swapping two characters at a random point."""
    mutation_idx = random.randint(0, len(individual)-2)
    return individual[:mutation_idx] + individual[mutation_idx+1] + individual[mutation_idx] + individual[mutation_idx+2:]

def select_best(population, n_best):
    """ Score the given population against a sample from the total training population. """
    test_population = dataset["class"].sample(TEST_POP_SIZE, random_state=SEED).reset_index(drop=True)
    scores = []
    for p in population:
        score = 0
        for t in test_population:
            score += distance(p,t)
        score = score/TEST_POP_SIZE
        scores.append(score)
    sorted_population = [p for _, p in sorted(zip(scores, population))]
    return sorted_population[:n_best], min(scores)

# Test cross_over and mutate on some simple srtings.
print(cross_over("abcdef", "vwxyz"))
print(cross_over("abcdef", "vwxyz"))
print(cross_over("abcdef", "vwxyz"))
print(cross_over("vwxyz", "abcdef"))
print(cross_over("vwxyz", "abcdef"))
print(cross_over("vwxyz", "abcdef"))

print(mutate("abcdef"))
print(mutate("abcdef"))
print(mutate("abcdef"))
print(mutate("abcdef"))

# What is the fittest member of the population to start with?
select_best(initial_population, 1)

def genetic_algorithm():
    """ This is just meant to be a simple naive baseline. No need to make it more complex that it needs to be. """
    population = initial_population

    for gen in range(GENERATIONS):
        # Select the top half of the population and then fill back to the original population limit by making children.
        population, _ = select_best(population, INIT_POP_SIZE//2)
        children = []
        for child in range(INIT_POP_SIZE//2):
            children.append(cross_over(population[random.randint(0, INIT_POP_SIZE//2 - 1)], population[random.randint(0, INIT_POP_SIZE//2 - 1)]))
        population.extend(children)
        
        # Add some mutations.
        for m in range(MUTATIONS):
            mutant = random.randint(0, INIT_POP_SIZE - 1)
            population[mutant] = mutate(population[mutant])
        
        # Report on the best so far.
        _, best_score = select_best(population, 1)        
        print(f"Generation: {gen} : {best_score}")
    
    return select_best(population, 1)
best_string, score = genetic_algorithm()

# What single string did we generate?
print(best_string[0], score)
'''
# Build the submission. The csv is huge and very repitative. .gz files can be submitted directly, so let's compress it.
subm = pd.read_csv('master.csv')
subm['sex'] = best_string[0]
subm.to_csv('submission.csv.gz', compression="gzip", index=False)'''