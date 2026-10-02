"""Parent selection and elite-population management (objectives are minimised)."""
import heapq
import random


def parent_selection(pop, m):
    """Rank-based selection; ``pop`` is assumed sorted from best to worst."""
    probs = [1 / (rank + 1 + len(pop)) for rank in range(len(pop))]
    return random.choices(pop, weights=probs, k=m)


def parent_selection_e1(pop, m):
    """Uniform selection."""
    return random.choices(pop, k=m)


def population_management(pop_input, size):
    """Keep the ``size`` best individuals with distinct objective values."""
    pop = [ind for ind in pop_input if ind['objective'] is not None]
    unique_pop = []
    unique_objectives = []
    for ind in pop:
        if ind['objective'] not in unique_objectives:
            unique_pop.append(ind)
            unique_objectives.append(ind['objective'])
    return heapq.nsmallest(min(size, len(pop)), unique_pop, key=lambda x: x['objective'])


def population_management_s1(pop_input, size):
    """Keep up to ``size`` individuals with distinct descriptions, worst objective first (input of s1)."""
    pop = [ind for ind in pop_input if ind['objective'] is not None]
    unique_pop = []
    unique_algorithms = []
    for ind in pop:
        if ind['algorithm'] not in unique_algorithms:
            unique_pop.append(ind)
            unique_algorithms.append(ind['algorithm'])
    return heapq.nlargest(min(size, len(pop)), unique_pop, key=lambda x: x['objective'])
