"""Glue between the search, the LLM operators and the problem evaluator."""
import copy
import logging
import random

import numpy as np

from . import population
from .operators import Evolution


class InterfaceEC:
    def __init__(self, m, llm_model, problem, llm_client, agre_stages=None):
        self.interface_eval = problem
        self.evol = Evolution(llm_model, problem.prompts, llm_client, agre_stages=agre_stages)
        self.m = m  # maximum number of parents for e1

    @staticmethod
    def check_duplicate_obj(pop, obj):
        return any(obj == ind['objective'] for ind in pop)

    @staticmethod
    def check_duplicate(pop, code):
        return any(code == ind['code'] for ind in pop)

    def evaluate(self, code):
        """Evaluate one heuristic; returns its rounded objective, inf if it failed, or None on timeout."""
        objs = self.interface_eval.batch_evaluate([code], 0)
        if objs == 'timeout':
            return None
        if objs[0] == float('inf'):
            return float('inf')
        return np.round(objs[0], 5)

    def _get_alg(self, pop, operator, father=None):
        offspring = {
            'algorithm': None,
            'thought': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            offspring['code'], offspring['thought'] = self.evol.i1()
        elif operator == "e1":
            real_m = min(random.randint(2, self.m), len(pop))
            parents = population.parent_selection_e1(pop, real_m)
            offspring['code'], offspring['thought'] = self.evol.e1(parents)
        elif operator == "e2":
            other = copy.deepcopy(pop)
            if father in pop:
                other.remove(father)
            parents = population.parent_selection(other, 1)
            parents.append(father)
            offspring['code'], offspring['thought'] = self.evol.e2(parents)
        elif operator == "m1":
            parents = [father]
            offspring['code'], offspring['thought'] = self.evol.m1(father)
        elif operator == "m2":
            parents = [father]
            offspring['code'], offspring['thought'] = self.evol.m2(father)
        elif operator == "s1":
            parents = pop
            offspring['code'], offspring['thought'] = self.evol.s1(pop)
        elif operator == "counter":
            parents = pop
            offspring['code'], offspring['thought'] = self.evol.counter(pop[0])
        elif operator == "refine":
            parents = [father]
            offspring['code'], offspring['thought'] = self.evol.refine(father)
        else:
            raise NotImplementedError(f"Evolution operator [{operator}] has not been implemented!")

        offspring['algorithm'] = self.evol.post_thought(offspring['code'], offspring['thought'])
        return parents, offspring

    def get_offspring(self, pop, operator, father=None):
        """Generate one offspring, retrying on LLM/parsing errors and (once) on duplicated code."""
        while True:
            try:
                p, offspring = self._get_alg(pop, operator, father=father)
                if pop and self.check_duplicate(pop, offspring['code']):
                    logging.debug("Duplicated code, retrying ...")
                    p, offspring = self._get_alg(pop, operator, father=father)
                return p, offspring
            except Exception as e:
                logging.info(f"Offspring generation with [{operator}] failed: {e}")

    def get_algorithm(self, eval_times, pop, operator):
        """Generate and evaluate until a valid heuristic with a new objective value is found (initialisation)."""
        while True:
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator)
            obj = self.evaluate(offspring['code'])
            if obj is None or obj == float('inf') or self.check_duplicate_obj(pop, obj):
                continue
            offspring['objective'] = obj
            return eval_times, offspring

    def evolve_algorithm(self, eval_times, pop, node, operator):
        """Apply ``operator`` to ``node``; up to 3 attempts. Returns (eval_times, offspring or None)."""
        for _ in range(3):
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator, father=node)
            obj = self.evaluate(offspring['code'])
            if obj is None:
                return eval_times, None
            if obj == float('inf') or self.check_duplicate(pop, offspring['code']):
                continue
            offspring['objective'] = obj
            return eval_times, offspring
        return eval_times, None
