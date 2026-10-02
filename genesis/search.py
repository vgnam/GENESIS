"""GENESIS search loop: Bayesian MCTS over LLM-generated heuristics with AGRE refinement."""
import copy
import json
import logging
import random

from . import population
from .interface import InterfaceEC
from .llm import LLMClient
from .problem import Problem
from .tree import OPERATORS, AblationConfig, GenesisTree


class Genesis:
    def __init__(self, cfg, root_dir, workdir):
        self.cfg = cfg
        self.problem = Problem(cfg, root_dir)
        self.output_path = f"{workdir}/"

        self.llm_model_names = [cfg.model]
        self.init_size = cfg.init_pop_size
        self.pop_size = cfg.pop_size
        self.fe_max = cfg.max_fe
        self.n_children = cfg.n_children
        self.max_depth = cfg.max_depth
        self.m = 5
        self.eval_times = 0

        abl = cfg.get("ablation") or {}
        self.ablation = AblationConfig(
            posterior_sampling=bool(abl.get("posterior_sampling", True)),
            adaptive_op_selection=bool(abl.get("adaptive_op_selection", True)),
            global_sharing=bool(abl.get("global_sharing", True)),
            ucb_c=float(abl.get("ucb_c", 1.0)),
        )
        self.use_agre = bool(abl.get("agre", True))
        self.agre_stages = dict(abl.get("agre_stages") or {})

        logging.info(f"LLM: {self.llm_model_names}, operators: {OPERATORS}")
        logging.info(f"Ablation: {self.ablation}, AGRE: {self.use_agre}, AGRE stages: {self.agre_stages}")
        random.seed(cfg.seed)

        self.interface_ecs = {
            name: InterfaceEC(self.m, name, self.problem, LLMClient(name, cfg.temperature), self.agre_stages)
            for name in self.llm_model_names
        }

    @staticmethod
    def add2pop(pop, offspring):
        """Add ``offspring`` unless a heuristic with the same description is already in ``pop``."""
        if any(ind.get('algorithm') == offspring['algorithm'] for ind in pop):
            return False
        pop.append(offspring)
        return True

    def _path_set(self, tree, cur_node, option, nodes_set):
        """Heuristics given to the operator as context (``None`` = skip this expansion)."""
        if option == 's1':
            path = []
            node = cur_node
            while node.code != "Root":
                path.append(copy.deepcopy(node.raw_info))
                node = node.parent
            path = population.population_management_s1(path, len(path))
            return path if len(path) > 1 else None
        if option == 'e1':
            return [copy.deepcopy(random.choice(child.children).raw_info) if child.children else child.raw_info
                    for child in tree.root.children]
        return nodes_set

    def expand(self, tree, cur_node, nodes_set, option, model_name):
        """
        Apply ``option`` to ``cur_node`` ``n_children`` times, refine the best child with AGRE,
        and add every valid new heuristic to the tree.
        """
        interface_ec = self.interface_ecs[model_name]

        path_set = self._path_set(tree, cur_node, option, nodes_set)
        if path_set is None:
            return nodes_set

        offsprings = []
        for _ in range(self.n_children):
            self.eval_times, offspring = interface_ec.evolve_algorithm(
                self.eval_times, path_set, cur_node.raw_info, option)
            if offspring is None:
                continue
            offsprings.append(offspring)
            father = "Root" if cur_node.is_root else cur_node.raw_info['objective']
            logging.info(f"Action: {option}, Father Obj: {father}, Candidate Obj: {offspring['objective']}, "
                         f"Depth: {cur_node.depth + 1}")

        if not offsprings:
            return nodes_set

        if self.use_agre:
            best_offspring = min(offsprings, key=lambda x: float(x['objective']))
            try:
                _, refined = interface_ec.get_offspring(path_set, "refine", father=best_offspring)
                self.eval_times += 1
                refined['objective'] = interface_ec.evaluate(refined['code'])
                if refined['objective'] is not None:
                    offsprings.append(refined)
                    logging.info(f"AGRE: Best Obj: {best_offspring['objective']}, Refined Obj: {refined['objective']}")
            except Exception as e:
                logging.info(f"AGRE refinement failed: {e}")

        for offspring in offsprings:
            if offspring['objective'] == float('inf'):
                continue
            if self.add2pop(nodes_set, offspring):
                nodes_set = population.population_management(nodes_set, min(len(nodes_set), self.pop_size))
                new_node = tree.new_node(offspring, cur_node, model_name)
                cur_node.add_child(new_node)
                cur_node.children_info.append(offspring)
                tree.backpropagate(new_node, op_name=option)

        return nodes_set

    def _add_initial(self, tree, offspring, model_name, op_name):
        node = tree.new_node(offspring, tree.root, model_name)
        tree.root.add_child(node)
        tree.root.children_info.append(offspring)
        tree.backpropagate(node, op_name)

    def initialize(self, tree):
        """One heuristic from scratch (i1), then ``init_size - 1`` diverse ones (e1) per LLM."""
        brothers = []
        for model_name, interface_ec in self.interface_ecs.items():
            self.eval_times, offspring = interface_ec.get_algorithm(self.eval_times, [], "i1")
            brothers = [offspring]
            self._add_initial(tree, offspring, model_name, "i1")

        seeds = list(brothers)
        additional = []
        for _ in range(1, self.init_size):
            for model_name, interface_ec in self.interface_ecs.items():
                self.eval_times, offspring = interface_ec.get_algorithm(self.eval_times, seeds, "e1")
                additional.append((offspring, model_name))

        for offspring, model_name in additional:
            brothers.append(offspring)
            self._add_initial(tree, offspring, model_name, "e1")

        return population.population_management(brothers, min(len(brothers), self.pop_size))

    def run(self):
        logging.info("- Initialization Start -")
        tree = GenesisTree(self.llm_model_names, max_depth=self.max_depth, ablation=self.ablation)
        nodes_set = self.initialize(tree)
        logging.info("- Initialization Finished - Evolution Start -")

        filename = None
        while self.eval_times < self.fe_max:
            logging.info(f"Current objectives in tree: {[round(float(x), 2) for x in tree.rank_list]}")

            target_node, _, (model_name, op) = tree.select_expansion_target(fe=self.eval_times)
            logging.info(f"Iter: {self.eval_times}/{self.fe_max}, Operator: {op}, LLM: {model_name}")

            nodes_set = self.expand(tree, target_node, nodes_set, op, model_name)
            nodes_set = population.population_management(nodes_set, min(len(nodes_set), self.pop_size))

            with open(f"{self.output_path}population_generation_{self.eval_times}.json", 'w') as f:
                json.dump(nodes_set, f, indent=5)
            filename = f"{self.output_path}best_population_generation_{self.eval_times}.json"
            with open(filename, 'w') as f:
                json.dump(nodes_set[0], f, indent=5)

        return nodes_set[0]["code"], filename
