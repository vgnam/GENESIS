"""Bayesian search tree of GENESIS.

Two kinds of nodes are used:

* ``HeuristicNode`` ("real" node): a concrete, evaluated heuristic.
* ``OperatorNode`` ("GEN" node): one per (LLM, operator) pair attached to every
  heuristic node. It represents the intent to expand its parent with that
  operator and is never evaluated directly.

Every node keeps a Normal-Inverse-Gamma (NIG) posterior over the meta-objective
of its subtree. Operator nodes additionally share a global NIG hyper-posterior
(``GlobalHyperPrior``). Selection is done by (hierarchical) Thompson sampling;
all objectives are minimised.
"""
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import invgamma, norm

OPERATORS = ['counter', 'e2', 'm1', 'm2', 's1']

# A heuristic node with this many real children is only traversed, never expanded.
MAX_CHILDREN = 7


@dataclass
class AblationConfig:
    """Switches for the ablation study; all True = full GENESIS."""
    posterior_sampling: bool = True     # False: deterministic UCB on posterior means
    adaptive_op_selection: bool = True  # False: operator drawn uniformly at random
    global_sharing: bool = True         # False: operator nodes ignore the global hyper-prior
    ucb_c: float = 1.0                  # exploration constant when posterior_sampling=False


def lcb_score(mean, visits, parent_visits, scale, c):
    """Deterministic UCB score for minimisation (lower is better)."""
    return mean - c * scale * math.sqrt(math.log(parent_visits + 1) / (visits + 1))


def nig_update(rewards, mu0, kappa0, nu0, tau2_0):
    """Conjugate NIG posterior given i.i.d. Normal observations. Returns (mu, kappa, nu, tau2)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    n = len(rewards)
    r_bar = rewards.mean()
    ssq = ((rewards - r_bar) ** 2).sum()
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * r_bar) / kappa_n
    nu_n = nu0 + n
    tau2_n = (nu0 * tau2_0 + ssq + (kappa0 * n) / kappa_n * (r_bar - mu0) ** 2) / nu_n
    return mu_n, kappa_n, nu_n, tau2_n


def nig_sample(mu, kappa, nu, tau2):
    """Draw (mean, variance) from NIG(mu, kappa, nu, tau2)."""
    sigma2 = invgamma.rvs(a=nu / 2.0, scale=nu * tau2 / 2.0)
    mean = norm.rvs(loc=mu, scale=math.sqrt(sigma2 / kappa))
    return mean, sigma2


class GlobalHyperPrior:
    """Global NIG hyper-posterior shared by all operator nodes (they use the same LLM)."""

    def __init__(self, mu_0=0, kappa_0=1.0, nu_0=2.0, tau2_0=1.0):
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        self.tau2_0 = tau2_0

        self.mu_post = self.mu_0
        self.kappa_post = self.kappa_0
        self.nu_post = self.nu_0
        self.tau2_post = self.tau2_0

    def update_global_posterior(self, gen_nodes):
        """Re-estimate the hyper-posterior from the local posterior means of ``gen_nodes``."""
        mu_locals = [g.mu_post for g in gen_nodes]

        n = len(gen_nodes)
        r_bar = np.mean(mu_locals) if mu_locals else self.mu_0
        ssq = np.var(mu_locals) * n if len(mu_locals) > 1 else 0

        kappa_n = self.kappa_0 + n
        self.mu_post = (self.kappa_0 * self.mu_0 + n * r_bar) / kappa_n
        self.kappa_post = kappa_n
        self.nu_post = self.nu_0 + n
        self.tau2_post = (self.nu_0 * self.tau2_0 + ssq
                          + (self.kappa_0 * n) / kappa_n * (r_bar - self.mu_0) ** 2) / self.nu_post

    def sample_hyperparameter(self):
        return nig_sample(self.mu_post, self.kappa_post, self.nu_post, self.tau2_post)


class OperatorNode:
    """GEN node: expanding its parent heuristic with one (LLM, operator) pair."""

    def __init__(self, parent, llm_model_name, operator_name, global_hyper, visits=0):
        self.parent = parent
        self.llm_model_name = llm_model_name
        self.operator_name = operator_name
        self.visits = visits
        self.global_hyper = global_hyper

        # The prior is the current global hyper-posterior
        self.mu_post = global_hyper.mu_post
        self.kappa_post = global_hyper.kappa_post
        self.nu_post = global_hyper.nu_post
        self.tau2_post = global_hyper.tau2_post

        self.rewards = []
        self._lambda = 1.2  # rate of the budget-dependent score scaling, see HeuristicNode.gen_score
        self.depth = getattr(parent, "depth", 0) + 1

    def update_posterior(self, new_reward, global_hyper):
        self.rewards.append(float(new_reward))
        self.mu_post, self.kappa_post, self.nu_post, self.tau2_post = nig_update(
            self.rewards, global_hyper.mu_post, global_hyper.kappa_post,
            global_hyper.nu_post, global_hyper.tau2_post)
        self.visits += 1


class HeuristicNode:
    """Real node: an evaluated heuristic (or the virtual root)."""

    def __init__(self, algorithm, code, obj, depth=0, is_root=False, parent=None, visit=0, raw_info=None,
                 llm_model_names=None, global_hyper=None):
        self.is_root = is_root
        self.algorithm = algorithm
        self.code = code
        self.parent = parent
        self.depth = depth
        self.children = []
        self.visits = visit
        self.raw_info = raw_info
        self.reward = obj  # meta-objective, lower is better
        self.children_info = []
        self.global_hyper = global_hyper

        # One operator node per (LLM, operator) pair
        self.gen_nodes = [OperatorNode(self, llm, op, global_hyper)
                          for llm in llm_model_names for op in OPERATORS]

        # Local NIG posterior over the subtree's meta-objective
        self.mu_prior = 0
        self.kappa_prior = 10
        self.nu_prior = 2.0
        self.tau2_prior = 1.0
        self.mu_post = self.mu_prior
        self.kappa_post = self.kappa_prior
        self.nu_post = self.nu_prior
        self.tau2_post = self.tau2_prior
        self.node_rewards = []

        self._generation_method = None
        self._generation_action = None

    def add_child(self, child_node):
        child_node.parent = self
        child_node.depth = self.depth + 1
        self.children.append(child_node)

    def sample_from_node_posterior(self):
        return nig_sample(self.mu_post, self.kappa_post, self.nu_post, self.tau2_post)[0]

    def update_node_posterior(self, new_reward):
        self.node_rewards.append(float(new_reward))
        self.visits += 1
        self.mu_post, self.kappa_post, self.nu_post, self.tau2_post = nig_update(
            self.node_rewards, self.mu_prior, self.kappa_prior, self.nu_prior, self.tau2_prior)

    def cont_candidates(self, num_samples=1, ablation=None):
        """Score every real child (CONT action); lower is better."""
        ablation = ablation or AblationConfig()
        candidates = []
        for i, child in enumerate(self.children):
            if ablation.posterior_sampling:
                score = np.mean([child.sample_from_node_posterior() for _ in range(num_samples)])
            else:
                score = lcb_score(child.mu_post, child.visits, self.visits, np.sqrt(self.tau2_post), ablation.ucb_c)
            logging.debug(f"CHILD {i}: mu_post={child.mu_post:.3f}, tau2={child.tau2_post:.3f}, score={score:.3f}")
            candidates.append(('CONT', i, score))
        return candidates

    def gen_score(self, gen_node, fe, global_params, ablation):
        """Score one operator node (GEN action); lower is better."""
        local_precision = gen_node.kappa_post / gen_node.tau2_post
        if ablation.global_sharing:
            # Shrink the local posterior towards the sampled global belief (precision = kappa / sigma2)
            mu_global, sigma2_global = global_params
            global_precision = self.global_hyper.kappa_post / sigma2_global
            combined_precision = local_precision + 0.5 * global_precision
            combined_mu = (gen_node.mu_post * local_precision + mu_global * global_precision) / combined_precision
        else:
            combined_precision = local_precision
            combined_mu = gen_node.mu_post

        if ablation.posterior_sampling:
            value = norm.rvs(loc=combined_mu, scale=np.sqrt(1.0 / combined_precision))
        else:
            value = lcb_score(combined_mu, gen_node.visits, self.visits, np.sqrt(self.tau2_post), ablation.ucb_c)

        # Budget-dependent scaling: as evaluations (fe) accumulate, GEN scores grow and CONT is favoured
        score = value * math.exp(gen_node._lambda * fe / 1000)

        logging.debug(f"GEN {gen_node.operator_name}: mu_post={gen_node.mu_post:.3f}, tau2={gen_node.tau2_post:.3f}, "
                      f"combined_mu={combined_mu:.3f}, value={value:.3f}, score={score:.3f}")
        return score

    def hierarchical_thompson_sampling(self, fe, num_samples=1, ablation=None):
        """
        Hierarchical Thompson sampling over GEN and CONT actions:
        1. sample the global hyperparameter phi ~ p(phi | D);
        2. for every operator node, sample theta_i ~ p(theta_i | phi, D_i);
        3. for every real child, sample from its local posterior;
        4. return the action with the lowest sample.
        The ablation switches replace individual steps (see AblationConfig).
        """
        ablation = ablation or AblationConfig()

        if ablation.posterior_sampling:
            global_params = self.global_hyper.sample_hyperparameter()
        else:
            global_params = (self.global_hyper.mu_post, self.global_hyper.tau2_post)

        gen_candidates = []
        for gen_node in self.gen_nodes:
            score = self.gen_score(gen_node, fe, global_params, ablation)
            gen_candidates.append(('GEN', (gen_node.llm_model_name, gen_node.operator_name), score))

        if gen_candidates and not ablation.adaptive_op_selection:
            # Expand-vs-continue still uses the best GEN score; only the operator is random
            best_gen_score = min(c[2] for c in gen_candidates)
            chosen = random.choice(self.gen_nodes)
            gen_candidates = [('GEN', (chosen.llm_model_name, chosen.operator_name), best_gen_score)]

        candidates = gen_candidates + self.cont_candidates(num_samples, ablation)
        return min(candidates, key=lambda x: x[2])

    def select_action(self, fe, num_samples=1, ablation=None):
        """Return ('GEN', (llm, operator), score) or ('CONT', child_index, score)."""
        if len(self.children) >= MAX_CHILDREN:
            return min(self.cont_candidates(num_samples, ablation), key=lambda x: x[2])
        return self.hierarchical_thompson_sampling(fe, num_samples, ablation)


class GenesisTree:
    def __init__(self, llm_model_names, max_depth=10, ablation=None):
        self.max_depth = max_depth
        self.ablation = ablation or AblationConfig()
        self.rank_list = []
        self.llm_model_names = llm_model_names
        self.all_rewards_store = defaultdict(list)
        self.global_hyper = GlobalHyperPrior()

        self.root = HeuristicNode(
            algorithm='Root',
            code="Root",
            obj=0,
            depth=0,
            is_root=True,
            llm_model_names=llm_model_names,
            global_hyper=self.global_hyper
        )

    def new_node(self, offspring, parent, model_name):
        """Create a heuristic node for an evaluated offspring (not yet attached to the tree)."""
        node = HeuristicNode(
            algorithm=offspring['algorithm'], code=offspring['code'],
            obj=float(offspring['objective']), parent=parent,
            depth=parent.depth + 1, visit=1, raw_info=offspring,
            llm_model_names=self.llm_model_names, global_hyper=self.global_hyper
        )
        node._generation_method = 'GEN'
        node._generation_action = model_name
        return node

    def select_expansion_target(self, fe):
        """Descend from the root and return (node, 'GEN', (llm_name, operator)) to expand."""
        current_node = self.root

        # The root is never expanded once it has children
        if current_node.children:
            best = min(current_node.cont_candidates(ablation=self.ablation), key=lambda x: x[2])
            current_node = current_node.children[best[1]]

        while current_node.depth < self.max_depth:
            action_type, action_info, _ = current_node.select_action(fe=fe, ablation=self.ablation)
            if action_type == 'GEN' or not current_node.children:
                return current_node, action_type, action_info
            current_node = current_node.children[action_info]

        # Reached max depth, must expand here
        chosen_gen = random.choice(current_node.gen_nodes)
        return current_node, 'GEN', (chosen_gen.llm_model_name, chosen_gen.operator_name)

    def backpropagate(self, node: HeuristicNode, op_name):
        score = float(node.reward)
        if score not in self.rank_list:
            self.rank_list.append(score)
            self.rank_list.sort()

        parent = node.parent

        # Update the operator node that generated this child
        for gen_node in parent.gen_nodes:
            if gen_node.operator_name == op_name:
                gen_node.update_posterior(score, self.global_hyper)
                break

        self.all_rewards_store[(node._generation_action, op_name)].append(score)

        # Without global sharing the hyper-prior stays at its initial value, so operator nodes are independent
        if self.ablation.global_sharing:
            self.global_hyper.update_global_posterior(parent.gen_nodes)

        # Update the local posteriors of all ancestors
        current = parent
        while current is not None:
            current.update_node_posterior(score)
            current.visits += 1
            current = current.parent
