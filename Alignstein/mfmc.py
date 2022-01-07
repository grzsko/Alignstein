import networkx as nx
import numpy as np


def match_chromatograms(ch_dists, penalty=40):
    # 1 - from, 2 - to, 0 - s, t node, -1 is trash node
    G = nx.DiGraph()
    ROUNDING_COEF = 10

    # WARNING! Both unmatched nodes pay for not pairing, contrary to pairing,
    # where distance is paid once. So half of penalty should be interpreted as max
    # distance for which it is acceptable to make a matching.

    inds = np.nonzero(ch_dists < np.inf)
    for i, j, dist in zip(*inds, ch_dists[inds]):
        G.add_edge((1, i), (2, j), capacity=1,
                   weight=int(round(dist * ROUNDING_COEF)))

    for i in range(ch_dists.shape[0]):
        G.add_edge((0, "s"), (1, i), capacity=1)
        G.add_edge((1, i), (-1, "trash"), capacity=1,
                   weight=penalty * ROUNDING_COEF)

    if ch_dists.shape[0] > ch_dists.shape[
        1]:  # if equal, add no extra rubbish path
        G.add_edge((-1, "trash"), (0, "t"),
                   capacity=ch_dists.shape[0] - ch_dists.shape[1])

    for i in range(ch_dists.shape[1]):
        G.add_edge((2, i), (0, "t"), capacity=1)
        G.add_edge((-1, "trash"), (2, i), capacity=1,
                   weight=penalty * ROUNDING_COEF)

    if ch_dists.shape[1] > ch_dists.shape[
        0]:  # if equal, add no extra rubbish path
        G.add_edge((0, "s"), (-1, "trash"),
                   capacity=ch_dists.shape[1] - ch_dists.shape[0])

    min_cost_flow = nx.max_flow_min_cost(G, (0, "s"), (0, "t"))

    return extract_matching_from_flow(min_cost_flow)


def match_chromatograms_with_gathering_nodes(ch_dists, penalty=40):
    # 1 - from, 2 - to, 0 - s, t node, -1 is trash node
    G = nx.DiGraph()
    ROUNDING_COEF = 10

    # WARNING! Both unmatched nodes pay for not pairing, contrary to pairing,
    # where distance is paid once. So half of penalty should be interpreted as max
    # distance for which it is acceptable to make a matching.

    # TODO think how to correctly formulate gathering and reimplement
    inds = np.nonzero(ch_dists < np.inf)
    for i, j, dist in zip(*inds, ch_dists[inds]):
        G.add_edge((1, i), (2, j), capacity=1,
                   weight=int(round(dist * ROUNDING_COEF)))

    for i in range(ch_dists.shape[0]):
        G.add_edge((0, "s"), (1, i), capacity=1)
        G.add_edge((1, i), (-1, "trash"), capacity=1,
                   weight=penalty * ROUNDING_COEF)

    if ch_dists.shape[0] > ch_dists.shape[1]:
        # if equal, add no extra rubbish path
        G.add_edge((-1, "trash"), (0, "t"),
                   capacity=ch_dists.shape[0] - ch_dists.shape[1])

    for i in range(ch_dists.shape[1]):
        G.add_edge((2, i), (0, "t"), capacity=1)
        G.add_edge((-1, "trash"), (2, i), capacity=1,
                   weight=penalty * ROUNDING_COEF)

    if ch_dists.shape[1] > ch_dists.shape[0]:
        # if equal, add no extra rubbish path
        G.add_edge((0, "s"), (-1, "trash"),
                   capacity=ch_dists.shape[1] - ch_dists.shape[0])

    min_cost_flow = nx.max_flow_min_cost(G, (0, "s"), (0, "t"))

    return extract_matching_from_flow(min_cost_flow)


def extract_matching_from_flow(min_cost_flow):
    matchings = []
    matched_from = set()
    matched_to = set()
    # unmatched_from = set()
    # matched_to = set()

    min_cost_flow.pop((0, "s"), None)
    min_cost_flow.pop((0, "t"), None)

    for from_type, from_id in min_cost_flow:
        if from_type == 1:
            for to_type, to_id in min_cost_flow[(from_type, from_id)]:
                if (to_type == 2 and
                        min_cost_flow[(from_type, from_id)][(to_type, to_id)]):
                    matchings.append((from_id, to_id))
                    matched_from.add(from_id)
                    matched_to.add(to_id)

    return matchings, matched_from, matched_to
