import networkx as nx
import numpy as np

ROUNDING_COEF = 10


def match_chromatograms(dists, penalty=40):
    """
    Do matching between features of two chromatograms.

    This function is just encoding in Netwrokx graph, the matching problem
    formulated in paper SI.

    Parameters
    ----------
    dists : distances between features. May containt infinity values
    penalty : penalty for feature not matching

    Returns
    -------

    """
    # 1 - from, 2 - to, 0 - s, t node, -1 is trash node
    G = nx.DiGraph()

    # WARNING! Both unmatched nodes pay for not pairing, contrary to pairing,
    # where distance is paid once. So half of penalty should be interpreted as
    # max distance for which it is acceptable to make a matching.
    half_penalty = penalty / 2

    inds = np.nonzero(dists < np.inf)
    for i, j, dist in zip(*inds, dists[inds]):
        G.add_edge((1, i), (2, j), capacity=1,
                   weight=int(round(dist * ROUNDING_COEF)))

    for i in range(dists.shape[0]):
        G.add_edge((0, "s"), (1, i), capacity=1)
        G.add_edge((1, i), (-1, "trash"), capacity=1,
                   weight=half_penalty * ROUNDING_COEF)

    if dists.shape[0] > dists.shape[1]:
        # if equal, add no extra rubbish path
        G.add_edge((-1, "trash"), (0, "t"),
                   capacity=dists.shape[0] - dists.shape[1])

    for i in range(dists.shape[1]):
        G.add_edge((2, i), (0, "t"), capacity=1)
        G.add_edge((-1, "trash"), (2, i), capacity=1,
                   weight=half_penalty * ROUNDING_COEF)

    if dists.shape[1] > dists.shape[0]:
        # if equal, add no extra rubbish path
        G.add_edge((0, "s"), (-1, "trash"),
                   capacity=dists.shape[1] - dists.shape[0])

    min_cost_flow = nx.max_flow_min_cost(G, (0, "s"), (0, "t"))

    return extract_matching_from_flow(min_cost_flow)


def match_chromatograms_gathered_by_clusters(dists, clusters, penalty=40):
    """
    Do matching with restriction that in every cluster only feature can be matched.

    This function is just encoding in Netwrokx graph, the matching problem
    formulated in paper SI.

    Parameters
    ----------
    dists : distances between features. May containt infinity values
    clusters :
    penalty : penalty for feature not matching

    Returns
    -------

    """
    G = nx.DiGraph()

    # 1 - from, 2 - to, 3 - extra cluster layaer, 0 - s, t node, -1 is trash node
    inds = np.nonzero(dists < np.inf)
    clusters_unique = np.unique(clusters)
    for i, j, dist in zip(*inds, dists[inds]):
        G.add_edge((1, i), (2, j), capacity=1,
                   weight=int(round(dist * ROUNDING_COEF)))

    for i in range(dists.shape[0]):
        G.add_edge((0, "s"), (1, i), capacity=1)
        G.add_edge((1, i), (-1, "trash"), capacity=1,
                   weight=penalty * ROUNDING_COEF)

    if dists.shape[0] > len(clusters_unique):
        # if equal, add no extra rubbish path
        G.add_edge((-1, "trash"), (0, "t"),
                   capacity=dists.shape[0] - len(clusters_unique))

    for cluster in clusters_unique:
        G.add_edge((3, cluster), (0, "t"), capacity=1)
        G.add_edge((-1, "trash"), (3, cluster), capacity=1,
                   weight=penalty * ROUNDING_COEF)

    for i in range(dists.shape[1]):
        G.add_edge((2, i), (3, clusters[i]), capacity=1)

    if len(clusters_unique) > dists.shape[0]:
        # if equal, add no extra rubbish path
        G.add_edge((0, "s"), (-1, "trash"),
                   capacity=len(clusters_unique) - dists.shape[0])

    min_cost_flow = nx.max_flow_min_cost(G, (0, "s"), (0, "t"))

    return extract_matching_from_flow(min_cost_flow, clusters)


def extract_matching_from_flow(min_cost_flow, node_masking_clusters=None):
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
                    masked_to_id = node_masking_clusters[to_id] if node_masking_clusters else to_id
                    matchings.append((from_id, masked_to_id))
                    matched_from.add(from_id)
                    matched_to.add(masked_to_id)

    return matchings, matched_from, matched_to
