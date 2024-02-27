"""
Test graphlet counts
"""
import networkx as nx
import pandas as pd


def count_graphlets_between_nodes(graph, source, target):
    triangle_count = sum(1 for _ in nx.all_simple_paths(graph, source=source, target=target, cutoff=2) if len(_) == 3)
    # two_star_count = sum(1 for _ in nx.all_pairs_star(graph, nodes=[source, target]) if _[target] >= 2)
    two_star_count = sum(1 for node in graph if graph.degree(node) == 2)
    four_clique_count = sum(1 for _ in nx.cliques_containing_node(graph, nodes=[source, target]) if len(_) == 4)
    four_chordal_cycle_count = sum(
        1 for cycle in nx.simple_cycles(graph) if len(cycle) == 4 and source in cycle and target in cycle
    )
    four_tailed_triangle_count = sum(
        1
        for _ in nx.all_simple_paths(graph, source=source, target=target, cutoff=3)
        if len(set(_)) == 4 and _[-1] == target
    )
    four_cycle_count = sum(
        1 for cycle in nx.simple_cycles(graph) if len(cycle) == 4 and source in cycle and target in cycle
    )
    three_star_count = sum(1 for _ in nx.all_pairs_star(graph, nodes=[source, target]) if _[target] >= 3)
    four_path_count = sum(1 for _ in nx.all_simple_paths(graph, source=source, target=target, cutoff=4))
    all_shortest_paths = nx.all_shortest_paths(graph, source=source, target=target, cutoff=4)
    # # Count 2-star
    # # two_star_count = sum(1 for _, count in nx.all_pairs_star(G_given) for count in count.values())
    # two_star_count = sum(1 for node in G_given if G_given.degree(node) == 2)

    # # Count 4-clique
    # four_clique_count = sum(1 for _ in nx.find_cliques(G_given) if len(_) == 4)

    # # Count 4-chordal-cycle
    # four_chordal_cycle_count = sum(1 for cycle in nx.cycle_basis(G_given) if len(cycle) == 4)

    # # Count 4-tailed-triangle
    # four_tailed_triangle_count = sum(1 for cycle in nx.cycle_basis(G_given) if len(cycle) == 4 and len(set(cycle)) == 3)

    # # Count 4-cycle
    # four_cycle_count = sum(1 for cycle in nx.cycle_basis(G_given) if len(cycle) == 4)

    # # Count 3-star
    # # three_star_count = sum(1 for _, count in nx.all_pairs_star(G_given) for count in count.values() if count >= 2)
    # three_star_count = sum(1 for node in G_given if G_given.degree(node) == 3)

    # # Count 4-path
    # # four_path_count = sum(1 for path in nx.all_simple_paths(G_given, length=4))
    # four_path_count = sum(1 for source in G_given for target in G_given if source != target for path in nx.all_simple_paths(G_given, source, target, cutoff=4))

    # Display the counts
    print("2-star count:", two_star_count)
    print("4-clique count:", four_clique_count)
    print("4-chordal-cycle count:", four_chordal_cycle_count)
    print("4-tailed-triangle count:", four_tailed_triangle_count)
    print("4-cycle count:", four_cycle_count)
    print("3-star count:", three_star_count)
    print("4-path count:", four_path_count)

    return [
        triangle_count,
        triangle_count,
        two_star_count,
        four_clique_count,
        four_chordal_cycle_count,
        four_tailed_triangle_count,
        four_cycle_count,
        three_star_count,
        four_path_count,
        all_shortest_paths,
    ]


def get_graphlet_counts(G_given):
    results = []

    for edge in G_given.edges():
        source, target = edge
        graphlet_counts = count_graphlets_between_nodes(G_given, source, target)
        results.append([source, target] + graphlet_counts)

    columns = [
        "src",
        "dst",
        "triangle",
        "triangle",
        "2-star",
        "4-clique",
        "4-chordal-cycle",
        "4-tailed-triangle",
        "4-cycle",
        "3-star",
        "4-path",
        "all_shortest_paths",
    ]
    df = pd.DataFrame(results, columns=columns)

    return df
