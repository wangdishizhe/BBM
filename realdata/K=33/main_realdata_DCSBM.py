import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
import matplotlib.colors as mcolors
from DCSBM_score import DCSBM_score
from accuracy import compute_best_accuracy

def read_graph(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G

def read_community(file_path):
    membership = {}
    with open(file_path, 'r') as f:
        for line in f:
            node, label = map(int, line.strip().split())
            membership[node] = label
    return membership

def generate_one_hot_vectors(membership, num_nodes, num_classes):
    Z = np.zeros((num_nodes, num_classes), dtype=int)
    for node, label in membership.items():
        Z[node][label - 1] = 1
    return Z

def extract_largest_connected_component(G):
    LCC = max(nx.connected_components(G), key=len)
    return G.subgraph(LCC).copy()

def create_5x3_layout_ordered(G, membership, grid_width=20.0, grid_height=20.0, radius_divisor=2.0, seed=42):
    random.seed(seed)
    departments = sorted(list(set(membership.values())))
    rows, cols = 5, 3
    cell_width = grid_width / cols
    cell_height = grid_height / rows
    grid_centers = []
    for r in range(rows):
        for c in range(cols):
            center_x = cell_width / 2 + c * cell_width
            center_y = cell_height / 2 + r * cell_height
            grid_centers.append((center_x, center_y))
    dept_to_center = {}
    for dept, center in zip(departments, grid_centers):
        dept_to_center[dept] = center
    radius_max = min(cell_width, cell_height) / radius_divisor
    pos = {}
    for node, dept in membership.items():
        center = dept_to_center[dept]
        r_offset = radius_max * math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)
        dx = r_offset * math.cos(theta)
        dy = r_offset * math.sin(theta)
        pos[node] = (center[0] + dx, center[1] + dy)
    return pos

def visualize_true_community(G, membership_true_ranked, pos, K_true, palette):
    plt.figure(figsize=(10, 10))
    node_size = 40; edge_alpha = 0.2; edge_width = 0.5
    cmap = mcolors.ListedColormap(palette)
    node_colors = [membership_true_ranked[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap, node_size=node_size, alpha=0.9, vmin=1, vmax=K_true)
    edges = [(u, v) for (u, v) in G.edges() if u != v]
    edge_colors = ['gray' if membership_true_ranked[u] != membership_true_ranked[v] else 'purple' for (u, v) in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, alpha=edge_alpha, width=edge_width)
    plt.title(f"True Community (K={K_true})")
    plt.axis('off')
    plt.savefig("true_community.png", dpi=300, format="PNG")
    plt.close()

def visualize_predicted_community_and_scatter(G, membership_pred_ranked, pos, K_true, palette, degree_info_file, output_prefix="dept_scatter"):

    plt.figure(figsize=(10, 10))
    node_size = 40; edge_alpha = 0.2; edge_width = 0.5
    node_colors = [palette[membership_pred_ranked[node] - 1] for node in G.nodes()]  
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.9)
    edges = [(u, v) for (u, v) in G.edges() if u != v]
    edge_colors = ['gray' if membership_pred_ranked[u] != membership_pred_ranked[v] else 'purple' for (u, v) in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, alpha=edge_alpha, width=edge_width)
    plt.title(f"Predicted Community DCBM (K={K_true})")
    plt.axis('off')
    plt.savefig("predicted_community_DCBM.png", dpi=300, format="PNG")
    plt.close()

    dept_data = {}
    with open(degree_info_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            dept, z_hat_index, intra, inter = parts
            dept = int(dept)
            z_hat_index = int(z_hat_index)
            intra = float(intra)
            inter = float(inter)
            if dept not in dept_data:
                dept_data[dept] = ([], [], [])
            dept_data[dept][0].append(intra)
            dept_data[dept][1].append(inter)
            dept_data[dept][2].append(palette[z_hat_index - 1])  
     
    
    for dept in sorted(dept_data.keys()):
        x = dept_data[dept][0]
        y = dept_data[dept][1]
        colors = np.array(dept_data[dept][2])  
        print(f"Colors for department {dept}:", colors)
        plt.figure()
        plt.scatter(x, y, c=colors, alpha=1.0)  
        plt.xlabel("Intra_degrees")
        plt.ylabel("Inter_degrees")
        plt.title(f"Department {dept} Degree Info")
        plt.savefig(f"{output_prefix}_{dept}.png", dpi=300)
        plt.close()

def save_degree_info(G, membership, pre, output_filename):
    degree_info = []
    for node in G.nodes():
        dept = membership[node]
        p = pre[node]
        intra, inter = 0, 0
        for neighbor in G[node]:
            if neighbor == node:
                intra += 1
            elif membership[neighbor] == dept:
                intra += 1
            else:
                inter += 1
        degree_info.append((dept, intra, p, inter))
    degree_info.sort(key=lambda x: (x[0], x[1]))
    with open(output_filename, 'w') as f:
        for dept, intra, p, inter in degree_info:
            f.write(f"{dept} {p} {intra} {inter}\n")

def rank_by_size(membership):
    dept_count = {}
    for node, dept in membership.items():
        dept_count[dept] = dept_count.get(dept, 0) + 1
    sorted_depts = sorted(dept_count.keys(), key=lambda d: dept_count[d], reverse=True)
    new_label_map = {dept: i+1 for i, dept in enumerate(sorted_depts)}
    return {node: new_label_map[dept] for node, dept in membership.items()}

def calculate_averaged_internal_density(adj, labels):
    K = len(np.unique(labels))
    total_density = 0.0
    for k in range(1, K + 1):
        nodes = np.where(labels == k)[0]
        nk = len(nodes)
        community_subgraph = adj[np.ix_(nodes, nodes)]
        internal_edges = np.sum(community_subgraph) / 2
        possible_edges = nk * (nk - 1) / 2
        density = (internal_edges + 1) / (possible_edges + 1)
        total_density += density * nk
    return total_density / len(labels)

def main():
    graph_file = 'email-Eu-core.txt'
    community_file = 'email-Eu-core-department-labels.txt'
    G = read_graph(graph_file)
    UG = G.to_undirected()
    LCC = extract_largest_connected_component(UG)
    nodes_list = list(LCC.nodes())
    community_full = read_community(community_file)
    community_LCC = {node: community_full[node] for node in nodes_list}
    node_map = {old: new for new, old in enumerate(nodes_list)}
    G_lcc = nx.relabel_nodes(LCC, node_map)
    membership_true = {node_map[old]: community_LCC[old] for old in nodes_list}
    dept_count = {}
    for node, dept in membership_true.items():
        dept_count[dept] = dept_count.get(dept, 0) + 1
    sorted_depts = sorted(dept_count.keys(), key=lambda d: dept_count[d], reverse=True)
    top_depts = set(sorted_depts[:33])
    membership_true_filtered = {node: dept for node, dept in membership_true.items() if dept in top_depts}
    nodes_filtered = sorted(membership_true_filtered.keys())
    G_filtered = G_lcc.subgraph(nodes_filtered).copy()
    if not nx.is_connected(G_filtered.to_undirected()):
        LCC_filtered = max(nx.connected_components(G_filtered.to_undirected()), key=len)
        G_filtered = G_filtered.subgraph(LCC_filtered).copy()
        membership_true_filtered = {node: membership_true_filtered[node] for node in G_filtered.nodes()}
    mapping = {old: new for new, old in enumerate(sorted(G_filtered.nodes()))}
    G_filtered = nx.relabel_nodes(G_filtered, mapping)
    membership_true_filtered = {mapping[node]: dept for node, dept in membership_true_filtered.items()}
    membership_true_ranked = rank_by_size(membership_true_filtered)
    unique_depts = set(membership_true_ranked.values())
    K_true = len(unique_depts)
    num_nodes = G_filtered.number_of_nodes()
    Z_true = generate_one_hot_vectors(membership_true_ranked, num_nodes, K_true)
    A = nx.to_numpy_array(G_filtered, nodelist=range(num_nodes), dtype=int)
    Z_hat = DCSBM_score(A, K_true)
    membership_pred = {node: (np.argmax(Z_hat[node]) + 1) for node in range(num_nodes)}
    membership_pred_ranked = rank_by_size(membership_pred)
    degree_info_file = "department_degree_info_DCBM.txt"
    save_degree_info(G_filtered, membership_true_ranked, membership_pred_ranked, degree_info_file)
    pos = nx.kamada_kawai_layout(G_filtered)
    
    palette = sns.color_palette("Set1", K_true) if K_true == 9 else sns.color_palette("hls", K_true)
   # visualize_true_community(G_filtered, membership_true_ranked, pos, K_true, palette)
    #visualize_predicted_community_and_scatter(G_filtered, membership_pred_ranked, pos, K_true, palette, degree_info_file, output_prefix="dept_scatter_DCBM")

    Z_true_rank = generate_one_hot_vectors(membership_true_ranked, num_nodes, K_true)
    Z_hat_rank = generate_one_hot_vectors(membership_pred_ranked, num_nodes, K_true)
    acc = compute_best_accuracy(Z_hat_rank, Z_true_rank)
    z_true_labels = np.array([membership_true_ranked[node] for node in range(num_nodes)])
    z_pred_labels = np.array([membership_pred_ranked[node] for node in range(num_nodes)])
    density_true = calculate_averaged_internal_density(A, z_true_labels)
    density_pred = calculate_averaged_internal_density(A, z_pred_labels)
    with open('graph_info_DCBM.txt', 'w') as f:
        f.write(f"Number of nodes in filtered subgraph: {num_nodes}\n")
        f.write(f"Number of departments: {K_true}\n")
        f.write(f"Accuracy (ACC): {acc}\n")
        f.write(f"Averaged internal density (True): {density_true}\n")
        f.write(f"Averaged internal density (Pred): {density_pred}\n")
        f.write(f"Number of edges (including self-loops): {num_edges}\n")
    print(f"Metric ACC (rank-based): {acc}")

if __name__ == "__main__":
    main()