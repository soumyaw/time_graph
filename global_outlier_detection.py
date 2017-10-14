from networkx.algorithms import approximation
import operator
import networkx as nx
import numpy as np

def get_num_nodes(G):
	return nx.classes.function.number_of_nodes(G)

def get_num_edges(G):
	return nx.classes.function.number_of_edges(G)

def get_min_max_degree(G):
	degrees = sorted(G.degree().values())
	return [degrees[0], degrees[-1]]

def get_gcc_size_and_cc_num(G):
	gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
	G0 = gcc[0]
	return [len(G0), len(gcc)]

def get_core_num(G):
	G = nx.Graph(G)
	G.remove_edges_from(G.selfloop_edges())
	core_num = sorted(nx.algorithms.core.core_number(G).items(), key=operator.itemgetter(1), reverse=True)
	return core_num[0][1]

def jaccard_similarity_nodes(A, B):
	return float(len(set(A.nodes())&set(B.nodes())))/len(set(A.nodes())|set(B.nodes()))

def jaccard_similarity_edges(A, B):
	return float(len(set(A.edges())&set(B.edges())))/len(set(A.edges())|set(B.edges()))

def get_first_k_eig_val_adj(G, k):
	eig_val_adj = nx.linalg.spectrum.adjacency_spectrum(G)
	ret = [eig.real for eig in eig_val_adj]
	while len(ret) < k:
		ret.append(0.0)	
	return sorted(ret, reverse=True)[:k]
	
def get_first_k_eig_val_lap(G, k):
	eig_val_lap = nx.linalg.spectrum.laplacian_spectrum(G)
	ret = [eig.real for eig in eig_val_lap]
	while len(ret) < k:
		ret.append(0.0)	
	return sorted(ret, reverse=True)[:k]

def get_avg_clustering_coeff_approx(G):
	return approximation.clustering_coefficient.average_clustering(G)

def get_pairwise_dist_tiles(G):
	path_lens = nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path_length(G)
	dict_lens = {}
	for src in path_lens.keys():
		for dest in path_lens[src]:
			dict_lens[(src, dest)] = path_lens[src][dest]
	sorted_lens = sorted(dict_lens.items(), key=operator.itemgetter(1))
	length = len(sorted_lens)
	return [sorted_lens[i*length/10][1] for i in range(1, 10)]

def inp_vectors(G, G_prev):
	result = []
	num_nodes = get_num_nodes(G)
	result.append(num_nodes)
	num_edges = get_num_edges(G)
	result.append(num_edges)
	extr_degree = get_min_max_degree(G)
	result.extend(extr_degree)
	avg_degree = num_edges*2/float(num_nodes)
	result.append(avg_degree)
	gcc_size_and_cc_num = get_gcc_size_and_cc_num(G)
	result.extend(gcc_size_and_cc_num)
	core_num = get_core_num(G)
	result.append(core_num)
	jac_sim_nodes = jaccard_similarity_nodes(G, G_prev)
	result.append(jac_sim_nodes)
	jac_sim_edges = jaccard_similarity_edges(G, G_prev)
	result.append(jac_sim_edges)
	clus_coeff = get_avg_clustering_coeff_approx(G)
	result.append(clus_coeff)
	first_eig_adj = get_first_k_eig_val_adj(G, 4)
	result.extend(first_eig_adj)
	first_eig_lap = get_first_k_eig_val_lap(G, 4)
	result.extend(first_eig_lap)
	pw_dist = get_pairwise_dist_tiles(G)
	result.extend(pw_dist)
	return result
