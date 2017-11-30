
# coding: utf-8

import make_networkx_graph as mk_graph
import networkx as nx
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import random
import itertools
import operator
import datetime
import copy


def complete_graph_from_list(L, create_using=None):
    G = nx.empty_graph(len(L), create_using)
    if len(L) > 1:
        edges = itertools.combinations(L, 2)
        G.add_edges_from(edges)
    return G


def xor(A, B):
    return (A-B) | (B-A)


def diff(A, B):
    return A-B


def random_flip(edges, non_edges, percent):
    to_flip = int(percent*len(edges))
    edges_to_flip = random.sample(edges, to_flip)
    non_edges_to_flip = random.sample(non_edges, to_flip)
    return set((set(edges) - set(edges_to_flip)) | set(non_edges_to_flip)), set((set(non_edges) - set(non_edges_to_flip)) | set(edges_to_flip))


def inject(num_nodes, size, type='clique'):
    print "Injecting clique of size: "+str(size)
    H = complete_graph_from_list(range(num_nodes-size, num_nodes))
    H_edges = H.edges()
    return H_edges


def synth_data():
    path_init = 'data/synthetic/kron-g/A_'
    M = 5
    time_steps = 80
    noise = [0, 0.001, 0.01, 0.1, 0.5, 0.8, 1.0]
    len_noise = len(noise)
    anom_size = [i*25 for i in range(1,11)]
    len_anom = len(anom_size)
    num_eigs = 4
    eig_mat = np.ones((len_noise, time_steps-1, num_eigs))
    eig_vec_mat = []
    for i in range(len_noise):
    	print "Noise level "+str(noise[i])
        A = scp.sparse.load_npz(path_init+str(M)+'_'+str(noise[i])+'_'+str(0)+'.npz')
        G = nx.from_scipy_sparse_matrix(A)
        G_edges = set(G.edges())
        prev_edge_set = copy.deepcopy(G_edges)
        num_nodes = len(G.nodes())
        k = 0
        curr_i_list = []
        for j in range(1, time_steps):
            A = scp.sparse.load_npz(path_init+str(M)+'_'+str(noise[i])+'_'+str(j)+'.npz')
            G = nx.from_scipy_sparse_matrix(A)
            G_edges = set(G.edges())
            edge_set = copy.deepcopy(G_edges)
            if (j+1)%8 == 0:
                print "Timestep: "+str(j)
                H_edges = inject(num_nodes, anom_size[k])
                k+=1
                edge_set |= set(H_edges)
            diff_edges = edge_set
            if len(diff_edges) > 0:
                result = nx.Graph()
                result.add_edges_from(diff_edges)
                A = nx.to_scipy_sparse_matrix(result)
                A = A.asfptype()
                vals, vecs = scp.sparse.linalg.eigs(A)
                vals_vecs = sorted(zip(np.real(vals),np.real(vecs)), key=lambda x: x[0], reverse=True)
                vals = [x[0] for x in vals_vecs]
                vecs = [x[1] for x in vals_vecs]
                eig_sz = len(vals)
                to_copy = min(eig_sz, num_eigs)
                eig_mat[i, j-1, :to_copy] = vals[:to_copy]
                prev_edge_set = copy.deepcopy(edge_set)
                curr_i_list.append(vecs)
        eig_vec_mat.append(curr_i_list)
        plt.plot(range(1, time_steps), eig_mat[i,:])
        plt.savefig('results/plot_'+str(noise[i])+'_'+str(num_eigs)+'curr.jpg')
        plt.close()
        plt.plot(range(1, time_steps), np.log(eig_mat[i,:]))
        plt.savefig('results/log_plot_'+str(noise[i])+'_'+str(num_eigs)+'curr.jpg')
        plt.close()
    return eig_mat, eig_vec_mat


def synth_data_alt():
    path_init = 'data/synthetic/kron-g/A_'
    M = 5
    time_steps = 80
    noise = [0, 0.001, 0.01, 0.1, 0.5, 0.8, 1.0]
    len_noise = len(noise)
    anom_size = [i*25 for i in range(1,11)]
    len_anom = len(anom_size)
    num_eigs = 4
    eig_mat = np.ones((len_noise, time_steps-1, num_eigs))
    for i in range(len_noise):
    	print "Noise level "+str(noise[i])
        A = scp.sparse.load_npz(path_init+str(M)+'_'+str(0)+'_'+str(0)+'.npz')
        G = nx.from_scipy_sparse_matrix(A)
        G_edges = set(G.edges())
        prev_edge_set = copy.deepcopy(G_edges)
        prev_non_edges = set(nx.non_edges(G))
        num_nodes = len(G.nodes())
        k = 0 
        for j in range(1, time_steps):
            G_edges, G_non_edges = random_flip(prev_edge_set, prev_non_edges, noise[i])
            edge_set = copy.deepcopy(G_edges)
            if (j+1)%8 == 0:
                print "Timestep: "+str(j)
                H_edges = inject(num_nodes, anom_size[k])
                k+=1
                edge_set |= set(H_edges)
            diff_edges = xor(edge_set, prev_edge_set)
            print len(diff_edges)
            if len(diff_edges) > 0:
                result = nx.Graph()
                result.add_edges_from(diff_edges)
                A = nx.to_scipy_sparse_matrix(result)
                A = A.asfptype()
                vals, vecs = scp.sparse.linalg.eigs(A)
                vals_vecs = sorted(zip(np.real(vals),np.real(vecs)), key=lambda x: x[0], reverse=True)
                vals = [x[0] for x in vals_vecs]
                eig_sz = len(vals)
                to_copy = min(eig_sz, num_eigs)
                eig_mat[i, j-1, :to_copy] = vals[:to_copy]
                prev_edge_set = copy.deepcopy(G_edges)
                prev_non_edges = copy.deepcopy(G_non_edges)
        plt.plot(range(1, time_steps), eig_mat[i,:])
        plt.savefig('results/plot_alt_'+str(noise[i])+'_'+str(num_eigs)+'.jpg')
        plt.close()
        plt.plot(range(1, time_steps), np.log(eig_mat[i,:]))
        plt.savefig('results/log_plot_alt_'+str(noise[i])+'_'+str(num_eigs)+'.jpg')
        plt.close()


def real_data():
    file_name = 'data/enron_unix.txt'
    granularity = 'daily'
    time_unit = 86400
    edges = mk_graph.get_edges(file_name, granularity)
    times = sorted(edges.keys())

    len_keys = len(times)

    num_eigs = 1
    
    eig_val = np.ones((len_keys - 1, num_eigs))

    edges_prev = edges[times[0]]

    for i in range(1, len_keys):
        differ = set(edges[times[i]])
        if len(differ) > 0:
            G = nx.Graph()
            G.add_edges_from(differ)
            A = nx.to_numpy_matrix(G)
            vals, vecs = np.linalg.eig(A)
            vals = sorted(np.real(vals), reverse=True)
            eig_sz = len(vals)
            to_copy = min(eig_sz, num_eigs)
            eig_val[i-1, :to_copy] = vals[:to_copy]    
    plt.plot(times[1:], eig_val)
    plt.savefig('results/plot_enron_'+str(num_eigs)+'curr.jpg')
    plt.close()
    plt.plot(times[1:], np.log(eig_val))
    plt.savefig('results/log_plot_enron_'+str(num_eigs)+'curr.jpg')
    plt.close()
    combined = sorted(zip(times[1:], eig_val[:,0]), key=operator.itemgetter(1), reverse=True)
    n = 20
    print "The top "+str(n)+" anomalies based on peak principle eigenvalues:"
    for i in range(n):
        print datetime.datetime.utcfromtimestamp(int(combined[i][0])*time_unit).strftime('%Y-%m-%d %H:%M:%S')
        print combined[i][1]
        print '...'


def detect_and_attribute(eig_mat, eig_vec_mat, thresh=0.08):    
    noise = [0, 0.001, 0.01, 0.1, 0.5, 0.8, 1.0]
    len_noise = len(noise)
    time_steps = 80
    for i in range(len_noise):
    	print "Noise level "+str(noise[i])
        tol = thresh*np.sum(eig_mat[i, :, 0])/time_steps
        # print tol
        for j in range(time_steps-2):
            if eig_mat[i, j+1, 1] - eig_mat[i, j, 1] > tol:
                print "Injected discontinuity detected at "+str(j+2)
       	print "..."

print "Real Enron Data..."
real_data()
print "..."
print "Synthetic Data (Kronecker + Clique Injection)"
eig_mat, eig_vec_mat = synth_data()
print "..."
detect_and_attribute(eig_mat, eig_vec_mat)

