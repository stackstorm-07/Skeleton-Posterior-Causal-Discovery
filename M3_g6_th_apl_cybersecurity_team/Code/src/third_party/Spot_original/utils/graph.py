import os, sys
import itertools
import tempfile, time
import numpy as np
from typing import List
from ananke.graphs import ADMG
import igraph as ig
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph

def pprint_pag(G):
    """
    Function to pretty print out PAG edges

    :param G: Ananke ADMG with 'pag_edges' attribute.
    :return: None.
    """
    print ('-'*10)
    print (f'Nodes: {list(G.vertices.keys())}')
    for edge in G.pag_edges:
        print (f'{edge["u"]} {edge["type"]} {edge["v"]}')


def write_admg_to_file(G, filename):
    """
    Function to write ADMG to file in correct format for PAG conversion.

    :param G: Ananke ADMG.
    :return: None.
    """

    f = open(filename, 'w')
    f.write("Graph Nodes:\n")
    f.write(','.join(['X'+str(v) for v in G.vertices]) + '\n\n')
    f.write("Graph Edges:\n")
    counter = 1
    for Vi, Vj in G.di_edges:
        f.write(str(counter) + '. X' + str(Vi) + ' --> X' + str(Vj) + '\n')
        counter += 1
    for Vi, Vj in G.bi_edges:
        f.write(str(counter) + '. X' + str(Vi) + ' <-> X' + str(Vj) + '\n')
        counter += 1
    f.close()

def write_pag_to_file(G: GeneralGraph, filename):
    """
    :param G:causal-learn GeneralGraph.
    :return: None.
    """
    nodes = G.get_node_names()
    edges = G.get_graph_edges()
    f = open(filename, 'w')
    f.write("Graph Nodes:\n")
    f.write(','.join([v for v in nodes]) + '\n\n')
    f.write("Graph Edges:\n")
    counter = 1
    for edge in edges:
        Vi, Vj = edge.get_node1().get_name(), edge.get_node2().get_name()
        edge_str = list("o-o")
        if edge.get_endpoint1() == Endpoint.ARROW:
            edge_str[0] = "<"
        if edge.get_endpoint1() == Endpoint.TAIL:
            edge_str[0] = "-"
        if edge.get_endpoint2() == Endpoint.ARROW:
            edge_str[2] = ">"
        if edge.get_endpoint2() == Endpoint.TAIL:
            edge_str[2] = "-"
        edge_str = "".join(edge_str)
        f.write(str(counter) + '. ' + str(Vi) + f" {edge_str} " + str(Vj) + '\n')
        counter += 1
    f.close()

def has_cycle(G):
    g = ig.Graph()

def inducing_path(G, Vi, Vj):
    """
    Checks if there is an inducing path between Vi and Vj in G.

    :return: boolean indicator whether there is an inducing path.
    """

    # easy special case of directed adjacency
    if Vi in G.parents([Vj]) or Vj in G.parents([Vi]) or Vi in G.siblings([Vj]):
        return True

    ancestors_ViVj = G.ancestors([Vi, Vj])
    visit_stack = [s for s in G.siblings([Vi]) if s in ancestors_ViVj]
    visit_stack += [c for c in G.children([Vi]) if c in ancestors_ViVj]
    visited = set()

    while visit_stack:
        if Vj in visit_stack or Vj in G.parents(visit_stack):
            return True

        v = visit_stack.pop()
        visited.add(v)
        visit_stack.extend(set([s for s in G.siblings([v]) if s in ancestors_ViVj]) - visited)
    return False


def mag_projection(G):
    """
    Get MAG projection of an ADMG G.

    :param G: Ananke ADMG.
    :return: Ananke ADMG corresponding to a MAG.
    """

    G_mag = ADMG(G.vertices)

    # iterate over all vertex pairs
    for Vi, Vj in itertools.combinations(G.vertices, 2):

        # check if there is an inducing path
        if inducing_path(G, Vi, Vj):
            # connect based on ancestrality
            if Vi in G.ancestors([Vj]):
                G_mag.add_diedge(Vi, Vj)
            elif Vj in G.ancestors([Vi]):
                G_mag.add_diedge(Vj, Vi)
            else:
                G_mag.add_biedge(Vi, Vj)

    return G_mag

def has_cycle(W1):
    """
    Check whether an adjacency matrix has almost cycles.
    :param W1: Directed Edge Adjacency Matrix.
    """
    return not ig.Graph.Weighted_Adjacency(W1).is_dag()

def admg_to_pag(G, mag_proj:bool=True):
    """
    Write an ADMG G to file, and then convert it to a PAG using tetrad

    :param G: Ananke ADMG.
    :return: Ananke ADMG with an 'pag_edges' attribute corresponding to a PAG.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # write to disk for tetrad
        if mag_proj: mag = mag_projection(G)
        else: mag = G
        mag_path = os.path.join(tmpdir, "G.mag")
        pag_path = os.path.join(tmpdir, "G.mag.pag")
        write_admg_to_file(mag, mag_path)

        # convert to pag and write to disk
        os.system(f'java -classpath "utils/tetrad-lib-6.8.0.jar:utils/xom-1.3.5.jar:utils/" convertMag2Pag {mag_path}')

        # load back into new ADMG and return
        # G = read_tetrad_graph(pag_path)
        lines = open(pag_path, 'r').read().strip().split('\n')
        nodes = lines[1].split(';')
        nodes = [str(node[1:]) for node in nodes] # remove X

        edges = []
        for line in lines[4:]:
            edge = line.split('. ')[1].split(' ')
            edges.append({'u':str(edge[0][1:]), 'v':str(edge[2][1:]), 'type':edge[1]})

        G = ADMG(nodes)
        G.pag_edges = edges


    return G

def read_tetrad_graph(graph_path:str):
    lines = open(graph_path, 'r').read().strip().split('\n')
    nodes = lines[1].split(';')
    nodes = [str(node) for node in nodes] # remove X

    G = ADMG(nodes)
    for line in lines[4:]:
        edge = line.split('. ')[1].split(' ')
        if edge[1] == "-->": G.add_diedge(str(edge[0]), str(edge[2]))
        elif edge[1] == "<->": G.add_biedge(str(edge[0]), str(edge[2]))
        elif edge[1] == "---": G.add_biedge(str(edge[0]), str(edge[2])) # temporially workaround; very rare case
        else:
            print(line)
            raise NotImplementedError()
    return G

def read_causal_cmd_graph(graph_path:str):
    lines = open(graph_path, 'r').read().strip().split('\n')
    start_idx = [i for i in range(len(lines)) if "Graph Edges:" in lines[i]][0]
    nodes_str = lines[start_idx-2].strip().split(";")
    nodes = {k:GraphNode(f"X{int(k)+1}") for k in nodes_str}
    pag = GeneralGraph(list(nodes.values()))
    lines = lines[start_idx+1:]
    for line in lines:
        if line.strip() == "": break
        edge = line.split('. ')[1].split(' ')
        if edge[1][0] == "o":
            ep1 = Endpoint.CIRCLE
        elif edge[1][0] == "-":
            ep1 = Endpoint.TAIL
        elif edge[1][0] == "<":
            ep1 = Endpoint.ARROW
        else:
            raise Exception("undefined edge endpoint")
        if edge[1][2] == "o":
            ep2 = Endpoint.CIRCLE
        elif edge[1][2] == "-":
            ep2 = Endpoint.TAIL
        elif edge[1][2] == ">":
            ep2 = Endpoint.ARROW
        else:
            raise Exception("undefined edge endpoint")
        pedge = Edge(nodes[edge[0]], nodes[edge[2]], ep1, ep2)
        pag.add_edge(pedge)
    return pag

def sample_mag(G:GeneralGraph):
    with tempfile.TemporaryDirectory() as tmpdir:
        pag_path = os.path.join(tmpdir, "G.pag")
        mag_path = os.path.join(tmpdir, "G.pag.mag")
        write_pag_to_file(G, pag_path)
        os.system(f'java -classpath "utils/tetrad-lib-6.8.0.jar:utils/xom-1.3.5.jar:utils/" convertPag2Mag {pag_path}')
        admg = read_tetrad_graph(mag_path)
    return admg

def pag2cpag(G):
    """
    Convert Ananke ADMG to Causal-Learn PAG.

    :param G: Ananke ADMG PAG.
    :return: Causal-Learn PAG.
    """
    nodes = {k:GraphNode(k) for k in G.vertices.keys()}
    pag = GeneralGraph(list(nodes.values()))
    for edge in G.pag_edges:
        if edge["type"][0] == "o":
            ep1 = Endpoint.CIRCLE
        elif edge["type"][0] == "-":
            ep1 = Endpoint.TAIL
        elif edge["type"][0] == "<":
            ep1 = Endpoint.ARROW
        else:
            raise Exception("undefined edge endpoint")
        if edge["type"][2] == "o":
            ep2 = Endpoint.CIRCLE
        elif edge["type"][2] == "-":
            ep2 = Endpoint.TAIL
        elif edge["type"][2] == ">":
            ep2 = Endpoint.ARROW
        else:
            raise Exception("undefined edge endpoint")
        pedge = Edge(nodes[edge["u"]], nodes[edge["v"]], ep1, ep2)
        pag.add_edge(pedge)
    return pag

def adj2admg(D: np.ndarray, B: np.ndarray, nodes: List[str]= None):
    """
    Convert adjacency matrix to Ananke ADMG.

    :param D: adjacency matrix for directed edges.
    :param B: adjacency matrix for bidirected edges.
    :return: Ananke ADMG..
    """
    if nodes == None:
        nodes = [f"X{i+1}" for i in range(D.shape[0])]
    admg = ADMG(nodes)
    for ix, iy in np.ndindex(D.shape):
        if D[ix, iy] == 1:
            admg.add_diedge(nodes[ix], nodes[iy])
    for ix in range(B.shape[0]):
        for iy in range(ix):
            if B[ix, iy] == 1:
                admg.add_biedge(nodes[ix], nodes[iy])
    return admg

def compare_skl(est_edges: List[Edge], true_edges: List[Edge]):
    def _adjacency_match(e1:Edge, e2:Edge):
        if e1.get_node1().get_name() == e2.get_node1().get_name() and e1.get_node2().get_name() == e2.get_node2().get_name(): return True
        elif e1.get_node2().get_name() == e2.get_node1().get_name() and e1.get_node1().get_name() == e2.get_node2().get_name(): return True
        else: return False
    
    skl_true_positive = 0
    for true_edge in true_edges:
        for est_edge in est_edges:
            if _adjacency_match(true_edge, est_edge):
                skl_true_positive += 1
        
    try:
        skl_precision = skl_true_positive / len(est_edges)
        skl_recall = skl_true_positive / len(true_edges)
        skl_f1 = 2 * (skl_precision * skl_recall) / (skl_precision + skl_recall)
    except:
        return {}
    

    return {"skl_f1": skl_f1, "skl_precision": skl_precision, "skl_recall": skl_recall}

def compare_pag(est_edges: List[Edge], true_edges: List[Edge]):
    
    def _adjacency_match(e1:Edge, e2:Edge):
        if e1.get_node1().get_name() == e2.get_node1().get_name() and e1.get_node2().get_name() == e2.get_node2().get_name(): return True
        elif e1.get_node2().get_name() == e2.get_node1().get_name() and e1.get_node1().get_name() == e2.get_node2().get_name(): return True
        else: return False
    
    def _full_match(e1:Edge, e2:Edge):
        if _adjacency_match(e1, e2):
            if e1.get_node1().get_name() == e2.get_node1().get_name():
                return e1.get_endpoint1() == e2.get_endpoint1() and e1.get_endpoint2() == e2.get_endpoint2()
            else:
                return e1.get_endpoint2() == e2.get_endpoint1() and e1.get_endpoint1() == e2.get_endpoint2()

    true_positive = 0
    skl_true_positive = 0

    est_arrowheads = set()
    true_arrowheads = set()



    est_tails = set()
    true_tails = set()



    for true_edge in true_edges:
        for est_edge in est_edges:
            if _adjacency_match(true_edge, est_edge):
                skl_true_positive += 1
                # arrowhead_tp, arrowhead_fn, arrowhead_fp, tail_tp, tail_fn, tail_fp = _check_endpoint(true_edge, est_edge, arrowhead_tp, arrowhead_fn, arrowhead_fp, tail_tp, tail_fn, tail_fp)
            if _full_match(true_edge, est_edge):
                true_positive += 1
                break
    
    for true_edge in true_edges:
        if true_edge.get_endpoint1() == Endpoint.ARROW:
            true_arrowheads.add(tuple([true_edge.get_node1().get_name(), true_edge.get_node2().get_name()]))
        if true_edge.get_endpoint2() == Endpoint.ARROW:
            true_arrowheads.add(tuple([true_edge.get_node2().get_name(), true_edge.get_node2().get_name()]))
        if true_edge.get_endpoint1() == Endpoint.TAIL:
            true_tails.add(tuple([true_edge.get_node1().get_name(), true_edge.get_node2().get_name()]))
        if true_edge.get_endpoint2() == Endpoint.TAIL:
            true_tails.add(tuple([true_edge.get_node2().get_name(), true_edge.get_node2().get_name()]))
    
    for est_edge in est_edges:
        if est_edge.get_endpoint1() == Endpoint.ARROW:
            est_arrowheads.add(tuple([est_edge.get_node1().get_name(), est_edge.get_node2().get_name()]))
        if est_edge.get_endpoint2() == Endpoint.ARROW:
            est_arrowheads.add(tuple([est_edge.get_node2().get_name(), est_edge.get_node2().get_name()]))
        if est_edge.get_endpoint1() == Endpoint.TAIL:
            est_tails.add(tuple([est_edge.get_node1().get_name(), est_edge.get_node2().get_name()]))
        if est_edge.get_endpoint2() == Endpoint.TAIL:
            est_tails.add(tuple([est_edge.get_node2().get_name(), est_edge.get_node2().get_name()]))

    arrowhead_tp = len(set.intersection(est_arrowheads, true_arrowheads))
    arrowhead_fn = len(true_arrowheads - est_arrowheads)
    arrowhead_fp = len(est_arrowheads - true_arrowheads)
    tail_tp = len(set.intersection(est_tails, true_tails))
    tail_fn = len(true_tails - est_tails)
    tail_fp = len(est_tails - true_tails)

    precision = true_positive / len(est_edges) if len(est_edges) != 0 else 0
    recall = true_positive / len(true_edges) if len(true_edges) != 0 else 0
    
    skl_precision = skl_true_positive / len(est_edges) if len(est_edges) != 0 else 0
    skl_recall = skl_true_positive / len(true_edges) if len(true_edges) != 0 else 0

    arrowhead_precision = arrowhead_tp / (arrowhead_tp+arrowhead_fp) if  arrowhead_tp+arrowhead_fp != 0 else 0
    arrowhead_recall = arrowhead_tp / (arrowhead_tp+arrowhead_fn) if  arrowhead_tp+arrowhead_fn != 0 else 0

    tail_precision = tail_tp / (tail_tp+tail_fp) if  tail_tp+tail_fp != 0 else 0
    tail_recall = tail_tp / (tail_tp+tail_fn) if  tail_tp+tail_fn != 0 else 0

    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0
    try:
        skl_f1 = 2 * (skl_precision * skl_recall) / (skl_precision + skl_recall)
    except:
        skl_f1 = 0
    try:
        arrowhead_f1 = 2 * (arrowhead_precision * arrowhead_recall) / (arrowhead_precision + arrowhead_recall)
    except:
        arrowhead_f1 = 0
    try:
        tail_f1 = 2 * (tail_precision * tail_recall) / (tail_precision + tail_recall)
    except:
        tail_f1 = 0
    
    perf = {
        "f1": f1, "fdr": 1-precision, "recall": recall, 
        "skl_f1": skl_f1, "skl_fdr": 1-skl_precision, "skl_recall": skl_recall,
        "arrowhead_f1": arrowhead_f1, "arrowhead_fdr": 1-arrowhead_precision, "arrowhead_recall": arrowhead_recall, 
        "tail_f1": tail_f1, "tail_fdr": 1-tail_precision, "tail_recall": tail_recall, 
    }

    # for key in perf.keys():
    #     perf[key] = round(perf[key], 2)
    return perf