import os
import csv
import math
import keras
import numpy as np
from queue import Queue

from utils import Digraph
from model_processer import read_models, model_to_dag
# from models.lenet import Lenet

from models.inception_resnet_v2 import InceptionResNetV2
from models.inception_v3 import InceptionV3
from models.mobilenet import MobileNet
# from models.music_tagger_crnn import MusicTaggerCRNN
from models.resnet50 import ResNet50
from models.vgg16 import VGG16
from models.vgg19 import VGG19
from models.xception import Xception
from models.nasnet import NASNet

def remove_files():
    files = ['result1.csv']
    for file in files:
        if if_exists(file):
            os.remove(file)

def create_dirs():
    dirs = ['models_visulization']
    for dir in dirs:
        if not if_exists(dir):
            os.makedirs(dir)

create_dirs()
remove_files()

def get_N(dag):
    return len(dag)

def get_M(dag):
    m = 0
    adj = dag.get_adj_v()

    for adj_i in adj:
        m += len(adj_i)
    return m

def _get_in_degrees(dag):
    in_ks = [0] * get_N(dag)
    adj = dag.get_adj_v()
    for adj_i in adj:
        for node_index in adj_i:
            in_ks[node_index] += 1
    return in_ks

def _get_out_degrees(dag):
    adj = dag.get_adj_v()
    out_ks  = [len(adj_i) for adj_i in adj]
    return out_ks

def get_ave_k(dag):
    n = get_N(dag)
    m = get_M(dag)
    return 2 * m / n

def _get_E_i(dag, i):
    adj_m = dag.get_adj_m()
    e_i = 0

    for j in range(0, get_N(dag)):
        for k in range(j, get_N(dag)):
            e_i += adj_m[i][j] * adj_m[j][k] * adj_m[k][i]
    return e_i

def _get_Es(dag):
    adj_m = dag.get_adj_m()
    es = [0] * get_N(dag)

    for i in range(0, get_N(dag)):
        for j in range(0, get_N(dag)):
            for k in range(j, get_N(dag)):
                if i != j and i != k:
                    es[i] += adj_m[i][j] * adj_m[j][k] * adj_m[k][i]
    return es

def calculate_C(e, d):
    if d == 0 or d == 1:
        return 0
    else:
        return 2 * e / (d * (d - 1))

def _get_Cs(dag):
    es = _get_Es(dag)
    out_ks = _get_out_degrees(dag)
    return [calculate_C(es[i], out_ks[i]) for i in range(0, get_N(dag))]

def get_ave_C(dag):
    Cs = _get_Cs(dag)
    return sum(Cs) / get_N(dag)

def _get_Ls(dag):
    INFINITY = -1
    adj_v = dag.get_adj_v()
    ret = []
    for node_i in range(0, get_N(dag)):
        q = Queue()
        dist = [INFINITY] * get_N(dag)

        for i in range(0, get_N(dag)):
            dist[i] = INFINITY

        dist[node_i] = 0
        q.put(node_i)

        while not q.empty():
            v = q.get()
            for w in adj_v[v]:
                if dist[w] == INFINITY:
                    dist[w] = dist[v] + 1
                    q.put(w)
        ret.append(dist)
    return np.maximum(np.asarray(ret), 0)

def get_ave_L(dag):
    n = get_N(dag)
    ls = _get_Ls(dag)
    return 2 * np.sum(ls) / (n * (n -1))

def _get_coreness(dag):
    adj_k = dag.get_adj_k()
    coreness = 1
    remove_nodes = []
    visited_nodes = []
    fronts = {}

    while len(visited_nodes) < len(dag):
        # first removing process
        for key in adj_k.keys():
            if len(adj_k[key]) < coreness:
                remove_nodes.append(key)
        for node in remove_nodes:
            if node in adj_k:
                adj_k.pop(node)

        visited_nodes.extend(remove_nodes)

        # remove_nodes if not empty
        if len(remove_nodes) > 0:
            for key in adj_k.keys():
                for node_index in adj_k[key]:
                    if node_index in remove_nodes:
                        adj_k[key].remove(node_index)

            # recheck the situaion
            for key in adj_k.keys():
                if len(adj_k[key]) < coreness:
                    remove_nodes.append(key)
            for node in remove_nodes:
                if node in adj_k:
                    adj_k.pop(node)

            for key in adj_k.keys():
                for node_index in adj_k[key]:
                    if node_index in remove_nodes:
                        adj_k[key].remove(node_index)

            if (coreness - 1) in fronts:
                fronts[(coreness - 1)].extend(remove_nodes)
            else:
                fronts[(coreness - 1)] = remove_nodes.copy()
            visited_nodes.extend(remove_nodes)
        else:
            coreness += 1

        remove_nodes.clear()
    return fronts

def get_coreness_graph(dag):
    fronts = _get_coreness(dag)
    return max([key for key in fronts.keys()])

def evaluate(dag):
    return get_N(dag), get_M(dag), get_ave_k(dag), get_ave_C(dag), get_ave_L(dag)

# test01 for a normal dag
def test1():
    G = {
        'a': list('bcdef'),
        'b': list('ac'),
        'c': list('abd'),
        'd': list('ace'),
        'e': list('ad'),
        'f': list('a')
    }

    dag = Digraph()
    for u in G:
        for v in G[u]:
            dag.addEdge(u, v)

    print(get_coreness_graph(dag))
    adj_v = dag.get_adj_v()
    adj_m = dag.get_adj_m()

    print(adj_v)
    print(adj_m)
    print(_get_E_i(dag, 0))
    print(_get_Es(dag))
    print(_get_out_degrees(dag))
    print(_get_Cs(dag))
    print(_get_Ls(dag))

    print(dag)
    print('N = {}'.format(get_N(dag)) )
    print('M = {}'.format(get_M(dag)) )
    print('<k> = {}'.format(get_ave_k(dag)) )
    print('C = {}'.format(get_ave_C(dag)))
    print('L = {}'.format(get_ave_L(dag)))

# test 2: test for models
def test2():
    # model_classess = [InceptionResNetV2, InceptionV3, MobileNet, ResNet50, VGG16, VGG19, Xception, NASNet]
    model_classess = [NASNet]
    for model_class in model_classess:
        model = model_class()
        dag = model_to_dag(model)

        with open('result1.csv', mode='a+', newline='') as f:
            data = [str(model_class.__name__)]
            data.extend(evaluate(dag))
            writer = csv.writer(f)
            writer.writerow(data)

        print('model name : {}'.format(model_class.__name__))
        print('N = {}'.format(get_N(dag)) )
        print('M = {}'.format(get_M(dag)) )
        print('<k> = {}'.format(get_ave_k(dag)) )
        print('C = {}'.format(get_ave_C(dag)))
        print('L = {}'.format(get_ave_L(dag)))
        print('coreness = {}'.format(get_coreness_graph(dag)))

if __name__ == '__main__':
    test1()
    test2()


