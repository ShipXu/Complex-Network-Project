import os
import csv
from nasbench import api
from utils import Digraph
from evaluator import get_N, get_M, get_ave_k, get_ave_C, get_ave_L, get_coreness_graph

def if_exists(filepath):
    return os.path.exists(filepath)

def remove_files():
    files = ['result2.csv']
    for file in files:
        if if_exists(file):
            os.remove(file)

remove_files()

# Replace this string with the path to the downloaded nasbench.tfrecord before
# executing.
NASBENCH_TFRECORD = r'D:\data\nas\nasbench_only108.tfrecord'
# NASBENCH_TFRECORD = '/home/shipxu/data/nasbench_only108.tfrecord'

def construct_dag_frommatrix(matrix):
    length, width = matrix.shape
    assert(length == width)
    dag = Digraph()

    for i in range(0, length):
        for j in range(0, width):
            if matrix[i][j] == 1:
                dag.addEdge(i, j)

    return dag

if __name__ == '__main__':
    # Load the data from file (this will take some time)
    nasbench = api.NASBench(NASBENCH_TFRECORD)
    num_records = 10000
    count = 0

    adj_ms = []

    for unique_hash in nasbench.hash_iterator():
        if count >= num_records:
            break

        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
        adj_m = fixed_metrics['module_adjacency']
        params = fixed_metrics['trainable_parameters']
        acc1 = computed_metrics[108][0]['final_test_accuracy']
        acc2 = computed_metrics[108][1]['final_test_accuracy']
        acc3 = computed_metrics[108][2]['final_test_accuracy']
        acc = (acc1 + acc2 + acc3) / 3

        dag = construct_dag_frommatrix(adj_m)
        N = get_N(dag)
        M = get_M(dag)
        ave_k = get_ave_k(dag)
        ave_c = get_ave_C(dag)
        ave_l = get_ave_L(dag)
        coreness = get_coreness_graph(dag)

        with open('result2.csv', mode='a+', newline='') as f:
            data = [N, M, ave_k, ave_c, ave_l, coreness, acc, params]
            writer = csv.writer(f)
            writer.writerow(data)

        count += 1
