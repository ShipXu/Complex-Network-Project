import numpy as np

class Node(object):
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.content == other.content

    def __hash__(self):
        return hash(self.content)

class Edge(object):
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node


class Union_Find(object):
    def __init__(self, count):
        self.count = count
        self.id = [i for i in range(0, count)]

    def addNode(self):
        self.id.append(self.count)
        self.count += 1

    def find(self, p):
        return self.id[p]

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def union(self, p, q):
        pid = self.find(p)
        qid = self.find(q)
        if pid == qid:
            return
        for i in range(0, len(self.id)):
            if self.id[i] == pid:
                self.id[i] = qid
        self.count -= 1

    def get_counts(self):
        result = {}
        for i in set(self.id):
            result[i] = self.id.count(i)
        return result

class Digraph(object):
    def __init__(self):
        self.nodes = []
        self.nodes_indexes = {}
        self.adj = []

    def containsNode(self, node):
        return node in self.nodes_indexes

    def addNode(self, node):
        if not self.containsNode(node):
            self.nodes.append(node)
            self.adj.append(list())
            self.nodes_indexes[node] = len(self.nodes) - 1

        return self.nodes_indexes[node]

    def _addEdge(self, from_index, to_index):
        self.adj[from_index].append(to_index)

    def addEdge(self, from_node, to_node):
        from_index = self.addNode(from_node)
        to_index = self.addNode(to_node)
        self._addEdge(from_index, to_index)

    def __len__(self):
        return len(self.nodes)

    def get_adj_v(self):
        return self.adj

    def get_adj_m(self):
        len_n = len(self)
        adj_m = np.zeros((len_n, len_n), dtype=np.int)
        for i, adj_i in enumerate(self.adj):
            for node_index in adj_i:
                adj_m[i][node_index] = 1
        return adj_m
    
    def get_adj_k(self):
        adj_v = self.get_adj_v()
        adj_k = {}
        for i, adj_i in enumerate(adj_v):
            adj_k[i] = adj_i
        return adj_k

    def __str__(self):
        ret = ('------------------DAG desciption------------\n'               +
                'dag.adj : {}'.format(str(self.adj))                          + '\n')

        ret += 'dag.nodes_indexes:\n{\n'
        for node in self.nodes_indexes.keys():
            ret += '\t{} : {},\n'.format(str(node), str(self.nodes_indexes[node]))
        ret += '}\n'

        ret += 'dag.nodes:'
        ret += str([str(node) for node in self.nodes])
        return ret + '\n'

if __name__ == '__main__':
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

    print(dag)
    print(dag.get_adj_v())
    print(dag.get_adj_m())

