from utils import Node
from utils import Digraph

def read_models(model):
    top_tensors = {}
    bottom_tensors = {}
    layer_map = {}
    for layer in model.layers:
        layer_map[layer.name] = layer
        for i in range(len(layer._inbound_nodes)):
            outNode = layer.get_output_at(i)
            if not isinstance(outNode, list):
                if outNode not in top_tensors:
                    top_tensors[outNode] = [layer]
                else:
                    top_tensors[outNode].append(layer)
            else:
                for tensor in outNode:
                    if tensor not in top_tensors:
                        top_tensors[tensor] = [layer]
                    else:
                        top_tensors[tensor].append(layer)

        for i in range(len(layer._inbound_nodes)):
            inNode = layer.get_input_at(i)
            if not isinstance(inNode, list):
                if inNode not in bottom_tensors:
                    bottom_tensors[inNode] = [layer]
                else:
                    bottom_tensors[inNode].append(layer)
            else:
                for tensor in inNode:
                    if tensor not in bottom_tensors:
                        bottom_tensors[tensor] = [layer]
                    else:
                        bottom_tensors[tensor].append(layer)
    return layer_map, top_tensors, bottom_tensors

def model_to_dag(model):
    dag = Digraph()
    _, top_tensors, bottom_tensors = read_models(model)
    for key in top_tensors:
        if key in bottom_tensors:
            for from_layer in top_tensors[key]:
                for top_layer in bottom_tensors[key]:
                    if from_layer != top_layer:
                        from_node = Node(from_layer.name, from_layer)
                        top_node = Node(top_layer.name, top_layer)
                        dag.addEdge(from_node, top_node)
    return dag