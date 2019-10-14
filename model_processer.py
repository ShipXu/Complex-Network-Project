from utils import Node
from utils import Digraph
from keras.utils import plot_model
from models.lenet import Lenet
from models.resnet50 import ResNet50
from models.mobilenet import MobileNet

def visualize_model(model, path='models_visualization/model.png'):
    plot_model(model, to_file=path)

def read_models(model):
    top_tensors = {}
    bottom_tensors = {}
    layer_map = {}
    for layer in model.layers:
        layer_map[layer.name] = layer
        for i in range(len(layer._inbound_nodes)):
            outTensor = layer.get_output_at(i)
            if not isinstance(outTensor, list):
                if outTensor not in top_tensors:
                    top_tensors[outTensor] = [layer]
                else:
                    top_tensors[outTensor].append(layer)
            else:
                for tensor in outTensor:
                    if tensor not in top_tensors:
                        top_tensors[tensor] = [layer]
                    else:
                        top_tensors[tensor].append(layer)

        for i in range(len(layer._inbound_nodes)):
            inTensor = layer.get_input_at(i)
            if not isinstance(inTensor, list):
                if inTensor not in bottom_tensors:
                    bottom_tensors[inTensor] = [layer]
                else:
                    bottom_tensors[inTensor].append(layer)
            else:
                for tensor in inTensor:
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

if __name__ == '__main__':
    # model = Lenet()
    # model = ResNet50()
    model = MobileNet()
    visualize_model(model, path='models_visualization/MobileNet.png')
    dag = model_to_dag(model)
    print(dag)