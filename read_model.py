import os
from utils import Digraph
from models.vgg16 import VGG16
from keras.models import load_model
from evaluator import EntropyEvaluator
from utils import read_models

if __name__ == '__main__':
    os.chdir('D:\data\keras_models')
    model = VGG16(include_top=True)
    dag = Digraph()
    layer_map, top_tensors, bottom_tensors = read_models(model)
    for key in top_tensors:
        if key in bottom_tensors:
            for from_layer in top_tensors[key]:
                for top_layer in bottom_tensors[key]:
                    if from_layer != top_layer:
                        dag.addEdge(from_layer, top_layer)
    print(dag)