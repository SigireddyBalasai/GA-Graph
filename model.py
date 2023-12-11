import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Zpad import ZeroPadConcatLayer
from PoolingCustom import MaxPoolingCustom, AveragePoolingCustom
from CustomConv2D import Conv2DPadLayer
from CustomLocallyConnected import ConLocPadLayer

def create_model(dag, input_size, output_size):
    input_layer = tf.keras.layers.Input(input_size, name='input_layer')
    layer_dict = {0: input_layer}
    input_layers = []
    print(dag.nodes)
    print(dag.edges)
    print(nx.topological_sort(dag))
    for node in nx.topological_sort(dag):
        predecessors = list(dag.predecessors(node))
        predecessors.sort()
        node_type = dag.nodes[node]['state']
        if 'input' == node_type:
            input_layers.append(layer_dict[node])

        elif 'hidden' == node_type:
            layer_type = dag.nodes[node]['layer']
            layers = [layer_dict[node] for node in predecessors]
            if(len(layers) > 1):
                concat_layer = ZeroPadConcatLayer(name=f'{node}_concatenate')(layers)
                concat_layer = tf.keras.layers.Dropout(rate=0.5, name=f'{node}_dropout')(concat_layer)
            else:
                concat_layer = layers[0]
            if layer_type == 'DropOut':
                layer_dict[node] = tf.keras.layers.Dropout(rate=dag.nodes[node]['rate'], name=f'{node}.{"".join(list(map(str,predecessors)))}dropout')(concat_layer)
            elif layer_type == 'Dense':
                layer_dict[node] = tf.keras.layers.Dense(units=dag.nodes[node]['units'], activation=dag.nodes[node]['activation'], name=f'{node}.{"".join(list(map(str,predecessors)))}dense')(concat_layer)
            elif layer_type == 'Conv2D':
                layer_dict[node] = Conv2DPadLayer(filters=dag.nodes[node]['filters'], kernel_size=dag.nodes[node]['kernel_size'], activation=dag.nodes[node]['activation'], name=f'{node}.{"".join(list(map(str,predecessors)))}conv2d', strides=1)(concat_layer)
            elif layer_type == 'MaxPooling2D':
                layer_dict[node] = MaxPoolingCustom(pool_size=dag.nodes[node]['pool_size'], name=f'{node}.{"".join(list(map(str,predecessors)))}maxpooling2d')(concat_layer)
            elif layer_type == 'AveragePooling2D':
                layer_dict[node] = AveragePoolingCustom(pool_size=dag.nodes[node]['pool_size'], name=f'{node}.{"".join(list(map(str,predecessors)))}averagepooling2d')(concat_layer)
            elif layer_type == 'GlobalAveragePooling2D':
                layer_dict[node] = tf.keras.layers.GlobalAveragePooling2D(name=f'{node}.{"".join(list(map(str,predecessors)))}globalaveragepooling2d',keepdims=True,data_format='channels_last')(concat_layer)
            #elif layer_type == 'LocallyConnected2D':
                #layer_dict[node] = ConLocPadLayer(filters=dag.nodes[node]['filters'], kernel_size=dag.nodes[node]['kernel_size'], activation=dag.nodes[node]['activation'], name=f'{node}.{"".join(list(map(str,predecessors)))}locallyconnected2d')(concat_layer)
            

        elif 'output' == node_type:
            if(len(predecessors) > 1):
                concat_layer = ZeroPadConcatLayer(name=f'{node}_concatenate')([layer_dict[node] for node in predecessors])
            else:
                concat_layer = layer_dict[predecessors[0]]
            drop = tf.keras.layers.Dropout(rate=0.8, name=f'{node}_dropout')(concat_layer)
            #pool = tf.keras.layers.Dense(name=f'{node}_globalaveragepooling2d')(drop)
            flatten = tf.keras.layers.Flatten(name=f'{node}_flatten')(drop)
            layer_dict[node] = tf.keras.layers.Dense(units=output_size,activation='softmax', name=f'node.{"".join(list(map(str,predecessors)))}dense')(flatten)

    output_layers = [layer_dict[node] for node in dag.nodes if 'output' in dag.nodes[node]['state']]
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layers)

    return model