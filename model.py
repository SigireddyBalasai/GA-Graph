import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Zpad import ZeroPadConcatLayer
from PoolingCustom import MaxPoolingCustom, AveragePoolingCustom
from CustomConv2D import Conv2DPadLayer
from CustomLocallyConnected import ConLocPadLayer
import random

class PreProcessing(tf.keras.layers.Layer):
    """this will take a 2d image and will apply flip horizantal vertical and pass to array"""
    def __init__(self, **kwargs):
        super(PreProcessing, self).__init__(**kwargs)
        
    def call(self,inputs):
        choices = random.choice(['horizantal','vertical','both'])
        if choices == 'horizantal':
            inputs = tf.image.flip_left_right(inputs)
        if choices == 'vertical':
            inputs = tf.image.flip_up_down(inputs)
        if choices == 'both':
            inputs = tf.image.flip_up_down(tf.image.flip_left_right(inputs))
        return inputs
    

def create_model(dag ,input_size, output_size):
    input_layer = PreProcessing(tf.keras.layers.Input(input_size, name='input_layer'))
    layer_dict = {'1-0': input_layer}
    input_layers = []
    #print(dag.nodes)
    #print(dag.edges)
    #print([node for node in nx.topological_sort(dag)])
    for node in nx.topological_sort(dag):
        #print(node)
        predecessors = list(dag.predecessors(node))
        predecessors.sort()
        node_type = dag.nodes[node]['state']
        print(node_type,node,predecessors,layer_dict)
        #print(node_type,node)
        if 'input' == node_type:
            input_layers.append(layer_dict[node])
        elif 'hidden_1' == node_type:
            layer_type = dag.nodes[node]['layer']
            
            #print(node,layer_type)
            #print(predecessors)
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
            elif layer_type == 'Conv2DTranspose':
                layer_dict[node] = tf.keras.layers.Conv2DTranspose(filters=dag.nodes[node]['filters'], kernel_size=dag.nodes[node]['kernel_size'], activation=dag.nodes[node]['activation'], name=f'{node}.{"".join(list(map(str,predecessors)))}conv2dtranspose')(concat_layer)
        
        elif 'hidden_2' == node_type:
            layer_type = dag.nodes[node]['layer']
            #print(layer_type)
            #print(predecessors)
            layers = [layer_dict[node] for node in predecessors]
            if(len(layers) > 1):
                concat_layer = tf.keras.layers.Concatenate(name=f'{node}_concatenate')(layers)
                concat_layer = tf.keras.layers.Dropout(rate=0.5, name=f'{node}_dropout')(concat_layer)
            else:
                concat_layer = layers[0]
            if layer_type == 'DropOut':
                layer_dict[node] = tf.keras.layers.Dropout(rate=dag.nodes[node]['rate'], name=f'{node}.{"".join(list(map(str,predecessors)))}dropout')(concat_layer)
            elif layer_type == 'Dense':
                layer_dict[node] = tf.keras.layers.Dense(units=dag.nodes[node]['units'], activation=dag.nodes[node]['activation'], name=f'{node}.{"".join(list(map(str,predecessors)))}dense')(concat_layer)
            elif layer_type == 'LSTM':
                resize = tf.keras.layers.Reshape((1,concat_layer.shape[1]), name=f'{node}_reshape')(concat_layer)
                layer_dict[node] = tf.keras.layers.LSTM(units=dag.nodes[node]['units'], activation=dag.nodes[node]['activation'], name=f'{node}.{"".join(list(map(str,predecessors)))}lstm')(resize)
        
        elif 'transition' == node_type:
            #print(layer_type)
            #print(predecessors)
            layers = [layer_dict[node] for node in predecessors]
            if(len(layers) > 1):
                concat_layer = ZeroPadConcatLayer(name=f'{node}_concatenate')(layers)
                concat_layer = tf.keras.layers.Dropout(rate=0.5, name=f'{node}_dropout')(concat_layer)
            else:
                concat_layer = layers[0]
            
            try:
                conv_max = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='relu', name=f'{node}_conv2d')(concat_layer)
                #conv_max = tf.keras.layers.MaxPooling2D(pool_size=conv_max.shape[1]//5, name=f'{node}_maxpooling2d')(conv_max)
                concat_layer = tf.keras.layers.Flatten(name=f'{node}___flatten')(conv_max)
                #concat_layer = tf.keras.layers.GlobalAveragePooling2D(name=f'{node}_globalaveragepooling2d',keepdims=True,data_format='channels_last')(concat_layer)
            except:
                pass
            layer_dict[node] = tf.keras.layers.Flatten(name=f'{node}_flatten')(concat_layer)

        elif 'output' == node_type:
            #print(predecessors)
            if(len(predecessors) > 1):
                concat_layer = tf.keras.layers.Concatenate(name=f'{node}_concatenate')([layer_dict[node] for node in predecessors])
            else:
                concat_layer = layer_dict[predecessors[0]]
            drop = tf.keras.layers.Dropout(rate=0.8, name=f'{node}_dropout')(concat_layer)
            #pool = tf.keras.layers.Dense(name=f'{node}_globalaveragepooling2d')(drop)
            layer_dict[node] = tf.keras.layers.Dense(units=output_size,activation='softmax', name=f'node.{"".join(list(map(str,predecessors)))}dense')(drop)

    output_layers = [layer_dict[node] for node in dag.nodes if 'output' in dag.nodes[node]['state']]
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layers)

    return model