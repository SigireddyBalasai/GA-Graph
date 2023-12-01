import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_model(dag, input_size, output_size):
    input_layer = tf.keras.layers.Input(shape=input_size, name='input_layer')
    layer_dict = {0: input_layer}
    input_layers = []
    for node in nx.topological_sort(dag):
        predecessors = list(dag.predecessors(node))
        node_type = dag.nodes[node]['state']
        if 'input' in node_type:
            input_layers.append(layer_dict[node])
        elif 'hidden' in node_type:
            if len(predecessors) > 1:
                #print(predecessors,layer_dict,[dag.nodes[node]['layer'] for node in predecessors])
                concat_layer = tf.keras.layers.Concatenate(name=f'{node}_concatenate')([layer_dict[node] for node in predecessors])
                if(dag.nodes[node]['layer'] == 'Dense'):
                    layer_dict[node] = tf.keras.layers.Dense(units=dag.nodes[node]['units'], activation=dag.nodes[node]['activation'], name= str(node))(concat_layer)
                elif(dag.nodes[node]['layer'] == 'DropOut'):
                    layer_dict[node] = tf.keras.layers.Dropout(rate=dag.nodes[node]['rate'], name= str(node))(concat_layer)
                elif(dag.nodes[node]['layer'] == 'Conv2D'):
                    layer_dict[node] = tf.keras.layers.Conv2D(filters=dag.nodes[node]['filters'], kernel_size=dag.nodes[node]['kernel_size'], activation=dag.nodes[node]['activation'], name= str(node),padding='same',strides=1)(concat_layer)
                elif(dag.nodes[node]['layer'] == 'MaxPooling2D'):
                    layer_dict[node] = tf.keras.layers.MaxPooling2D(pool_size=dag.nodes[node]['pool_size'],padding='same',strides=1, name= str(node))(concat_layer)
                elif(dag.nodes[node]['layer'] == 'AveragePooling2D'):
                    layer_dict[node] = tf.keras.layers.AveragePooling2D(pool_size=dag.nodes[node]['pool_size'],padding='same',strides=1, name= str(node))(concat_layer)

            else:
                if(dag.nodes[node]['layer'] == 'Dense'):
                    layer_dict[node] = tf.keras.layers.Dense(units=dag.nodes[node]['units'], activation=dag.nodes[node]['activation'], name= str(node))(layer_dict[predecessors[0]])
                elif(dag.nodes[node]['layer'] == 'DropOut'):
                    layer_dict[node] = tf.keras.layers.Dropout(rate=dag.nodes[node]['rate'], name= str(node))(layer_dict[predecessors[0]])
                elif(dag.nodes[node]['layer'] == 'Conv2D'):
                    layer_dict[node] = tf.keras.layers.Conv2D(filters=dag.nodes[node]['filters'], kernel_size=dag.nodes[node]['kernel_size'],strides=1,activation=dag.nodes[node]['activation'], name= str(node),padding='same')(layer_dict[predecessors[0]])
                elif(dag.nodes[node]['layer'] == 'MaxPooling2D'):
                    layer_dict[node] = tf.keras.layers.MaxPooling2D(pool_size=dag.nodes[node]['pool_size'],padding='same',strides=1,name= str(node))(layer_dict[predecessors[0]])
                elif(dag.nodes[node]['layer'] == 'AveragePooling2D'):
                    layer_dict[node] = tf.keras.layers.AveragePooling2D(pool_size=dag.nodes[node]['pool_size'],padding='same',strides=1,name= str(node))(layer_dict[predecessors[0]])
        elif 'output' in node_type:
            if len(predecessors) > 1:
                print(predecessors,layer_dict)
                concat_layer = tf.keras.layers.Concatenate(name=f'{node}_concatenate')([layer_dict[node] for node in predecessors])
                normalization = tf.keras.layers.GlobalAveragePooling2D(name=f'{node}_normalization')(concat_layer)
                flatten = tf.keras.layers.Flatten(name=f'{node}_flatten')(normalization)
                layer_dict[node] = tf.keras.layers.Dense(units=output_size, activation='sigmoid', name= str(node))(flatten)
            else:
                normalization = tf.keras.layers.GlobalAveragePooling2D(name=f'{node}_normalization')(layer_dict[predecessors[0]])
                flatten = tf.keras.layers.Flatten(name=f'{node}_flatten')(normalization)
                layer_dict[node] = tf.keras.layers.Dense(units=output_size, activation='sigmoid', name= str(node))(flatten)

    output_layers = [layer_dict[node] for node in dag.nodes if 'output' in dag.nodes[node]['state']]
    print(output_layers)
    model = tf.keras.models.Model(inputs=[layer_dict[0]], outputs=output_layers)

    return model