import tensorflow as tf
import numpy as np

def sampledlogcost(inputsize, embedsize, outputsize, batchsize=None, numindices=None):
    
    '''
    Creates a graph that implements the sampled logistic cost function
    off a two layer neural network, where the hidden layer as the 
    embedding.
    
       features -> Weights 0 ---(hidden-units)---> Weights 1 ---> cost
    
    Input:
    
       - inputsize: input feature size
       - embedsize: embedding size (how many hidden units)
       - outputsize: your label space size (only useful during training)
       
    Output:
    
       - dictionary of nodes in the graph with the following keys:
         - placeholder graph nodes used for inputs 
           > indices: specify the locations of the appropriate output vectors
             to use in optimization. The size is ( batchsize, #samples )                  
           > labels: a matrix/list of size ( batchsize, #samples ) that specifies
             whether or not we have a positive or negative sample with [+/- 1]
           > features: input features used as the input to the neural network
         - placeholder graph nodes used for outputs
           > hidden_units: when passing a feature into the neural network, this
             results in the embedding vector, which after training will yield
             the appropriate feature vector
           > cost: the cost function used for training
    '''

    # Dimensionality of the inputs
    # indices: batchsize x numindices 
    # labels: batchsize x numindices
    # features: batchsize x inputsize
    indices = tf.placeholder(tf.int32, shape=(batchsize,numindices))
    labels  = tf.placeholder(tf.int32, shape=(batchsize,numindices))
    features= tf.placeholder(tf.float32, shape=(batchsize, inputsize))

    # Initialize the variables
    invecs  = tf.Variable( tf.truncated_normal((inputsize, embedsize), stddev=0.4) )
    outvecs = tf.Variable( tf.truncated_normal((outputsize, embedsize), stddev=0.4) )

    # Do the computation
    hidden_layer = tf.matmul(features, invecs)
    hidden_layer_expanded = tf.expand_dims(hidden_layer, 1)
    indices_feed = tf.expand_dims(indices, 2)
    sampledvecs = tf.gather_nd( outvecs, indices_feed )
    dots = tf.reduce_sum(hidden_layer_expanded * sampledvecs, axis=2)
    logs = -tf.log( tf.sigmoid(dots) )
    unitcost = tf.reduce_sum(logs)
    totcost = tf.reduce_mean(unitcost)
    
    return {'indices': indices, 'labels': labels, 'features': features,
            'hidden_layer': hidden_layer, 'cost': totcost}

# Example graph
inputsize = 28
embedsize = 32
outputsize= 10
batchsize = 40
numsamples = 3

# Create the graph and initialize its variables
graph = sampledlogcost(inputsize, embedsize, outputsize, batchsize=batchsize)
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())

# Example input
npidx = np.random.randint(outputsize, size=(batchsize, numsamples))
nplab = 2*np.random.randint(2, size=( batchsize, numsamples ) )-1
npftr = np.random.randn( batchsize , inputsize )

# Set input to the graph
feed_dict = feed_dict={graph['features']: npftr, graph['indices']: npidx, graph['labels']: nplab}

# Run example through the graph. Get the embedding vector. This is what you run to get the embedding
# vector during inference.
embedding_vector = sess.run(graph['hidden_layer'], feed_dict=feed_dict)

# Run example through to the cost function. This is what you should optimize during training.
cost_value = sess.run(graph['cost'], feed_dict=feed_dict)

