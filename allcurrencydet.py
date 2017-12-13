import math
import numpy as np
import cv2
import tensorflow as tf
import matplotlib as plt
from tensorflow.python.framework import ops
np.random.seed(1)
# GRADED FUNCTION: one_hot_matrix

def relu(Z):

    A=np.log(1+np.exp(Z))
    return A
def convert_to_activated(Z):
    m,n=Z.shape
    for i in range(n):
        sums=0
        for j in range(m):
            sums+=Z[j][i]
        for k in range(m):
            Z[j][i]/=sums
    return Z


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """

    ### START CODE HERE ###

    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C,name="C")

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels,C,axis=0)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###

    return one_hot

# GRADED FUNCTION: create_placeholders

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X =tf.placeholder(tf.float32,shape=[n_x,None],name = 'X')
    Y = tf.placeholder(tf.float32,shape=[n_y,None], name = 'Y')
    ### END CODE HERE ###

    return X, Y

# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [25,5400], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [4,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [4,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                               # A1 = relu(Z1)
    Z2 =  tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,Z2),b3)                                      # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###

    return Z3

# GRADED FUNCTION: compute_cost

def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    #z = tf.placeholder(tf.float32, name = "z")
    #y = tf.placeholder(tf.float32, name = "y")

    # Use the loss function (approx. 1 line)
    #cost = tf.nn.sigmoid_cross_entropy_with_logits(logits =z,  labels =y)

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)


    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits, labels = labels))

    ### END CODE HERE ###

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()

    ### END CODE HERE ###
    #print(X_train)
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)

    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            seed = seed + 1

            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
                ### END CODE HERE ###
            #print minibatch_cost
            #epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch,minibatch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(minibatch_cost)

        # plot the cost

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        #correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        #print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

image=cv2.imread('/home/anand-ros/Desktop/neuralnetwork/10.jpg')
image1=cv2.imread('/home/anand-ros/Desktop/neuralnetwork/20.jpg')
image2=cv2.imread('/home/anand-ros/Desktop/neuralnetwork/50.jpg')
image3=cv2.imread('/home/anand-ros/Desktop/neuralnetwork/100.jpg')

arr=np.reshape(image,image.shape[0]*image.shape[1]*3)
arr1=np.reshape(image1,image1.shape[0]*image1.shape[1]*3)
arr2=np.reshape(image2,image2.shape[0]*image2.shape[1]*3)
arr3=np.reshape(image3,image3.shape[0]*image3.shape[1]*3)
X1=np.asmatrix([arr,arr1,arr2,arr3])
X1=X1.T
X1 = X1/255.0
Y1=np.asmatrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
############################################################################
#Give input here
X_test=np.asmatrix([arr2]).T
X_test=X_test/255.0
############################################################################


Y_test=np.asmatrix([1,0,0,0]).T
parameters = model(X1, Y1, X_test, Y_test)
#print parameters

W1 = parameters['W1']
b1 = parameters['b1']
W2 = parameters['W2']
b2 = parameters['b2']
W3 = parameters['W3']
b3 = parameters['b3']

### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
Z1 = np.dot(W1,X_test)+b1                                              # Z1 = np.dot(W1, X) + b1
#print Z1
A1 = relu(Z1)                                             # A1 = relu(Z1)
#print A1
Z2 =  np.dot(W2,A1)+b2                                              # Z2 = np.dot(W2, a1) + b2
#A2 = np.relu(Z2)                                              # A2 = relu(Z2)
Z3 = np.dot(W3,Z2)+b3                                      # Z3 = np.dot(W3,Z2) + b3
### END CODE HERE ###

pred=np.argmax(Z3)
if pred==0:
    print "Rs.10 note"
elif pred==1:
    print "Rs.20 note"
elif pred==2:
    print "Rs.50 note"
else:
    print "Rs.100 note"
