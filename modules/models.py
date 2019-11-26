from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2, l1_l2


def create_model(n_inputs=20, n_hidden_layers=1, n_nodes=20, dropout_rate=0., l1=0., l2=0., lr=0.001):
  model = Sequential()

  print("\nSetting up neural network with {0} hidden layers and {1} nodes in each layer".format(n_hidden_layers, n_nodes))
  print("\n")

  kernel_regularizer = l1_l2(l1=l1, l2=l2)

  # Add input + first hidden layer
  model.add(Dense(n_nodes, input_dim=n_inputs, activation="relu", 
                  use_bias=True, 
                  kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                  kernel_regularizer=kernel_regularizer, bias_regularizer=None, activity_regularizer=None, 
                  kernel_constraint=None, bias_constraint=None))

  # Add hidden layers
  hidden_layers_counter = 1
  for i_hidden_layer in range(n_hidden_layers-1):

    if dropout_rate > 0.:
      # Add dropout layer before every normal hidden layer
      print("Adding droput layer with dropout rate of", dropout_rate)
      model.add(Dropout(dropout_rate, noise_shape=None, seed=None))

    hidden_layers_counter += 1
    print("Adding hidden layer #", hidden_layers_counter)
    print("\n")
    model.add(Dense(n_nodes, activation='relu',
                    use_bias=True, 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                    kernel_regularizer=kernel_regularizer, bias_regularizer=None, activity_regularizer=None, 
                    kernel_constraint=None, bias_constraint=None))

  # Add output layer
  model.add(Dense(1, activation="sigmoid"))

  model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                loss="binary_crossentropy",
                #metrics=['binary_crossentropy']#, 'accuracy']#, 'balanced_accuracy_score']
                )
  return model

