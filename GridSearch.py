######################################## importing libraries ########################################

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() #Scaling data between 0 and 1
from torch import nn
from sysidentpy.metrics import mean_squared_error
from sysidentpy.metrics import r2_score
from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function._basis_function import Polynomial
import matplotlib.pyplot as plt

######################################## collecting and scaling data ########################################
paths_train = ['C:/Users/Avell/Desktop/TCC/Tests0.txt_mean_11_39_02.txt',
               'C:/Users/Avell/Desktop/TCC/Tests1.txt_mean_08_33_39.txt',
               'C:/Users/Avell/Desktop/TCC/Tests1.txt_mean_11_15_31.txt',
               'C:/Users/Avell/Desktop/TCC/Tests2.txt_mean_08_26_59.txt'] # list with the train data paths

paths_test = ['C:/Users/Avell/Desktop/TCC/Tests_valid0.txt_mean_09_41_14.txt'] # list with the test data paths
              
paths_mpc = ['C:/Users/Avell/Desktop/TCC/Tests_MPC0.txt_mean_12_38_39.txt'] # list with the mpc validation data paths


X_train, y_train, X_test, y_test, X_mpc, y_mpc = [], [], [], [], [], []

cb_scale = []
cold_scale = []
hot_scale = []
y_scale = []

cut = 10 #how many seconds to insert a point in the dataset

c = [18796, 9148, 20815, 10970] #exclude wrong data points
i = 0

for path in paths_train: #training data
  a = pd.read_csv(path, skiprows = 6, sep = '\t', header=0, encoding='cp1252')
  a = pd.DataFrame(a[:c[i]])
  aux1 = []
  aux2 = []
  i = i+1
  for j in range(0, int(len(a)/cut)):
    aux1.append(a.iloc[int(cut*j),[0,1,2,3,4,47,53,54]])                              
    aux2.append([a.iloc[int(cut*j),43]])                                              
    cb_scale.append(a.iloc[int(cut*j),[47]])
    cold_scale.append(a.iloc[int(cut*j),[53]])
    hot_scale.append(a.iloc[int(cut*j),[54]])
    y_scale.append([a.iloc[int(cut*j),43]])
  
  X_train.append(np.asarray(aux1))
  y_train.append(np.asarray(aux2))



for path in paths_test: #test data
  b = pd.read_csv(path, skiprows = 6, sep = '\t', header=0, encoding='cp1252')
  b = pd.DataFrame(b)
  aux1 = []
  aux2 = []
  for j in range(0, int(len(b)/cut)):
    aux1.append(b.iloc[int(cut*j),[0,1,2,3,4,47,53,54]])                              
    aux2.append([b.iloc[int(cut*j),43]])                                             

  X_test.append(np.asarray(aux1))
  y_test.append(np.asarray(aux2))



for path in paths_mpc: #mpc test data
  c = pd.read_csv(path, skiprows = 6, sep = '\t', header=0, encoding='cp1252')
  c = pd.DataFrame(b)
  aux1 = []
  aux2 = []
  for j in range(0, int(len(b)/cut)):
    aux1.append(b.iloc[int(cut*j),[0,1,2,3,4,47,53,54]])                                    
    aux2.append([b.iloc[int(cut*j),43]])                                                                             

  X_mpc.append(np.asarray(aux1))
  y_mpc.append(np.asarray(aux2))

  #Scaling data based on training data
cbscale = scaler.fit(cb_scale)
coldscale = scaler.fit(cold_scale)
hotscale = scaler.fit(hot_scale)
yscale = scaler.fit(y_scale)

for i in range(0, len(X_train)):
  for j in range(0, len(X_train[i])):
    X_train[i][j][5] = cbscale.transform(X_train[i][j][5].reshape(1,-1))
    X_train[i][j][6] = coldscale.transform(X_train[i][j][6].reshape(1,-1))
    X_train[i][j][7] = hotscale.transform(X_train[i][j][7].reshape(1,-1))
    y_train[i][j] = yscale.transform(y_train[i][j].reshape(1,-1))

for i in range(0, len(X_test)):
  for j in range(0, len(X_test[i])):
    X_test[i][j][5] = cbscale.transform(X_test[i][j][5].reshape(1,-1))
    X_test[i][j][6] = coldscale.transform(X_test[i][j][6].reshape(1,-1))
    X_test[i][j][7] = hotscale.transform(X_test[i][j][7].reshape(1,-1))
    y_test[i][j] = yscale.transform(y_test[i][j].reshape(1,-1))

for i in range(0, len(X_mpc)):
  for j in range(0, len(X_mpc[i])):
    X_mpc[i][j][5] = cbscale.transform(X_mpc[i][j][5].reshape(1,-1))
    X_mpc[i][j][6] = coldscale.transform(X_mpc[i][j][6].reshape(1,-1))
    X_mpc[i][j][7] = hotscale.transform(X_mpc[i][j][7].reshape(1,-1))
    y_mpc[i][j] = yscale.transform(y_test[i][j].reshape(1,-1))

########################################defining the grid for the grid search########################################

xlags = [[[1],[1],[1],[1],[1],[1],[1],[1]],
         [[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]],
         [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
         [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],
         [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
         [[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]],
         [[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]],
         [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]],
         [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
         [[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]]
        ] #range of input lags

ylags = [1,2,3,4,5,6,7,8,9,10] #range of output lags

neurons = [0.75,1,1.25,1.5,1.75,2,2.5,3] #range of neurons

nbs_hl = [2,3,4,5,6] #range of hidden layers
######################################## grid searching with 50 epochs ########################################
best_result = ''
results_compiled = []
best_results_compiled = []
r2_compiled = []
best_r2_results = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
r2_best = -50
models = 0
o = 0
tloss_compiled = []
vloss_compiled = []
for xlag in xlags:
    for ylag in ylags:
        for neuron in neurons:
            for nb_hl in nbs_hl: 
                nlag = np.max(xlag)*8 + ylag
                n = int(neuron*(nlag)) #Calculate the number of neurons

                basis_function = Polynomial(degree=1)
                narx_net = NARXNN(
                    ylag=ylag, #select ylag
                    xlag=xlag, #select xlag
                    basis_function=basis_function,
                    model_type="NARMAX",
                    loss_func='mse_loss',
                    optimizer='Adam',
                    epochs=50, #chose epochs
                    verbose=True,
                    learning_rate = 3e-04,
                    optim_params={'betas': (0.9, 0.999),
                        'eps': 1e-05} # optional parameters of the optimizer
                )
                torch.manual_seed(1) #random state seed
                class NARX(nn.Module):
                    def __init__(self, nlag, nb_hidden_layer, n):
                        super().__init__()
                        self.lin1 = nn.Linear(nlag, n) #The first layer must have the same number as the regressors
                        self.fcs = nn.ModuleList(nb_hidden_layer*[nn.Linear(n, n)])
                        self.lin3 = nn.Linear(n, 1)
                        self.tanh = nn.Tanh()
                        self.drop = nn.Dropout(p = 0.2)
                        
                        for m in self.modules():
                            if isinstance(m, nn.Linear):
                                nn.init.xavier_normal_(m.weight)

                    def forward(self, xb):
                        z = self.lin1(xb)
                        z = self.tanh(z)
                        for l in self.fcs:
                            z = self.tanh(l(z))
                        z = self.lin3(z)

                        return z

                narx_net.net = NARX(nlag = nlag, nb_hidden_layer=nb_hl, n=n)

                narx_net.fit(X=X_train[0], y=y_train[0], X_test=X_test[0], y_test=y_test[0])
                yhat = narx_net.predict(X=X_test[0], y=y_test[0])
                r2 = r2_score(y_test[0].flatten(), yhat.flatten())
                tloss = np.min(narx_net.train_loss) # Training Loss
                best_epoch_train = narx_net.train_loss.index(tloss)
                vloss = np.min(narx_net.val_loss) # Validation Loss
                best_epoch_valid = narx_net.val_loss.index(vloss)
                tloss_compiled.append(narx_net.train_loss)
                vloss_compiled.append(narx_net.val_loss)

                results = {'model number':models,
                            'xlag': np.max(xlag),
                            'ylag': ylag,
                            'neurons':n,
                            'hidden layers': nb_hl,
                            'best training loss':tloss,
                            'best epoch train': best_epoch_train,
                            'best validation loss':vloss,
                            'best epoch valid':best_epoch_valid,
                            'r2'      :r2
                        }

                results_compiled.append(results)
                if r2>r2_best: 
                    best_result = results
                    best_prediction = yhat
                    r2_best = r2
                if r2>np.min(best_r2_results):
                    index = best_r2_results.index(np.min(best_r2_results))
                    best_r2_results[index]=r2
                    if len(best_results_compiled)>=20:
                        del best_results_compiled[index]
                    best_results_compiled.insert(index,results)

                models = models+1
                
best_results_compiled = sorted(best_results_compiled, key=lambda d: d['r2'], reverse=True) # sort the results according to r2
results_compiled = sorted(results_compiled, key=lambda d: d['r2'], reverse=True) 
f = open('C:/Users/Avell/Desktop/TCC/GridSearch.txt','w+') #save all the results
f.write(str(best_result)+'\n')
for r in results_compiled:
  f.write(str(r)+'\n')
f.close()

f = open('C:/Users/Avell/Desktop/TCC/GridSearch_best.txt','w+') #save the 20 best reults
f.write(str(best_result)+'\n')
for r in best_results_compiled:
  f.write(str(r)+'\n')
f.close()

######################################## train for 300 epochs the 20 best results ########################################
best_300epochs = []
tloss_compiled = []
vloss_compiled = []

r2_best = -50
best_result300 = ''
for results in best_results_compiled:
    xlag = xlags[int(results['xlag']-1)]
    ylag = int(results['ylag'])
    nlag = np.max(xlag)*8 + ylag
    n = results['neurons'] #Calculate the number of neurons

    basis_function = Polynomial(degree=1)

    narx_net = NARXNN(
        ylag=ylag, #select ylag
        xlag=xlag, #select xlag
        basis_function=basis_function,
        model_type="NARMAX",
        loss_func='mse_loss',
        optimizer='Adam',
        epochs=100, #chose epochs
        verbose=True,
        learning_rate = 3e-04,
        optim_params={'betas': (0.9, 0.999),
            'eps': 1e-05} # optional parameters of the optimizer
        )

    class NARX(nn.Module):
        def __init__(self, nlag, nb_hidden_layer, n):
            super().__init__()
            self.lin1 = nn.Linear(nlag, n) #The first layer must have the same number as the regressors
            self.fcs = nn.ModuleList(nb_hidden_layer*[nn.Linear(n, n)])
            self.lin3 = nn.Linear(n, 1)
            self.tanh = nn.Tanh()
            self.drop = nn.Dropout(p = 0.2)
                        
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

        def forward(self, xb):
            z = self.lin1(xb)
            z = self.tanh(z)
            for l in self.fcs:
                z = self.tanh(l(z))
            z = self.lin3(z)

            return z

    narx_net.net = NARX(nlag = nlag, nb_hidden_layer=nb_hl, n=n)

    narx_net.fit(X=X_train[0], y=y_train[0], X_test=X_test[0], y_test=y_test[0])
    yhat = narx_net.predict(X=X_test[0], y=y_test[0])
    r2 = r2_score(y_test[0].flatten(), yhat.flatten())
    tloss = np.min(narx_net.train_loss) # Training Loss
    best_epoch_train = narx_net.train_loss.index(tloss)
    vloss = np.min(narx_net.val_loss) # Validation Loss
    best_epoch_valid = narx_net.val_loss.index(vloss)
    tloss_compiled.append(narx_net.train_loss)
    vloss_compiled.append(narx_net.val_loss)
    results = {'xlag': xlag,
                'ylag': ylag,
                'neurons':n,
                'hidden layers:':nb_hl,
                'best training loss':tloss,
                'best epoch train': best_epoch_train,
                'best validation loss':vloss,
                'best epoch valid':best_epoch_valid,
                'r2'      :r2
                       }

    best_300epochs.append(results)

    if r2<r2_best:
        best_result300 = results
        best_prediction300 = yhat
        r2_best = r2

best_300epochs = sorted(best_300epochs, key=lambda d: d['r2'], reverse=True) 

f = open('C:/Users/Avell/Desktop/TCC/GridSearch_bests300.txt','w+') # save the results obtained wiuth 300 epochs
f.write(str(best_result300)+'\n')
for r in best_300epochs:
  f.write(str(r)+'\n')
f.close()

f = open('C:/Users/Avell/Desktop/TCC/bests300_tlosses.txt','w+') # save the training loss curves for each parameter combination
f.write(str(tloss_compiled)+'\n')
f.close()

f = open('C:/Users/Avell/Desktop/TCC/bests300_vlosses.txt','w+') # save the valid loss curves for each parameter combination
f.write(str(vloss_compiled)+'\n')
f.close()
