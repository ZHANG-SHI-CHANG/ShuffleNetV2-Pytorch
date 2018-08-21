import os

root = os.getcwd()

#DataLoader
TrainPath = os.path.join(root,'dataset','train')
ValPath = os.path.join(root,'dataset','val')
TestPath = os.path.join(root,'dataset','test')
BatchSize = 2
Workers = 2

#Distributed
useDistributed = False#False           ,True
DistBackend = 'gloo'
DistUrl = 'tcp://224.66.41.62:23456'
WorldSize = 1

#GPU
useGPU = False        #(0,1)¡¢1¡¢False ,False

#model
usePretrain = False
StartEpoch = 0
Epoch = 10
Lr = 0.1
Momentum = 0.9
WeightDecay = 1e-4
ModelPath = os.path.join(root,'model_store')
PretrainModelPath = os.path.join(ModelPath,'model_best.pth')
PrintFreq = 10
Mode = 'Test'#'Train','Val','Test'