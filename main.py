from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import math
import pandas as pd
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils


validation_data_split = 0.3
num_epochs = 1000
model_batch_size = 128
tb_batch_size = 64
early_patience = 200
drop_out = 0.2
second_dense_layer_nodes  = 2
maxAcc = 0.0
maxIter = 0
C_Lambda = 0.9
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 4
PHI = []
IsSynthetic = False

#Converting all the .csv files to Dataframes
ho_same_pairs = pd.read_csv("human_observed_same_pairs.csv",header=0)
ho_diff_pairs = pd.read_csv("human_observed_diffn_pairs.csv",header=0)
ho_features = pd.read_csv("HumanObserved-Features-Data.csv",header=0, index_col=0)

gsc_same_pairs = pd.read_csv("gsc_same_pairs.csv",header=0)
gsc_same_pairs = gsc_same_pairs.loc[random.sample(range(gsc_same_pairs.shape[0]),7900),:]
gsc_diff_pairs = pd.read_csv("gsc_diffn_pairs.csv",header=0)
gsc_diff_pairs = gsc_diff_pairs.loc[random.sample(range(gsc_diff_pairs.shape[0]),7900),:]
gsc_features = pd.read_csv("GSC-Features.csv",header=0)

#GetConcatDatabase() is used to perform concatenation operation on both the Human Observed and GSC Datasets
def GetConcatDatabase(same_pairs, diff_pairs, features, opt, r):
	temp = pd.merge(same_pairs, features,  how='inner', left_on=['img_id_A'], right_on = ['img_id'])
	XDatasame = pd.merge(temp, features,  how='inner', left_on=['img_id_B'], right_on = ['img_id'])
	XDatasame = XDatasame.drop(columns=['img_id_x', 'img_id_y'])

	temp2 = pd.merge(diff_pairs, features,  how='inner', left_on=['img_id_A'], right_on = ['img_id'])
	XDatadiff = pd.merge(temp2, features,  how='inner', left_on=['img_id_B'], right_on = ['img_id'])
	XDatadiff = XDatadiff.drop(columns=['img_id_x', 'img_id_y'])

	if(opt==1):
		XConcat = pd.concat([XDatasame, XDatadiff.take(np.random.permutation(len(XDatadiff))[:r])])
		XConcat = XConcat.iloc[np.random.permutation(len(XConcat))]
		XConcat = XConcat.reset_index(drop=True)
	else:
		XConcat = pd.concat([XDatasame, XDatadiff])
		XConcat = XConcat.iloc[np.random.permutation(len(XConcat))]
		XConcat = XConcat.reset_index(drop=True)


	Target = XConcat['target']
	XConcat = XConcat.drop(columns=['img_id_A', 'img_id_B', 'target'])
	XConcat = XConcat[(XConcat!= 0).any(1)]
	return XConcat,Target


#GetSubtractDatabase() is used to perform subtraction operation on both the Human Observed and GSC Datasets

def GetSubtractDatabase(same_pairs, diff_pairs, features, opt, r):
	sameA = pd.merge(same_pairs, features,  how='inner', left_on=['img_id_A'], right_on = ['img_id'])
	sameA = sameA.drop(columns=['img_id','img_id_A', 'img_id_B'])
	sameB = pd.merge(same_pairs, features,  how='inner', left_on=['img_id_B'], right_on = ['img_id'])
	sameB = sameB.drop(columns=['img_id','img_id_A', 'img_id_B'])
	same_main = np.abs(sameA.sub(sameB,fill_value=0))
	same_main['target'] = 1

	diffA = pd.merge(diff_pairs, features,  how='inner', left_on=['img_id_A'], right_on = ['img_id'])
	diffA = diffA.drop(columns=['img_id','img_id_B', 'img_id_A'])
	diffB = pd.merge(diff_pairs, features,  how='inner', left_on=['img_id_B'], right_on = ['img_id'])
	diffB = diffB.drop(columns=['img_id','img_id_B', 'img_id_A'])
	diff_main = np.abs(diffA.sub(diffB,fill_value=0))

	if(opt==1):
		XSubtract = pd.concat([same_main, diff_main.take(np.random.permutation(len(diff_main))[:r])])
		XSubtract = XSubtract.iloc[np.random.permutation(len(XSubtract))]
		XSubtract = XSubtract.reset_index(drop=True)
	else:
		XSubtract = pd.concat([same_main[:3800], diff_main[:3800]])
		XSubtract = XSubtract.iloc[np.random.permutation(len(XSubtract))]
		XSubtract = XSubtract.reset_index(drop=True)

	Target = XSubtract['target']
	XSubtract = XSubtract.drop(columns=['target'])
	XSubtract = XSubtract[(XSubtract!= 0).any(1)]
	return XSubtract,Target

# GenerateTrainingTarget() is used to partition 80% of total target values inorder to generate target vector for training.
def GenerateTrainingTarget(Sample_Target,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(Sample_Target)*(TrainingPercent*0.01)))
    t           = Sample_Target[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

# GenerateTrainingData() is used to partition 80% of total raw input data inorder to generate training matrix.
def GenerateTrainingData(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData)*0.01*TrainingPercent))
    d2 = rawData[0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

# GenerateValData() is used to partition 10% of total raw input data inorder to generate validation matrix.
def GenerateValData(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")
    return dataMatrix

# GenerateTargetVector() is used to partition 10% of total target values inorder to generate target vector for validation.
def GenerateValTarget(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# GenerateBigSigma() is used to calculate the covariance matrix which is used in calculating radial basis functions.
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])
        varVect.append(np.var(vct))

    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

# GetScalar() is used to calculate the scalar value to be used in finding radial basis function.
def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L

# GetRadialBasisOut() is used to calculate the exponential value of one radial basis function for Linear Regression.
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

# GetLogBasis() is used to generate log value for error function of Logistic Regression
def GetLogBasis(DataRow,MuRow):
	R = np.subtract(DataRow,MuRow)
	T = np.transpose(R)
	phi_x = np.sqrt(np.dot(R,T))
	return phi_x

# GetPhiMatrix() is used to calculate the design matrix which is the combinaton of multiple radial basis functions and is generated for Linear Regression.
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
	TrainingLen = math.ceil(len(Data)*(TrainingPercent*0.01))
	PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
	BigSigInv = np.linalg.pinv(BigSigma) # Here we calculate inverse of the covariance matrix(Big Sigma)
	for  C in range(0,len(MuMatrix)):
		for R in range(0,int(TrainingLen)):
			PHI[R][C] = GetRadialBasisOut(Data[R], MuMatrix[C], BigSigInv) # Constructing the Design matrix with each element as one radial basis function
    #print ("PHI Generated..")
	return PHI

# GetLogPhiMatrix() is used to calculate the design matrix which is the combinaton of multiple radial basis functions for Logistic Regression.
def GetLogPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
        DataT = np.transpose(Data)
        TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
        #defining the size of design matrix
        PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
        #performing inversion of variance matrix
        for  C in range(0,len(MuMatrix)):
            for R in range(0,int(TrainingLen)):
                #inputting values of design matrix using gaussian basis function
                PHI[R][C] = GetLogBasis(DataT[R], MuMatrix[C])
        return PHI

# GetValTest() is used to generate the output value Y, using Weights W & the Design matrix for Linear Regression
def GetValTestLinear(Val_PHI,W):
    Y = np.dot(W,np.transpose(Val_PHI))
    #print ("Test Out Generated..")
    return Y

# GetValTest() is used to generate the output value Y, using Weights W & the Design matrix for Logistic Regression
def GetValTestLogistic(VAL_PHI,W):
	scaler = MinMaxScaler().fit(VAL_PHI[:])
	VAL_PHI[:] = scaler.transform(VAL_PHI[:])
	Y = 1/(1+np.exp(-np.dot(W,np.transpose(VAL_PHI))))
	#print(Y)
	return Y

# GetErms() is used to find out the root mean square error between expected output & the output we get after performing linear regression.
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
  #  t=0
    accuracy = 0.0
    counter = 0
  #  val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    #print ("Accuracy Generated..")
    #print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

# GetLossFunction() is used to calculate the loss value to update the weights for Logistic Regression
def GetLossFunction(VAL_TEST_OUT,ValDataAct):
        sum = 0.0
        accuracy = 0.0
        counter = 0
        for i in range (0,len(VAL_TEST_OUT)):
            sum = sum - ((ValDataAct[i])*math.log(VAL_TEST_OUT[i]) + (1- ValDataAct[i])*math.log(1- VAL_TEST_OUT[i]))
            if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
                counter+=1
        accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
        return (str(accuracy) + ',' +  str(sum/len(VAL_TEST_OUT)))


# Training & Testing for Linear Regression
def GetLinearRegression(Dataset,Target,forwhat, use):

	TrainTarget = np.array(GenerateTrainingTarget(Target, TrainingPercent))
	TrainData = GenerateTrainingData(Dataset, TrainingPercent)

	ValTarget = np.array(GenerateValTarget(Target, ValidationPercent, len(TrainTarget)))
	ValData = GenerateValData(Dataset, ValidationPercent, len(TrainData))

	TestTarget = np.array(GenerateValTarget(Target,TestPercent, (len(TrainTarget)+len(ValTarget))))
	TestData = GenerateValData(Dataset, ValidationPercent, (len(TrainData)+len(ValData)))

	kmeans = KMeans(n_clusters=M, random_state=0).fit(TrainData) #Using Kmeans clustering we are clustering the Training Data consisting of approximately 58 thousand data to 10 clusters.
	Mu = kmeans.cluster_centers_ # Deriving the centroid of the clusters generated.

	BigSigma     = GenerateBigSigma(np.transpose(Dataset), Mu, TrainingPercent,IsSynthetic)
	Training_PHI = GetPhiMatrix(Dataset, Mu, BigSigma, TrainingPercent)
	Test_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100)
	Val_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)

	print ('----------------------------------------------------')
	print ('--------------Please Wait for a minute!--------------')
	print ('----------------------------------------------------')

	W = np.zeros(len(Mu))
	W = np.transpose(W)
	W_T_Next        = np.add(220, W)
	La           = 2 # The Lambda is the regularization element used to avoid overfitting
	learningRate = 0.01 #The learning rate is used to define the rate at which the model can find the minimum value for the problem. If the rate is too high then the model will take too big steps and may diverge from the solution. On the other hand, if the rate is too low, then the model may not be able to converge to the minimum solution.
	L_Erms_Val   = []
	L_Erms_TR    = []
	L_Erms_Test  = []
	#W_Mat        = []

	for i in range(0,1263):
		#print ('---------Iteration: ' + str(i) + '--------------')
		# These values generated below are used to find the Weigths W using the Gradient Decent solution.
		Delta_E_D     = -np.dot((TrainTarget[i] - np.dot(W_T_Next,Training_PHI[i])),Training_PHI[i])# Here we calculate the Delta Change in E_D value
		La_Delta_E_W  = np.dot(La,W_T_Next) # # Here we calculate the Delta Change in E_W value. Here we also include the regularization factor(Lamda) to avoid overfitting of the model
		Delta_E       = np.add(Delta_E_D,La_Delta_E_W) # Here the change(Delta) in Error is calculated
		Delta_W       = -np.dot(learningRate,Delta_E) # Finally the change in weights is calculated based on the learning rate(eta)
		W_T_Next      = W_T_Next + Delta_W # The weights are updated with the delta value

		#-----------------TrainingData Accuracy---------------------#
		TR_TEST_OUT   = GetValTestLinear(Training_PHI,W_T_Next)
		Erms_TR       = GetErms(TR_TEST_OUT,TrainTarget)
		L_Erms_TR.append(float(Erms_TR.split(',')[1]))

		#-----------------ValidationData Accuracy---------------------#
		VAL_TEST_OUT  = GetValTestLinear(Val_PHI,W_T_Next)
		Erms_Val      = GetErms(VAL_TEST_OUT,ValTarget)
		L_Erms_Val.append(float(Erms_Val.split(',')[1]))

		#-----------------TestingData Accuracy---------------------#
		TEST_OUT      = GetValTestLinear(Test_PHI,W_T_Next)
		Erms_Test = GetErms(TEST_OUT,TestTarget)
		L_Erms_Test.append(float(Erms_Test.split(',')[1]))


	print ('--Linear Regression Solution For '+ forwhat +' Dataset using '+ use +'--')
	print ("M = 4 \nLambda  = 2\neta=0.01")
	print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
	print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
	print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

# Training & Testing for Logistic Regression
def GetLogisticRegression(Dataset,Target,forwhat, use):

	TrainTarget = np.array(GenerateTrainingTarget(Target, TrainingPercent))
	TrainData = GenerateTrainingData(Dataset, TrainingPercent)

	ValTarget = np.array(GenerateValTarget(Target, ValidationPercent, len(TrainTarget)))
	ValData = GenerateValData(Dataset, ValidationPercent, len(TrainData))

	TestTarget = np.array(GenerateValTarget(Target,TestPercent, (len(TrainTarget)+len(ValTarget))))
	TestData = GenerateValData(Dataset, ValidationPercent, (len(TrainData)+len(ValData)))

	kmeans = KMeans(n_clusters=M, random_state=0).fit(TrainData) #Using Kmeans clustering we are clustering the Training Data consisting of approximately 58 thousand data to 10 clusters.
	Mu = kmeans.cluster_centers_ # Deriving the centroid of the clusters generated.

	Training_PHI = GetLogPhiMatrix(np.transpose(Dataset), Mu, TrainingPercent)
	Test_PHI     = GetLogPhiMatrix(np.transpose(TestData), Mu, 100)
	Val_PHI      = GetLogPhiMatrix(np.transpose(ValData), Mu, 100)

	print ('----------------------------------------------------')
	print ('--------------Please Wait for a minute!--------------')
	print ('----------------------------------------------------')

	W = np.zeros(len(Mu))
	W = np.transpose(W)
	W_T_Next        = np.add(0, W)
	La           = 2 # The Lambda is the regularization element used to avoid overfitting
	learningRate = 0.01 #The learning rate is used to define the rate at which the model can find the minimum value for the problem. If the rate is too high then the model will take too big steps and may diverge from the solution. On the other hand, if the rate is too low, then the model may not be able to converge to the minimum solution.
	L_Erms_Val   = []
	L_Erms_TR    = []
	L_Erms_Test  = []
	#W_Mat        = []

	for i in range(0,1263):
		#print ('---------Iteration: ' + str(i) + '--------------')
		# These values generated below are used to find the Weigths W using the Gradient Decent solution.
		Delta_E_D     = -np.dot((TrainTarget[i] - 1/(1+np.exp(np.dot(W_T_Next,Training_PHI[i])))),Training_PHI[i])# Here we calculate the Delta Change in E_D value
		La_Delta_E_W  = np.dot(La,W_T_Next) # # Here we calculate the Delta Change in E_W value. Here we also include the regularization factor(Lamda) to avoid overfitting of the model
		Delta_E       = np.add(Delta_E_D,La_Delta_E_W) # Here the change(Delta) in Error is calculated
		Delta_W       = -np.dot(learningRate,Delta_E) # Finally the change in weights is calculated based on the learning rate(eta)
		W_T_Next      = W_T_Next + Delta_W # The weights are updated with the delta value

		#-----------------TrainingData Accuracy---------------------#
		TR_TEST_OUT   = GetValTestLogistic(Training_PHI,W_T_Next)
		Erms_TR       = GetLossFunction(TR_TEST_OUT,TrainTarget)
		L_Erms_TR.append([float(Erms_TR.split(',')[0]),float(Erms_TR.split(',')[1])])

		#-----------------ValidationData Accuracy---------------------#
		VAL_TEST_OUT  = GetValTestLogistic(Val_PHI,W_T_Next)
		Erms_Val      = GetLossFunction(VAL_TEST_OUT,ValTarget)
		L_Erms_Val.append([float(Erms_Val.split(',')[0]),float(Erms_TR.split(',')[1])])

		#-----------------TestingData Accuracy---------------------#
		TEST_OUT      = GetValTestLogistic(Test_PHI,W_T_Next)
		Erms_Test = GetLossFunction(TEST_OUT,TestTarget)
		L_Erms_Test.append([float(Erms_Test.split(',')[0]),float(Erms_TR.split(',')[1])])

	L_Erms_TR = np.array(L_Erms_TR)
	L_Erms_Val = np.array(L_Erms_Val)
	L_Erms_Test = np.array(L_Erms_Test)
	print ('--Logistic Regression Solution For '+ forwhat +' Dataset using '+ use +'--')
	print ("M = 4 \nLambda  = 2\neta=0.01")
	value = [L_Erms_Test[x,:] for x in range(np.shape(L_Erms_Test)[0]) if L_Erms_Test[x,1]==min(L_Erms_Test[:,1])]
	print("Accuracy:"+ str(value[0][0]))

# Training & Testing for building a Neural Network
def GetNeuralNetwork(Dataset,Target,input_size,first_dense_layer_nodes,opt):
    # A model is a defination of a neural network that will decide how the network will actually work. The model is a critcal part of any machine learning program because it helps in deciding various factors involved in a machine learning code such as algorithm used, type of dataset used, learning rate of the algorithm and much more.
    # Dense is a type of Neural Network Layer that allows all the node of one layer to be connected to every node of the next layer using weights. On the other hand, an Activation function is a value approximation function used by neural networks to bring linearity to the network model. The reason why we use the dense() function first and then the activation function is that we have to first define the layer using dense() and then approximate its output for the next layer is the activation function.
    # The Sequential model is defined as a model that can be used to add layers to a neural network in a sequential manner. By using the add() method of the Sequentail model object, we can independently add as many layers as we want such that each new layer takes the output of the previous layer as a input. Due to this its ease of use and ability to independently add layers, we are using the Sequential Model.
    model = Sequential()

    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))

    # Dropout rate can be described as the rate of neurons that will be dropped from the network for every training process. If we run the model for too many epochs, there is a possibility of overfitting the model on the training data. In order to avoid overfitting, we are using this technique called Dropout Rate, which is a part of the Regularization process. Generally 20% -50% dropout rate will be enough for a model.
    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes, input_dim=first_dense_layer_nodes))
    model.add(Activation('relu'))

    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Softmax is a activation function that assigns probabilites to each classification class in the problem such that their sum much be 1. Softmax is used in models that are working on multi-class problems. As the output layer of this model have four classes i.e. Fizz, Buzz, FizzBuzz & Other, we have to use Softmax activation function in the output layer.

    model.summary()

    # Categorical_crossentropy loss function can be described as a loss function that unlike Binary crossenntropy function works on multi-class output models. This loss function orders the outputs into various categories, hence the name Categrical Cross Entropy. Moreover unlike Binary CrossEntropy which uses the sigmoid activation function, this loss function uses the softmax activation function. It is because of these reasons that we are using the Categorical CrossEntropy Loss Function.
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

    history = model.fit(Dataset
					 , Target
					 , validation_split=validation_data_split
					 , epochs=num_epochs
					 , batch_size=model_batch_size
					 , callbacks = [tensorboard_cb,earlystopping_cb]
                   )

    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10,15))

#Executing the code:
def main():

	print ('UBITname      = ameyakir')
	print ('Person Number = 50292574')
	print("--------Project 2: Handwritting Recognition---------")

	HOConcatSet,HOTargetC = GetConcatDatabase(ho_same_pairs,ho_diff_pairs,ho_features, 1, 790)
	HOConcatSet = np.asarray(HOConcatSet)

	HOSubtractSet,HOTargetS = GetSubtractDatabase(ho_same_pairs,ho_diff_pairs,ho_features, 1, 790)
	HOSubtractSet = np.asarray(HOSubtractSet)

	GSCConcatSet,GSCTargetC = GetConcatDatabase(gsc_same_pairs,gsc_diff_pairs,gsc_features, 0, 0)
	GSCConcatSet = np.asarray(GSCConcatSet)

	GSCSubtractSet,GSCTargetS = GetSubtractDatabase(gsc_same_pairs,gsc_diff_pairs,gsc_features, 0, 0)
	GSCSubtractSet = np.asarray(GSCSubtractSet)

	#Performing Linear Regression
	GetLinearRegression(HOConcatSet,HOTargetC,"Human-Observed","Concatenation")
	GetLinearRegression(HOSubtractSet,HOTargetS,"Human-Observed","Subtraction")
	GetLinearRegression(GSCConcatSet,GSCTargetC,"GSC","Concatenation")
	GetLinearRegression(GSCSubtractSet,GSCTargetS,"GSC","Subtraction")

	#Performing Logistic Regression
	GetLogisticRegression(HOConcatSet,HOTargetC,"Human-Observed","Concatenation")
	GetLogisticRegression(HOSubtractSet,HOTargetS,"Human-Observed","Subtraction")
	GetLogisticRegression(GSCConcatSet,GSCTargetC,"GSC","Concatenation")
	GetLogisticRegression(GSCSubtractSet,GSCTargetS,"GSC","Subtraction")

	# Here we are converting a single column target vector to two columns as a input to the Neural Network
	HOTargetC = np_utils.to_categorical(HOTargetC,2)
	HOTargetS = np_utils.to_categorical(HOTargetS,2)
	GSCTargetC = np_utils.to_categorical(GSCTargetC,2)
	GSCTargetS = np_utils.to_categorical(GSCTargetS,2)

	#Building Artificial Neural Network
	GetNeuralNetwork(HOConcatSet,HOTargetC,18,512,"rmsprop")
	GetNeuralNetwork(HOSubtractSet,HOTargetS[:1578],9,512,"rmsprop")
	GetNeuralNetwork(GSCConcatSet,GSCTargetC,1024,512,"sgd")
	GetNeuralNetwork(GSCSubtractSet,GSCTargetS,512,1000,"sgd")



if __name__ == '__main__':
    main()


