% Neural Network Binary-classification
% Code written by Moukafih Nabil, April. 2019.
% Command: Neural_Model
% Code uses an NLP (122 => 25 => 1) to perform event classification on the
% NSL-KDD dataset. The script changes mainly two parameters of the
% model (cost and training functions) to see their impact on the
% performance of the model. Next, a second classifier is used to improve the
% overall accuracy of the best parameters. 
% As a side note, I've left many stuff (plotting functions, training parameters, etc.)
% commented for those who are interested.
%
% Feel free to use modify this code.
clear all ; close all; clc

%% =========== Part 1: Loading Data =============

%% Load Training Data
fprintf('Loading Data ...\n');

load('dataset.mat'); % training data stored in arrays X, y
X_training=X_training';
Y_training=Y_training';
X_testing=X_testing';
Y_testing=Y_testing';

%% Create the neural network
% 1, 2: ONE input, TWO layers (one hidden layer and one output layer)
% [1; 1]: both 1st and 2nd layer have a bias node
% [1; 0]: the input is a source for the 1st layer
% [0 0; 1 0]: the 1st layer is a source for the 2nd layer
% [0 1]: the 2nd layer is a source for your output
net = network(1, 2, [1; 1], [1; 0], [0 0; 1 0], [0 1]);
net.inputs{1}.size = 122; % input size
net.layers{1}.size = 25; % hidden layer size
net.layers{2}.size = 1; % output layer size

net.performParam.normalization = 'standard';

%% Transfer function in layers
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';

net.layers{1}.initFcn = 'initnw';
net.layers{2}.initFcn = 'initnw';

net=init(net);

%% divide data into training and test
net.divideFcn= 'dividerand';
net.divideParam.trainRatio = 80/100; % 80% training
net.divideParam.valRatio = 20/100; % 20% validation set
net.divideParam.testRatio = 0/100; % 20% validation set

%% Performance functions

%net.performFcn = 'mae'; %Mean absolute error performance function.
net.performFcn = 'mse'; %Mean squared error performance function.
%net.performFcn = 'sae'; %Sum absolute error performance function.
%net.performFcn = 'sse'; %Sum squared error performance function.
%net.performFcn = 'crossentropy'; %Cross-entropy performance function.


%net.performParam.regularization = 0.5;

%% Training functions
%net.trainFcn = 'traingd'; %Gradient descent  backpropagation
%net.trainFcn = 'traingdm'; %Gradient descent with momentum 
net.trainFcn = 'trainrp'; % Resilence backpropagation

%net.trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation
%net.trainFcn = 'traincgp'; % Conjugate gradient backpropagation with PR updates
%net.trainFcn = 'traincgf'; % Conjugate gradient backpropagation with FR updates

%net.trainFcn = 'trainbfg'; % BFGS
%net.trainFcn = 'trainlm'; % Levenberg-Marquardt backpropagation

%% Train the neural network
[net,tr] = train(net,X_training,Y_training); % return the network and training record

%% Test the Neural Network on the training set and Train SVM

fprintf('\nTesting the neural network on training set\n');
%pause
outputs = net(X_training);
errors = gsubtract(Y_training,outputs);
performance = perform(net,Y_training,outputs);
%fprintf('\nTraining Confusing Matrix\n');
%figure, plotconfusion(Y_training,outputs)
%figure, ploterrhist(errors)
figure, plotperform(tr)
%figure, plottrainstate(tr)

%% Ploting graphs

figure, plot(tr.time,tr.perf)
hold on

net=init(net);
net.trainFcn = 'traingdm'; %Gradient descent with momentum 
[net,tr1] = train(net,X_training,Y_training); % return the network and training record
plot(tr1.time,tr1.perf)
hold on


net=init(net);
net.trainFcn = 'trainrp'; % Resilence backpropagation 
[net,tr2] = train(net,X_training,Y_training); % return the network and training record
plot(tr2.time,tr2.perf)
hold on

net=init(net);
net.trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation
[net,tr3] = train(net,X_training,Y_training); % return the network and training record
plot(tr3.time,tr3.perf)
hold on

net=init(net);
net.trainFcn = 'traincgp'; % Conjugate gradient backpropagation with PR updates 
[net,tr4] = train(net,X_training,Y_training); % return the network and training record
plot(tr4.time,tr4.perf)
hold on

net=init(net);
net.trainFcn = 'traincgf'; % Conjugate gradient backpropagation with FR updates
[net,tr5] = train(net,X_training,Y_training); % return the network and training record
plot(tr5.time,tr5.perf)
hold on

net=init(net);
net.trainFcn = 'trainbfg'; % BFGS
[net,tr6] = train(net,X_training,Y_training); % return the network and training record
plot(tr6.time,tr6.perf)
hold on

net=init(net);
net.trainFcn = 'trainlm'; % Levenberg-Marquardt backpropagation
[net,tr7] = train(net,X_training,Y_training); % return the network and training record
plot(tr7.time,tr7.perf)
hold on


% net.performFcn = 'mse'; %Mean squared error performance function.
% [net,tr1] = train(net,X_training,Y_training); % return the network and training record
% plot(tr1.time,tr1.perf)
% hold on
% 
% net=init(net);
% net.performFcn = 'sae'; %Sum absolute error performance function.
% [net,tr2] = train(net,X_training,Y_training); % return the network and training record
% plot(tr2.time,tr2.perf)
% hold on
% 
% net=init(net);
% net.performFcn = 'sse'; %Sum squared error performance function.
% [net,tr3] = train(net,X_training,Y_training); % return the network and training record
% plot(tr3.time,tr3.perf)
% hold on
% 
% net=init(net);
% net.performFcn = 'crossentropy'; %Cross-entropy performance function.
% [net,tr4] = train(net,X_training,Y_training); % return the network and training record
% plot(tr4.time,tr4.perf)
% 
% legend('mae','mse','sae','sse','crossentropy');
% xlabel('time(s)');
% ylabel('Error');
%% -------------------------- test --------------------------
fprintf('\nTesting the neural network..\n');
%pause
outputs1 = net(X_testing);
error1 = gsubtract(Y_testing,outputs1);
performance1 = perform(net,Y_testing,outputs1);
fprintf('\nTesting Confusing Matrix\n');
figure, plotconfusion(Y_testing,outputs1)
%figure, ploterrhist(error1)






% ------------------------- training SVM ---------------------------
fprintf('Using SVM..');
pause
%c = cvpartition(size(X_training,2),'KFold',10);
%opts = struct('Optimizer','bayesopt','ShowPlots',false,'CVPartition',c,...
    % 'AcquisitionFunctionName','expected-improvement-plus');
    
Y_training_SVM = Y_training;
Y_training_SVM(Y_training_SVM==0)=-1;

SVMModel = fitcsvm(outputs',Y_training_SVM,'KernelFunction','polynomial');
%SVMModel = fitcsvm(outputs',Y_training,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
      %'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
      %'expected-improvement-plus','ShowPlots',false));

%[output_svm,score] = predict(SVMModel,outputs');
%fprintf('\nTraining Confusing Matrix with SVM\n');
%pause
%figure, plotconfusion(Y_training,output_svm')

[output_svm1,score1] = predict(SVMModel,outputs1');
fprintf('\nTesting Confusing Matrix with SVM\n');
pause
figure, plotconfusion(Y_testing,output_svm1')


%Train the SVM Classifier
SVMModel = fitcsvm(outputs',Y_training,'KernelFunction','rbf');
[label,score] = predict(SVMModel,outputs');


figure
plotconfusion(Y_testing,outputs)

figure
plotconfusion(Y_testing,label')

errors = gsubtract(Y_training,outputs);
performance = perform(net,Y_training,outputs);

%% Plots  (%training)
figure, plotperform(tr)
figure, plottrainstate(tr)

%% Test the Neural Network on the testing test
outputs1 = net(X_testing);
errors1 = gsubtract(Y_testing,outputs1);
performance1 = perform(net,Y_testing,outputs1);

figure, plotconfusion(Y_testing,outputs1)
figure, ploterrhist(errors1)


