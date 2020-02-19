%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code Written in Matlab R2018b.
% The scripts uses the NSL-KDD dataset (csv format), performs some
% pre-processing steps and stored the input and class labels separately
% The script also uses a script written by Christopher L. Stokely to
% perform One Hot Encoding on nominal features.
%
%
clear all; close all; clc;

%% Loading Data
training_file = fullfile('KDDTrain+.csv');
X_Train = readtable(training_file);

testing_file = fullfile('KDDTest+.csv');
X_Test = readtable(testing_file);

X_Test(:,'id') = []; % remove the 'id' column
X_Train(:,'id') = []; % remove the 'id' column

Y_Train = X_Train(:,'x_class_'); %extract labels
Y_Test = X_Test(:,'x_class_'); %extract labels

X_Test(:,'x_class_') = []; %delete labels
X_Train(:,'x_class_') = []; %delete labels


%% Categorical Encoding
% (I've seen scripts who applied integer-encoding instead of categorical
% encoding, but I don't recommend that.)

%Join both data
data = [X_Train; X_Test];

data=createOneHotEncoding(data,'x_protocol_type_');
data=createOneHotEncoding(data,'x_service_');
data=createOneHotEncoding(data,'x_flag_');

% It's possible to use the script 'createOneHotEncoding' to
% transform the class labels in the same way (this will give a binary vector), 
% but since it's a binary classification problem, we replaced the 'normal' 
% events with '0' and 'attack' with '1'.

% Train label
%Y_Train=createOneHotEncoding(Y_Train,'x_class_');
% Test label
%Y_Test=createOneHotEncoding(Y_Test,'x_class_');

% 0 normal, 1 anomaly
Y_Train.x_class_=replace(Y_Train.x_class_,'normal','0');
Y_Train.x_class_=replace(Y_Train.x_class_,'anomaly','1');

Y_Test.x_class_=replace(Y_Test.x_class_,'normal','0');
Y_Test.x_class_=replace(Y_Test.x_class_,'anomaly','1');

Y_Test.x_class_=str2double(Y_Test.x_class_);
Y_Train.x_class_=str2double(Y_Train.x_class_);

% Resplitting
fprintf('Splitting Data..\n');
X_training = data(1:size(X_Train,1),1:end);
X_testing = data(size(X_Train,1)+1:end,1:end);

clear data testing_file training_file X_Train X_Test;


%% Convert to Matrices and normalize data
X_training=X_training{:,:};
Y_Train = Y_Train{:,:};

X_testing=X_testing{:,:};
Y_Test = Y_Test{:,:};

% You can perform feature normalization if you want, but if you're using 
% neural networks, MAtlab can do that for you when create your model.
%[X_Training, ~, ~] = featureNormalize(X_training);
%[X_Testing, ~, ~] = featureNormalize(X_testing);

X_Training=X_training';
Y_Training = Y_Train';
X_Testing=X_testing';
Y_Testing = Y_Test';

clear X_training X_testing Y_Test Y_Train;

fprintf('Saving in mat file..\n');
save('data.mat','X_Training', 'Y_Training', 'X_Testing', 'Y_Testing');

