%% ML-Alogrithm Selection
% William Baumchen
close all; clear; clc

%% Data Preprocessing

% Import Data
datain = readmatrix("winequality-white.csv");
% Shuffle Data Entries for Splitting Data
% Set random seed for reproducibility
rng(12)
datain = datain(randperm(size(datain,1)),:);
% Set Fraction of Entries for Test Set
a = 0.2;
% Split Data
xTest = datain(1:round(a*size(datain,1)),1:11);
yTest = datain(1:round(a*size(datain,1)),12);
xTrain = datain(round(a*size(datain,1))+1:end,1:11);
yTrain = datain(round(a*size(datain,1))+1:end,12);

%% Linear Regression
% Create linear regression model(s), find the MSE for each trained model
linmdl = fitrlinear(xTrain,yTrain,'Verbose',0,'CrossVal','on');
mse(:,1) = kfoldLoss(linmdl,'Mode','individual');

%% Decision Tree
% Create  decision tree model(s), find the MSE for each trained model
treemdl = fitrtree(xTrain,yTrain,'CrossVal','on');
mse(:,2) = kfoldLoss(treemdl,'Mode','individual');

%% SVM Regression
% Create support vector machine model(s), find the MSE for each trained model
svmmdl = fitrsvm(xTrain,yTrain,'Verbose',0,'CrossVal','on');
mse(:,3) = kfoldLoss(svmmdl,'Mode','individual');

%% Kernal Regression
% Create kernal regression model(s), find the MSE for each trained model
kernmdl = fitrkernel(xTrain,yTrain,'Verbose',0,'CrossVal','on');
mse(:,4) = kfoldLoss(kernmdl,'Mode','individual');

%% Ensemble Regression
% Create ensemble learner model(s), find the MSE for each trained model
ensmdl = fitrensemble(xTrain,yTrain,'CrossVal','on');
mse(:,5) = kfoldLoss(ensmdl,'Mode','individual');

%% Plotting
% Plot boxplot of cross-validation MSE from fitted models
figure(1)
boxplot(mse,'Labels',{'Linear Regression','Decision Tree','Support Vector Machine','Kernal Regression','Ensemble Learner'})
title('Model Cross-Validation MSE')
ylabel('')

% Find the mse with respect to the test set
% Initialize matrix
mss = zeros(10,5);
for i = 1:10
    mss(i,1) = loss(linmdl.Trained{i},xTest,yTest);
    mss(i,2) = loss(treemdl.Trained{i},xTest,yTest);
    mss(i,3) = loss(svmmdl.Trained{i},xTest,yTest);
    mss(i,4) = loss(kernmdl.Trained{i},xTest,yTest);
    mss(i,5) = loss(ensmdl.Trained{i},xTest,yTest);
end
% Create boxplot of MSE w/respect to test set
figure(2)
boxplot(mss,'Labels',{'Linear Regression','Decision Tree','Support Vector Machine','Kernal Regression','Ensemble Learner'})
title('Test Data Mean-Squared Error')