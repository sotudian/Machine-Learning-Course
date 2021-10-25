% Shahabeddin Sotudian-94125091
% Question 2.2

%% DATA
clear;
clc;
load('ArrhythmiaDataset_without_missing.mat');

%% Under_sampling for class 1 - RANDOMLY- Diferent samples remove in diffrent run
Balanced_data = Under_sampling( ArrhythmiaDataset_without_missing );

%% Removing Classes: 7,8,14 and 15
R1=find(Balanced_data(:,279)==7);
Balanced_data(R1,:)=[];
R2=find(Balanced_data(:,279)==8);
Balanced_data(R2,:)=[];
R3=find(Balanced_data(:,279)==14);
Balanced_data(R3,:)=[];
R4=find(Balanced_data(:,279)==15);
Balanced_data(R4,:)=[];

%% DATA
% Training Data and Validation Data
Training_Validation_Data=Balanced_data(1:220,:);
% Test Data
Test_Data=Balanced_data(221:273,:);
Output_column=279;

%% Validation
% Parameter K  ==> ***5-fold cross validation***

[ErrorTotal_k, Optimal_k ] = K_fold_cross_validation( Training_Validation_Data);
fprintf('\n  ######################################################')
fprintf('\n                   Optimal k=%d% \n', Optimal_k)
fprintf('\n  ######################################################')
fprintf('\n                   Error=%.2f%\n',ErrorTotal_k(2,Optimal_k))
fprintf('\n  ######################################################\n\n\n')
k=Optimal_k;

% Type of distance
Type_of_Dist='cosine';

% Other-Validation
Output_KNN = KNN(Training_Validation_Data(1:88,:),Training_Validation_Data(89:220,:),k,Type_of_Dist,Output_column);
Validation(Training_Validation_Data(1:88,:),Output_KNN,k,Output_column)


%% Model Test
Output_KNN_test = KNN(Test_Data,Training_Validation_Data,k,Type_of_Dist,Output_column);
fprintf('\n  ######################################################')
fprintf('\n                        Model Test ')
fprintf('\n  ######################################################\n')
%  Classification Accuracy Test data
classes=Test_Data(:,Output_column);
assignedClasses=Output_KNN_test;
confus=confusionmat(classes, assignedClasses);
fprintf('\n  Confusion Matrix: \n')
  disp(confus)
fprintf('\n ---------------------------------------------------\n ')
Sumcolumn=sum(confus,1);
SumRow=sum(confus,2);
fprintf('\n  Classification Accuracy for Test data: \n')
for p=1:size(confus,1)
ClassificationAccuracy(p) = 100*(  sum(diag(confus))   /   (sum(diag(confus))+Sumcolumn(p)+SumRow(p)-2*confus(p,p))   );
fprintf(['\n KNN Classifier Accuracy for class ',num2str(p),' = %.3f%\n'], ClassificationAccuracy(p))
end
Accuracy=mean(ClassificationAccuracy);
fprintf(['\n Classifier Accuracy for Test data = %.3f%\n'], Accuracy)
misclassified = sum(confus(:)) - sum(diag(confus));
fprintf('\n ---------------------------------------------------\n ')
fprintf('\nNumber of misclassified samples = %d%\n', misclassified)
fprintf('\n ---------------------------------------------------\n ')

