% Shahabeddin Sotudian-94125091
% Question 2.1
clear;
clc;
%% DATA

load('IrisDataset.mat');
% Training Data
Training_Data=cat(1,IrisDataset(1:25,:),IrisDataset(51:75,:),IrisDataset(101:126,:));
% Test Data
Test_Data=cat(1,IrisDataset(39:50,:),IrisDataset(88:100,:),IrisDataset(139:150,:));
% Validation Data
Validation_Data=cat(1,IrisDataset(26:38,:),IrisDataset(76:87,:),IrisDataset(127:138,:));


%% K-Nearest Neighbor Algorithm

% Parameter K
k=[1,5,11,51];

% Type of distance
Type_of_Dist='cosine';
Output_column=5;

% Validation
for i=1:4
Output_KNN(:,i) = KNN(Validation_Data,Training_Data,k(i),Type_of_Dist,Output_column);
Validation(Validation_Data,Output_KNN(:,i),k(i),Output_column)
end


%% Model test
Output_KNN_test = KNN(Test_Data,Training_Data,11,Type_of_Dist,Output_column);

%  Classification Accuracy Test data
classes=Test_Data(:,Output_column);
assignedClasses=Output_KNN_test;
confus=confusionmat(classes, assignedClasses);
fprintf('\n  ######################################################')
fprintf('\n                         Model Test ')
fprintf('\n  ######################################################')
fprintf('\n  Confusion Matrix: \n')
  disp(confus)
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
fprintf('\nNumber of misclassified samples = %d%\n', misclassified)







