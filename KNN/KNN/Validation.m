% Shahabeddin Sotudian-94125091
function [  ] = Validation(Test_Data,Output_KNN,k,Output_column)
fprintf('\n  ######################################################')
fprintf(['\n           Validation for KNN with k= ',num2str(k)])
fprintf('\n  ######################################################')

%% Confusion Matrix
classes=Test_Data(:,Output_column);
assignedClasses=Output_KNN;
confus=confusionmat(classes, assignedClasses);
fprintf('\n  Confusion Matrix: \n')
  disp(confus)
fprintf('\n ---------------------------------------------------\n ') 
 
%%  Classification Accuracy
Sumcolumn=sum(confus,1);
SumRow=sum(confus,2);
for p=1:size(confus,1)
ClassificationAccuracy(p) = 100*(  sum(diag(confus))   /   (sum(diag(confus))+Sumcolumn(p)+SumRow(p)-2*confus(p,p))   );
fprintf(['\n KNN Classifier Accuracy for class ',num2str(p),' = %.3f%\n'], ClassificationAccuracy(p))
end
Accuracy=mean(ClassificationAccuracy);
fprintf(['\n Classifier Accuracy = %.3f%\n'], Accuracy)
fprintf('\n ---------------------------------------------------\n ') 

%% Number of misclassified samples
misclassified = sum(confus(:)) - sum(diag(confus));
fprintf('\nNumber of misclassified samples = %d%\n', misclassified)
fprintf('\n ---------------------------------------------------\n ') 

%% Precision
for p=1:size(confus,1)
 Precision(p)=confus(p,p)/Sumcolumn(p);
 fprintf(['\nPrecision for class ',num2str(p),' = %.3f%\n'], Precision(p))
end
fprintf('\n ---------------------------------------------------\n ') 

%% Recall-Sensitivity
for p=1:size(confus,1)
 Recall(p)=confus(p,p)/SumRow(p);
 fprintf(['\nRecall and Sensitivity for class ',num2str(p),' = %.3f%\n'], Recall(p))
end
fprintf('\n ---------------------------------------------------\n ') 

%% F-measure
for p=1:size(confus,1)
F_measure(p)=(2*Precision(p)*Recall(p))/(Precision(p)+Recall(p));
fprintf(['\nF-measure for class ',num2str(p),' = %.3f%\n'], F_measure(p))
end
fprintf('\n ---------------------------------------------------\n ') 

%% Specificity 
for p=1:size(confus,1)
Specificity(p)=(sum(diag(confus))-confus(p,p)) /  (sum(diag(confus))+Sumcolumn(p)-2*confus(p,p));
fprintf(['\nSpecificity for class ',num2str(p),' = %.3f%\n'], Specificity(p))
end

fprintf('\n \n ') 
end

