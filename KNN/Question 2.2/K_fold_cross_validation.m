
% Shahabeddin Sotudian-94125091
% 5_fold_cross_validation

function [ErrorTotal_k, Optimal_k ] = K_fold_cross_validation( Training_Validation_Data )

Folds=[];
% 5_fold_cross_validation
Folds(:,:,1)=Training_Validation_Data(1:44,:);
Folds(:,:,2)=Training_Validation_Data(45:88,:);
Folds(:,:,3)=Training_Validation_Data(89:132,:);
Folds(:,:,4)=Training_Validation_Data(133:176,:);
Folds(:,:,5)=Training_Validation_Data(177:220,:);
% Type of distance
Type_of_Dist='cosine';
Output_column=279;

 for i=1:50
     
     
     for F=1:5
         
         Validation_Data=Folds(:,:,F);
            if F==1
            Training_Data=cat(1,Folds(:,:,2),Folds(:,:,3),Folds(:,:,4),Folds(:,:,5));
            elseif F==2
            Training_Data=cat(1,Folds(:,:,1),Folds(:,:,3),Folds(:,:,4),Folds(:,:,5));
            elseif F==3
            Training_Data=cat(1,Folds(:,:,1),Folds(:,:,2),Folds(:,:,4),Folds(:,:,5));
            elseif F==4
            Training_Data=cat(1,Folds(:,:,1),Folds(:,:,2),Folds(:,:,3),Folds(:,:,5));
            elseif F==5
            Training_Data=cat(1,Folds(:,:,1),Folds(:,:,2),Folds(:,:,3),Folds(:,:,4));
         end
     
 Output_KNN= KNN(Validation_Data,Training_Data,i,Type_of_Dist,Output_column);
 
 %  Classification error %
classes=Validation_Data(:,Output_column);
assignedClasses=Output_KNN;
confus=confusionmat(classes, assignedClasses);
Sumcolumn=sum(confus,1);
SumRow=sum(confus,2);
for p=1:size(confus,1)
ClassificationError(p) = 100*(1-(  sum(diag(confus))   /   (sum(diag(confus))+Sumcolumn(p)+SumRow(p)-2*confus(p,p))   ));
end
ErrorFolds(F)=mean(ClassificationError);

     end
  ErrorTotal_k(1,i)=i;   
  ErrorTotal_k(2,i)=mean(ErrorFolds);   
     
     
 end
 
 
 
 
 %% ploting
 plot(ErrorTotal_k(1,:),ErrorTotal_k(2,:))
 xlabel('K')
 ylabel('Classification Error %')
 axis([1 50 15 30])
 %% Best K
 S=find(ErrorTotal_k(2,:)==min(ErrorTotal_k(2,:)));
 Optimal_k=ErrorTotal_k(1,S(1));
 
end
 