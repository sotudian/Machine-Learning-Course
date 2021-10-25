function [ Best_BC_RS_KS,Tuning_Results ] = Tuning_RBF( X,Y )
%% Tuning with 10-fold cross-validation -----RBF-----


% BoxConstraint
box=[0.001,0.01,0.1,1,10,100];
% rbf_sigma 
rbf_sigma =[0.01,0.1,2,5];
% KernelScale
KernelScale=[0.1,1,5,10];

% box=[0.01,5];
% rbf_sigma =[0.001,0.01,1.5,5];
% KernelScale=[0.01,0.1];

Best_BC_RS_KS=[0,0,0,0,100];
f=1;
for j=1:size(box,2)
    for p=1:size(rbf_sigma ,2)
        for k=1:size(KernelScale,2)
        svm = fitcsvm(X,Y,'KernelFunction','rbf',...
               'BoxConstraint',box(j),'Standardize',true,'KernelScale',KernelScale(k));
%            10-fold cross validation.
       cv = crossval(svm);
%        Estimate the out-of-sample misclassification rate.
   Tuning_Results(f,1)=f;
   Tuning_Results(f,2)=box(j);
   Tuning_Results(f,3)=rbf_sigma (p);
   Tuning_Results(f,4)=KernelScale(k);
   Tuning_Results(f,5)=kfoldLoss(cv); 
   
   if Best_BC_RS_KS(1,5)>Tuning_Results(f,5)
       Best_BC_RS_KS=Tuning_Results(f,:);
   end
   
      f=f+1;
        end
    end
end


end

