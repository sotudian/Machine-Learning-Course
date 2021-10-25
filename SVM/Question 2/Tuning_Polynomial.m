function [ Best_BC_PO_KS,Tuning_Results ] = Tuning_Polynomial( X,Y )
%% Tuning with 10-fold cross-validation -----Polynomial-----

% BoxConstraint
box=[0.001,0.01,0.1,1,10,100];
% PolynomialOrder
PolynomialOrder=[1,3,5];
% KernelScale
KernelScale=[0.1,1,4,10];


% box=[0.01,0.1];
% PolynomialOrder=[1,3,4];
% KernelScale=[0.01,0.1];

Best_BC_PO_KS=[0,0,0,0,100];
f=1;
for j=1:size(box,2)
    for p=1:size(PolynomialOrder,2)
        for k=1:size(KernelScale,2)
        
    svm = fitcsvm(X,Y,'KernelFunction','Polynomial',...
               'PolynomialOrder',PolynomialOrder(p),'BoxConstraint',box(j),'Standardize',true,'KernelScale',KernelScale(k));
%            10-fold cross validation.
       cv = crossval(svm);
%        Estimate the out-of-sample misclassification rate.
   Tuning_Results(f,1)=f;
   Tuning_Results(f,2)=box(j);
   Tuning_Results(f,3)=PolynomialOrder(p);
   Tuning_Results(f,4)=KernelScale(k);
   Tuning_Results(f,5)=kfoldLoss(cv); 
   
   if Best_BC_PO_KS(1,5)>Tuning_Results(f,5)
       Best_BC_PO_KS=Tuning_Results(f,:);
   end
   
      f=f+1;
        end
    end
end


end

