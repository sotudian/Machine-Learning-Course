clear
clc
load('ArrhythmiaDataset_with_missing.mat');
% finding the missing values position(Attribute 14 is removed)
e=1;
for r=11:14
    j=1;
   for i=1:452
  s=ArrhythmiaDataset(i,r);  
    if isnan(s)
    missing_position(j,r-10)=i;
    missing_Rows(e)=i;
    j=j+1;
    e=e+1;
    end
   end
end

% Missing-data imputation for Attribute 11-15(Attribute 14 is removed)
% Using k-Nearest_Position neighborhood Alg
Training_Data_final=ArrhythmiaDataset;
Training_Data_final(missing_Rows,:)=[];
k=5;

for m1=1:4
Test_Data=[]; 
Training_Data=[]; 
Distance=[];
Nearest_Position_Missing=[];
K_Nearest_Missing=[];
KNN_matrix =[];      
        
if m1==1
    p1=find(missing_position(:,m1)>0);
    Test_Data=ArrhythmiaDataset(missing_position(p1,m1),:);
    Test_Data(:,11)=[];
    Training_Data=Training_Data_final;
    Training_Data(:,11)=[];
 elseif m1==2
    p2=find(missing_position(:,m1)>0);
    Test_Data=ArrhythmiaDataset(missing_position(p2,m1),:);
    Test_Data(:,12)=[];
    Training_Data=Training_Data_final;
    Training_Data(:,12)=[];
 elseif m1==3
    p3=find(missing_position(:,m1)>0);
    Test_Data=ArrhythmiaDataset(missing_position(p3,m1),:);
    Test_Data(:,13)=[];
    Training_Data=Training_Data_final;
    Training_Data(:,13)=[];
 elseif m1==4
    p4=find(missing_position(:,m1)>0);
    Test_Data=ArrhythmiaDataset(missing_position(p4,m1),:);
    Test_Data(:,14)=[];
    Training_Data=Training_Data_final;
    Training_Data(:,14)=[];
end


Distance =pdist2(Test_Data, Training_Data, 'cosine');
   
   % K Nearest_Position neighbors 
    [~,Nearest_Position_Missing] = sort(Distance,2);
    K_Nearest_Missing=Nearest_Position_Missing(:,1:k);
   % Mode 
   KNN_matrix = reshape(Training_Data(K_Nearest_Missing,m1+10),[],k);
    Imputation_Missing=mode(KNN_matrix,2);
   % if mode is 1
    Imputation_Missing(Imputation_Missing==1) = KNN_matrix(Imputation_Missing==1,1);
        
Imputation_Missing_Matrix(1:length(Imputation_Missing),m1)=Imputation_Missing;
end
 % Imputation matrix plotting
Imputation_Missing_Cell=num2cell( Imputation_Missing_Matrix);
Imputation_Missing_Cell(9:22,1)={'--'};
Imputation_Missing_Cell(2:22,3)={'--'};
Imputation_Missing_Cell(2:22,4)={'--'};
Imputation_Missing_Cell



