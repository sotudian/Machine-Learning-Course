 % Shahab Sotudian 94125091
 % Removeing Outliers

function [ ] = Remove_Outliers( IN_data,OUT_data ,dataset_Num)
n=size(IN_data,1);
fprintf('\n\n\n\n =======================================================================')
fprintf('\n                        Results for Data %d',dataset_Num)
fprintf('\n =======================================================================')


%  Linear Regression
coefficient=Linear_Regression(IN_data,OUT_data);



% MSE Total
estimated_y=zeros(n,1);
for r=1:n    
for t=1:size(coefficient,1) 
    
if t==size(coefficient,1) 
    estimated_y(r,1)=estimated_y(r,1)+ coefficient(t); 
else
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient(t)*IN_data(r,:));
    
end
end
end
MSE1=sum((OUT_data-estimated_y).^2)/n;

fprintf('\n\n 1- Total MSE for data with outlier =  %f',MSE1);



% Removing each data point and computing MSE
for k=1:n 
  B=IN_data;
  y1=OUT_data;
      B(k,:)=[];
      y1(k,:)=[];
coefficient_temp=Linear_Regression(B,y1);
estimated_y_temp=zeros(n-1,1);
for r=1:n-1
    
for t=1:size(coefficient_temp,1)    
if t==size(coefficient_temp,1) 
    estimated_y_temp(r,1)=estimated_y_temp(r,1)+ coefficient_temp(t); 
else
    estimated_y_temp(r,1)=estimated_y_temp(r,1)+sum(coefficient_temp(t)*B(r,:));
    
end
end
end  
MSE_removing(k,:)=sum((y1-estimated_y_temp).^2)/(n-1);
end
MSE_change=abs((MSE1*ones(size(MSE_removing,1),1))-MSE_removing);
MSE_change1=MSE_change;




% Number of outliers
Number_Outliers= Num_Outliers(IN_data);
fprintf('\n\n 2- Number of Outliers =  %d',Number_Outliers);



% Removing outliers
if Number_Outliers>0
for ee=1:Number_Outliers
[row(ee) column(ee)]=find(MSE_change==max(MSE_change));

IN_data(row(ee),:)=[];
OUT_data(row(ee),:)=[];
MSE_change(row(ee),:)=-100;
end
end



% Finding regression model and error for noiseless dataset
n=size(IN_data,1);
coefficient=Linear_Regression(IN_data,OUT_data);
% MSE
estimated_y=zeros(n,1);
for r=1:n    
for t=1:size(coefficient,1)    
if t==size(coefficient,1) 
    estimated_y(r,1)=estimated_y(r,1)+ coefficient(t); 
else
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient(t)*IN_data(r,:));  
end
end
end
MSE2=sum((OUT_data-estimated_y).^2)/n;



% Plotting
figure;
subplot(1,2,1)
plot(MSE_change1,'g')
axis([0 500 -1 MSE1/3])
title('MSE change for removing each data point(with outliers)')
subplot(1,2,2)
if Number_Outliers>0
MSE_change(row,:)=[];
end
plot(MSE_change,'g')
axis([0 500 -1 MSE1/3])
title('MSE change for removing each data point(without outliers)')
fprintf('')
fprintf('\n\n 3- Final Regression Model(Without Outliers) \n')

rowname = {'Value of coefficients'};
x1 = coefficient(1);
x2 =coefficient(2);
x3 = coefficient(3);
x4 =coefficient(4);
b=coefficient(5);
T1 = table(x1,x2,x3,x4,b,...
'RowNames',rowname)
    
fprintf('\n\n 4- Total MSE for data without outlier =  %f \n',MSE2);




end

