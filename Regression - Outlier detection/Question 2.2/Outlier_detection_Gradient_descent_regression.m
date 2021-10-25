function [ ] = Outlier_detection_Gradient_descent_regression( X5,Y5 ,dataset_Num)

fprintf('\n\n\n\n =======================================================================')
fprintf('\n                        Results for Data %d',dataset_Num)
fprintf('\n =======================================================================')

%%  Linear Regression

IN_data=X5;
OUT_data=Y5;
n=size(IN_data,1);
coefficient=Gradient_Descent_linear(IN_data,OUT_data);

n=size(IN_data,1);

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
fprintf('\n\n --OOOO     Linear regression (y=ax)     OOOO--\n')
fprintf('\n 1- Total MSE for data with outliers =  %f',MSE1);


% Removing each data point and computing MSE
for k=1:n 
  B=IN_data;
  y1=OUT_data;
      B(k,:)=[];
      y1(k,:)=[];
coefficient_temp=Gradient_Descent_linear(B,y1);
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

s=std(MSE_change1);
m=mean(MSE_change1);
theresh11=m+3*s;
theresh12=m-3*s;
Number_Outliers=0;
jj=1;
Noise_pos1=[];
for pp=1:n
    if MSE_change1(pp,:)>(theresh11) || MSE_change1(pp,:)<(theresh12)
     Number_Outliers=Number_Outliers+1;  
     Noise_pos1(jj)=pp;
     jj=jj+1;
    end
end

if isempty(Noise_pos1)
    fprintf('\n 3- There is not any outlier \n');
else
fprintf('\n 3- Outliers rows in dataset \n');
disp(Noise_pos1)
end


% MSE Without noise
if Number_Outliers>0
IN_data(Noise_pos1,:)=[];
OUT_data(Noise_pos1,:)=[];
A_linear=IN_data;
B_linear=OUT_data;
coefficient_optimal=Gradient_Descent_linear(IN_data,OUT_data);
n=size(IN_data,1);

% MSE Total
estimated_y=zeros(n,1);
for r=1:n    
for t=1:size(coefficient_optimal,1)    
if t==size(coefficient_optimal,1) 
    estimated_y(r,1)=estimated_y(r,1)+ coefficient_optimal(t); 
else
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient_optimal(t)*IN_data(r,:));
end
end
end
MSE3=sum((OUT_data-estimated_y).^2)/n;
fprintf(' 4- Total MSE for data without outliers =  %f',MSE3);
else
  fprintf(' 4- Total MSE for data without outliers =  %f',MSE1);  
end





fprintf('\n =======================================================================\n')


%%  Non Linear Regression

IN_data=X5;
OUT_data=Y5;
coefficient=Gradient_Descent_non_linear(IN_data,OUT_data);
n=size(IN_data,1);

% MSE Total
estimated_y=zeros(n,1);
for r=1:n    
for t=1:size(coefficient,1)
   
if t==9
    estimated_y(r,1)=estimated_y(r,1)+ coefficient(t);
elseif t<9 && t>4
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient(t)*IN_data(r,:));
    
elseif t<5 && t>0
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient(t)*((IN_data(r,:).^2))); 
end

end
end
MSE_non1=sum((OUT_data-estimated_y).^2)/n;
fprintf('\n --OOOO     Non Linear regression (y=ax^2+bx)     OOOO--\n')

fprintf('\n 1- Total MSE for data with outliers =  %f',MSE_non1);

% Removing each data point and computing MSE
for k=1:n 
  B=IN_data;
  y1=OUT_data;
      B(k,:)=[];
      y1(k,:)=[];
coefficient_temp=Gradient_Descent_non_linear(B,y1);
estimated_y_temp=zeros(n-1,1);
for r=1:n-1
for t=1:size(coefficient_temp,1)   
 
    if t==3
    estimated_y_temp(r,1)=estimated_y_temp(r,1)+ coefficient_temp(t);
elseif t==2
    estimated_y_temp(r,1)=estimated_y_temp(r,1)+sum(coefficient_temp(t)*B(r,:));
    else
    estimated_y_temp(r,1)=estimated_y_temp(r,1)+sum(coefficient_temp(t)*((B(r,:).^2)));
    
    end



end
end  
MSE_removing(k,:)=sum((y1-estimated_y_temp).^2)/(n-1);
end
MSE_change_non=abs((MSE_non1*ones(size(MSE_removing,1),1))-MSE_removing);
MSE_change2=MSE_change_non;

s=std(MSE_change2);
m=mean(MSE_change2);
theresh21=m+3*s;
theresh22=m-3*s;
Number_Outliers2=0;
jj=1;
Noise_pos2=[];
for pp=1:n
    if MSE_change2(pp,:)>(theresh21)|| MSE_change2(pp,:)<(theresh22)
     Number_Outliers2=Number_Outliers2+1;   
     Noise_pos2(jj)=pp;
     jj=jj+1;
    end
end

fprintf('\n 2- Number of Outliers =  %d',Number_Outliers2);
if isempty(Noise_pos2)
    fprintf('\n 3- There is not any outlier \n');
else
fprintf('\n 3- Outliers rows in dataset \n');
disp(Noise_pos2)
end
if Number_Outliers2>0
    
% MSE Without noise
IN_data(Noise_pos2,:)=[];
OUT_data(Noise_pos2,:)=[];

coefficient_non2=Gradient_Descent_non_linear(IN_data,OUT_data);
n=size(IN_data,1);

% MSE Total
estimated_y=zeros(n,1);
for r=1:n    
for t=1:size(coefficient_non2,1)
    
if t==9
    estimated_y(r,1)=estimated_y(r,1)+ coefficient_non2(t);
elseif t<9 && t>4
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient_non2(t)*IN_data(r,:));
    
elseif t<5 && t>0
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient_non2(t)*((IN_data(r,:).^2))); 
end

end
end
MSE_non2=sum((OUT_data-estimated_y).^2)/n;
fprintf(' 4- Total MSE for data without outliers =  %f',MSE_non2);
else
  fprintf(' 4- Total MSE for data without outliers =  %f',MSE_non1); 
end
fprintf('\n =======================================================================\n')




%% Optimal model
% Outlier data
IN_data=X5;
OUT_data=Y5;
IN_data(Noise_pos2,1)=1000000;
IN_data(Noise_pos1,1)=1000000;
[row_outliers,column_outliers]=find(IN_data(:,1)==1000000);
OUT_data(row_outliers,:)=[];
IN_data(row_outliers,:)=[];

% non linear regression
coefficient_non3=Gradient_Descent_non_linear(IN_data,OUT_data);
n=size(IN_data,1);
% MSE Total
estimated_y=zeros(n,1);
for r=1:n    
for t=1:size(coefficient_non3,1)
    
if t==9
    estimated_y(r,1)=estimated_y(r,1)+ coefficient_non3(t);
elseif t<9 && t>4
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient_non3(t)*IN_data(r,:));
    
elseif t<5 && t>0
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient_non3(t)*((IN_data(r,:).^2))); 
end

end
end
MSE_non5=sum((OUT_data-estimated_y).^2)/n;



%  Linear Regression
coefficient_linear_opt=Gradient_Descent_linear(IN_data,OUT_data);
n=size(IN_data,1);

% MSE Total
estimated_y=zeros(n,1);
for r=1:n    
for t=1:size(coefficient_linear_opt,1)    
if t==size(coefficient_linear_opt,1) 
    estimated_y(r,1)=estimated_y(r,1)+ coefficient_linear_opt(t); 
else
    estimated_y(r,1)=estimated_y(r,1)+sum(coefficient_linear_opt(t)*IN_data(r,:));
end
end
end
MSE3_opt=sum((OUT_data-estimated_y).^2)/n;
fprintf('\n\n --OOOO     Best Regression Model     OOOO--\n')
fprintf('\n 1- Total MSE for data without outlier for linear regression model =  %f',MSE3_opt);
fprintf('\n 2- Total MSE for data without outlier for non linear regression model =  %f',MSE_non5);

if MSE3_opt>MSE_non5
  fprintf('\n Result: Non-linear Regression Model is better');
  fprintf('\n\n 3- Final Regression Model is: \n')
  
rowname = {'Value of coefficients'};
a1 = coefficient(1);
a2 =coefficient(2);
a3 = coefficient(3);
a4 =coefficient(4);
b1 =coefficient(5);
b2 =coefficient(6);
b3 =coefficient(7);
b4 =coefficient(8);
c=coefficient(9);
T1 = table(a1,a2,a3,a4,b1,b2,b3,b4,c,...
'RowNames',rowname)
else
    fprintf('\n Result: Linear Regression Model is better');
      fprintf('\n\n 3- Final Regression Model is: \n')
  
rowname = {'Value of coefficients'};
a1 = coefficient(1);
a2 =coefficient(2);
a3 = coefficient(3);
a4 =coefficient(4);
b=coefficient(5);
T1 = table(a1,a2,a3,a4,b,...
'RowNames',rowname)
end

fprintf('\n =======================================================================\n')

% Plotting
dd1=1:1:500;
dd2=theresh11*ones(1,500);
dd3=theresh21*ones(1,500);
dd4=theresh12*ones(1,500);
dd5=theresh22*ones(1,500);

figure;
subplot(1,3,1)
plot(dd1,dd2,'g')
hold on
plot(dd1,dd4,'g')
hold on
plot(MSE_change1,'r')
axis([0 500 -1 theresh11*2])
title('MSE change for Linear model(with outliers)')

subplot(1,3,2)
plot(dd1,dd3,'g')
hold on
plot(dd1,dd5,'g')
hold on
plot(MSE_change2,'r')
axis([0 500 -1 theresh21*1.2])
title('MSE change for Non-Linear model(with outliers)')

subplot(1,3,3)
if MSE3_opt>MSE_non5
MSE_change2(row_outliers,:)=[];
plot(MSE_change2,'r')
axis([0 500 -1 theresh21*2])
title('MSE change for best Model(without outliers)')
else
MSE_change1(row_outliers,:)=[];
plot(MSE_change1,'r')
axis([0 500 -1 theresh11*2])
title('MSE change for best Model(without outliers)')
end

end





