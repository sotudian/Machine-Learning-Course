% Shahab Sotudian 94125091

clc;
clear;
close all;


%% load data
load x_onedim;
load y;
x_rasm=zeros(1,size(x,2));
n= numel(y);
ClassA = find(y==1);
ClassB = find(y==-1);

x1_map=[0,0,0,0,0];
x2_map=[9,0,1,4,16];
figure;
plot(x1_map(1,ClassA),x2_map(ClassA),'ro');
axis([-5 5 -5 25])
hold on;
plot(x1_map(1,ClassB),x2_map(ClassB),'bs');
axis([-5 5 -5 25])
legend('+1','-1');

%% Design SVM

C=1000;

Kernel=@(xi,xj) ((xi'*xj).^2);


H=zeros(n,n);
for i=1:n
    for j=i:n
        H(i,j)=y(i)*y(j)*Kernel(x(:,i),x(:,j));
        H(j,i)=H(i,j);
    end
end



f = -ones (n,1);

Aeq = y;
beq = 0;

lb = zeros(n,1);
ub = C*ones(n,1);

Alg{1} = 'interior-point-convex';
Alg{2} = 'trust-region-reflective';
Alg{3} = 'active-set';
options = optimset('Algorithm', Alg{1},...
    'Display','off',...
    'MaxIter',20);

alpha = quadprog(H,f,[],[], Aeq,beq,lb,ub,[],options);
alpha = alpha';
AlmostZero = (abs(alpha)< max(abs(alpha))/1e5);
alpha(AlmostZero) = 0;

S= find(alpha >0 & alpha < C);


% bias
b=0;
for i=S
    b=b+y(i)-MySVRFunc(x(i),alpha(S),y(S),x(S),Kernel);
end
b=b/numel(S)



% W
Fi_x(1,:)=[0,0,0,0,0];
Fi_x(2,:)=x(1,:).^2;

w=0;
for i=S
    w=w+alpha(i)*y(i)*Fi_x(:,i);
end
w




%% Plot Results

Curve=@(x1) MySVRFunc(x1,alpha(S),y(S),x(:,S),Kernel)+b;
CurveA=@(x1) MySVRFunc(x1,alpha(S),y(S),x(:,S),Kernel)+b+1;
CurveB=@(x1) MySVRFunc(x1,alpha(S),y(S),x(:,S),Kernel)+b-1;

figure;
plot(x(1,ClassA),x_rasm(ClassA),'ro');
% axis([-10 10 -5 5])
hold on;
plot(x(1,ClassB),x_rasm(ClassB),'bs');
% axis([-10 10 -5 5])
% legend('+1','-1');
plot(x(1,S),x_rasm(S),'ko','MarkerSize',12);
x1min=-10;
x1max=10;
x2min=-10;
x2max=10;

handle=ezplot(Curve,[x1min x1max x2min x2max]);
set(handle,'Color','k','LineWidth',2);

handleA=ezplot(CurveA,[x1min x1max x2min x2max]);
set(handleA,'Color','k','LineWidth',1,'LineStyle',':');

handleB=ezplot(CurveB,[x1min x1max x2min x2max]);
set(handleB,'Color','k','LineWidth',1,'LineStyle',':');

legend('+1','-1','support Vectors');





