% Shahab sotudian 94125091
% Determining number of outliers function

function [Number_Outliers ] = Num_Outliers( IN_data )
%% K-Nearest Neighbor for Outlier detection
% Parameter K
k=3;
% Type of distance
Type_of_Dist='euclidean';

for p=1:size(IN_data,1)
 % Distance for each data
Distance =pdist2(IN_data(p,:), IN_data, Type_of_Dist);
% K Nearest_Position neighbors 
    [~,Nearest_Position] = sort(Distance,2);
    K_Nearest=Nearest_Position(:,1:k);
    % Mode 
   KNN_matrix = reshape(IN_data(K_Nearest,4),[],k);
   Final_Results = mode(KNN_matrix,2);
 % if mode is 1
    Final_Results(Final_Results==1) = KNN_matrix(Final_Results==1,1);
    Final_Results_KNN(p,1)=p;
    Final_Results_KNN(p,2)=Final_Results/sum(Distance);
end
[values, order] = sort(Final_Results_KNN(:,2),'descend');
Potential_Outliers=order(1:floor(size(IN_data,1)*1.0),1);
pot_n=floor(size(IN_data,1)*1.0);
%% Z score

Num_Outliers=0;
thresh=3;
Z_data=IN_data(Potential_Outliers,:);
zzscore = (Z_data-repmat(mean(IN_data),pot_n,1))./repmat(std(IN_data),pot_n,1);
for g=1:pot_n
    Norm_mzscore(g,:) = abs(norm(zzscore(g,:)));
end
[K1,K2] = find(abs(Norm_mzscore) > thresh);
if ~isempty(K1),
    outliers =Z_data(K1,:) ;
    outlier_num = size(K1,1);
else
    outlier_num = 0;
end

Number_Outliers=outlier_num;

end

