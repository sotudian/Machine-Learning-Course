% Shahabeddin Sotudian-94125091
% k-Nearest_Position neighbor classifier

function Classification_Results = KNN(Test_Data,Training_Data,k,Type_of_Dist,Output_column)

   % Distance for each data

Distance =pdist2(Test_Data(:,1:Output_column-1), Training_Data(:,1:Output_column-1), Type_of_Dist);
   
   % K Nearest_Position neighbors 
    [~,Nearest_Position] = sort(Distance,2);
    K_Nearest=Nearest_Position(:,1:k);

   % Mode 
   KNN_matrix = reshape(Training_Data(K_Nearest,Output_column),[],k);
   Classification_Results = mode(KNN_matrix,2);

   % if mode is 1
    Classification_Results(Classification_Results==1) = KNN_matrix(Classification_Results==1,1);
    
end

