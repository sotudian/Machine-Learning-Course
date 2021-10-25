NBModel = fitNaiveBayes(Train_full_matrix(:,1:10770),Train_full_matrix(:,10771),'Distribution','mn');
predSpam = predict(NBModel,Train_full_matrix(:,1:10770));
misclass = sum(Train_full_matrix(:,10771)~=predSpam)/n