 %% Shahab sotudian
 % 1- Transferring sparse matrixes  to counting matrixes  
 % 2- Computing TF-IDF matrixes 
 % 3- Adding Outputs column to these matrixes 
 clear
 clc
 
%%  Train data
 load('IN_sparse_train.mat') 
 load('Out_sparse_train')
 Train_full_matrix=zeros(max(IN_sparse_train(:,1)),max(IN_sparse_train(:,2)));
 
 for i=1:size(IN_sparse_train,1)
 Train_full_matrix(IN_sparse_train(i,1),IN_sparse_train(i,2))=IN_sparse_train(i,3);
 end
 
 % TF-IDF Matrix
 [ TF_IDF_matrix_Train ] = TF_IDF( Train_full_matrix );
 
 % Adding output column
 Train_full_matrix=cat(2,Train_full_matrix,Out_sparse_train);
 TF_IDF_matrix_Train=cat(2,TF_IDF_matrix_Train,Out_sparse_train);
 
 
 %% Test data
 load('IN_sparse_test.mat') 
 load('Out_sparse_test')
 Test_full_matrix=zeros(max(IN_sparse_test(:,1)),max(IN_sparse_test(:,2)));
 
 for i=1:size(IN_sparse_test,1)
 Test_full_matrix(IN_sparse_test(i,1),IN_sparse_test(i,2))=IN_sparse_test(i,3);
 end
 
 % TF-IDF Matrix
 [ TF_IDF_matrix_Test ] = TF_IDF( Test_full_matrix );
 
  % Adding output column
 Test_full_matrix=cat(2,Test_full_matrix,Out_sparse_test);
 TF_IDF_matrix_Test=cat(2,TF_IDF_matrix_Test,Out_sparse_test);
 
 