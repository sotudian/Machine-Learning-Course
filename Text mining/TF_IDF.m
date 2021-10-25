function [ TF_IDF_matrix ] = TF_IDF( Full_matrix )
TF_matrix=zeros(size(Full_matrix));
IDF_matrix=zeros(size(Full_matrix));
TF_IDF_matrix=zeros(size(Full_matrix));
 
% TF
for i=1:size(Full_matrix,1)
    for j=1:size(Full_matrix,2)
     
    TF_matrix(i,j)= Full_matrix(i,j)/sum(Full_matrix(i,:));
    if isnan(TF_matrix(i,j))
     TF_matrix(i,j)=0;   
    end
    
    end
end
    
% IDF 
for i=1:size(Full_matrix,1)
    for j=1:size(Full_matrix,2)
     
        %  the number of documents the term is repeated in
            n = sum( ( Full_matrix(:,j) > 0 ), 1 );
        % compute idf 
            IDF_matrix(i,j)=log((200)/(n+1) ) ;  
    end
end

% TF-IDF
for i=1:size(Full_matrix,1)
    for j=1:size(Full_matrix,2)
        
     TF_IDF_matrix(i,j)=TF_matrix(i,j)* IDF_matrix(i,j);  
    end
end

end

