% Shahabeddin Sotudian-94125091
function [ Balanced_data ] = Under_sampling( ArrhythmiaDataset_without_missing )
% Under_sampling for class 1
w=find(ArrhythmiaDataset_without_missing(:,279)==1);
d=datasample(w,165,'Replace',false);
ArrhythmiaDataset_without_missing(d,:)=[];
Balanced_data=ArrhythmiaDataset_without_missing;
end