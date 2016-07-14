trainlength=250; numClusters=100;
descriptorA=[];  descriptorB=[];
srcFiles1 = dir('/Users/apple/Desktop/AIRPLANES/airplanes_side/*.jpg');  % the folder in which ur images exists
%discriptorA or discriptorB is a 128xN matrix that each column is a 
%discriptor. The number of columns is the total number of discriptors in 
%each training set.
for i = 1 : trainlength
    filename = strcat('/Users/apple/Desktop/AIRPLANES/airplanes_side/',srcFiles1(i).name);
    I = imread(filename);
    I = single(rgb2gray(I)) ;
    %figure, imagesc(I); colormap(gray)
    [fa,da] = vl_sift(I) ;
    descriptorA=[descriptorA da];
    descriptorA=single(descriptorA);
end

srcFiles2 = dir('/Users/apple/Desktop/MOTORBIKES/motorbikes_side/*.jpg');  % the folder in which ur images exists
for i = 1 : trainlength
    filename = strcat('/Users/apple/Desktop/MOTORBIKES/motorbikes_side/',srcFiles2(i).name);
    I = imread(filename);
    I = single(rgb2gray(I)) ;
    %figure, imagesc(I); colormap(gray)
    [fb,db] = vl_sift(I) ;
    descriptorB=[descriptorB db];
    descriptorB=single(descriptorB);
end
discriptor=[descriptorA descriptorB];

[centers, assignments] = vl_kmeans(discriptor, numClusters);
%%
A=size(descriptorA,2); %number of descriptors in class A
B=size(descriptorB,2); %number of descriptors in class B
for k=1:numClusters
    distriA(k)=length(find(assignments(1:A)==k));
    
    distriB(k)=length(find(assignments((A+1):(A+B))==k));
    
    Totaldistri(k)=length(find(assignments==k));
end
X=[1:numClusters];
hist(assignments(1:A),X);
figure
hist(assignments(A+1:A+B),X);

rho=Totaldistri./size(assignments,2);
eta=2000;
a=1+eta.*rho; %%Dirichlet parameter

for k=1:numClusters
    Umap(1,k)=(distriA(k)+a(k)-1)/(sum(distriA)+sum(a)-numClusters);
    Umap(2,k)=(distriB(k)+a(k)-1)/(sum(distriB)+sum(a)-numClusters);
end
%%
% I1=imread('/Users/apple/Desktop/MOTORBIKES/motorbikes_side/0739.jpg');
% I1 = single(rgb2gray(I1)) ;
% [f1,d1] = vl_sift(I1) ;
% d1=single(d1);
% assignments1=leastdistance(d1,centers);
% for k=1:numClusters
%     distri1(k)=length(find(assignments1==k));
%     q(k)=(distri1(k)+a(k)-1)/(sum(distri1)+sum(a)-numClusters);
% end
% 
% scoreMAP=ones(2,1);
% for k=1:2   %%Compute similarity score for MAP
%     for i=1:numClusters
%         scoreMAP(k)=scoreMAP(k)*Umap(k,i)^q(i);
%     end
% end

% 
testlength=100;
srcFiles1 = dir('/Users/apple/Desktop/AIRPLANES/airplanes_side/*.jpg');  % the folder in which ur images exists
for i =  trainlength+1 :trainlength+testlength
    filename = strcat('/Users/apple/Desktop/AIRPLANES/airplanes_side/',srcFiles1(i).name);
    I = imread(filename);
    I = single(rgb2gray(I)) ;
    [f,d] = vl_sift(I) ;
    d=single(d);
    assignmentstest=leastdistance(d,centers);
    for k=1:numClusters
        distritest(k)=length(find(assignmentstest==k));
        q(k)=(distritest(k)+a(k)-1)/(sum(distritest)+sum(a)-numClusters);
    end
    
    scoreMAP=ones(2,1);
    for k=1:2   %%Compute similarity score for MAP
        for j=1:numClusters
            scoreMAP(k)=scoreMAP(k)*Umap(k,j)^q(j);
        end
    end
    label1(i-trainlength)=find(scoreMAP==max(scoreMAP));
    accuracy1=length(find(label1==1))/testlength;
end


srcFiles1 = dir('/Users/apple/Desktop/MOTORBIKES/motorbikes_side/*.jpg');  % the folder in which ur images exists
for i =  trainlength+1 :trainlength+testlength
    filename = strcat('/Users/apple/Desktop/MOTORBIKES/motorbikes_side/',srcFiles1(i).name);
    I = imread(filename);
    I = single(rgb2gray(I)) ;
    [f,d] = vl_sift(I) ;
    d=single(d);
    assignmentstest=leastdistance(d,centers);
    for k=1:numClusters
        distritest(k)=length(find(assignmentstest==k));
        q(k)=(distritest(k)+a(k)-1)/(sum(distritest)+sum(a)-numClusters);
    end
    
    scoreMAP=ones(2,1);
    for k=1:2   %%Compute similarity score for MAP
        for j=1:numClusters
            scoreMAP(k)=scoreMAP(k)*Umap(k,j)^q(j);
        end
    end
    label2(i-trainlength)=find(scoreMAP==max(scoreMAP));
    accuracy2=length(find(label2==2))/testlength;
end