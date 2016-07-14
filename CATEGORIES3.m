close all;
trainlength = 200; numClusters = 200;
descriptorA = [];  descriptorB = []; descriptorC = [];

% the folder in which ur images exists
%discriptora or discriptorb is a 128xN matrix that each column is a
%discriptor. The nuberof columns is the total number of discriptors in
%each training set.

srcFiles1 = dir('/Users/apple/Desktop/GUITARS/guitars/*.jpg');
for i = 1 : trainlength
    filename = strcat('/Users/apple/Desktop/GUITARS/guitars/',srcFiles1(i).name);
    I = imread(filename);
    if size(I,3) == 3
        I = single(rgb2gray(I));
    end
        I = single(I);
    %figure, imagesc(I); colormap(gray)
    [fa,da] = vl_sift(I) ;
    descriptorA = [descriptorA da];
    descriptorA = single(descriptorA);
end

srcFiles2 = dir('/Users/apple/Desktop/MOTORBIKES/motorbikes_side/*.jpg');  % the folder in which ur images exists
for i = 1 : trainlength
    filename = strcat('/Users/apple/Desktop/MOTORBIKES/motorbikes_side/',srcFiles2(i).name);
    I = imread(filename);
    if size(I,3) == 3
        I = single(rgb2gray(I));
    end
    I = single(I);
    %figure, imagesc(I); colormap(gray)
    [fb,db] = vl_sift(I) ;
    descriptorB = [descriptorB db];
    descriptorB = single(descriptorB);
end

srcFiles3 = dir('/Users/apple/Desktop/HOUSES/houses/*.jpg');  % the folder in which ur images exists
for i = 1 : trainlength
    filename = strcat('/Users/apple/Desktop/HOUSES/houses/',srcFiles3(i).name);
    I = imread(filename);
    if size(I,3) == 3
        I = single(rgb2gray(I));
    end
    I = single(I);
    %figure, imagesc(I); colormap(gray)
    [fc,dc] = vl_sift(I) ;
    descriptorC = [descriptorC dc];
    descriptorC = single(descriptorC);
end

discriptor = [descriptorA descriptorB descriptorC];

[centers, assignments] = vl_kmeans(discriptor, numClusters);
%%
A = size(descriptorA,2); %number of descriptors in class A
B = size(descriptorB,2); %number of descriptors in class B
C = size(descriptorC,2);
for k = 1:numClusters
    distriA(k) = length(find(assignments(1:A) == k));
    
    distriB(k) = length(find(assignments((A + 1):(A + B)) == k));
    
    distriC(k) = length(find(assignments((A + B + 1):(A + B + C)) == k));
    
    Totaldistri(k) = length(find(assignments == k));
end
X = 1:numClusters;
hist(assignments(1:A),X);
figure
hist(assignments(A + 1:A + B),X);
figure
hist(assignments(A + B + 1:A + B + C),X);

rho = Totaldistri ./ size(assignments,2);
eta = 2000;
a = 1 + eta .* rho; %%Dirichlet parameter

for k = 1:numClusters
    Umap(1,k) = (distriA(k) + a(k) - 1) / (sum(distriA) + sum(a) - numClusters);
    Umap(2,k) = (distriB(k) + a(k) - 1) / (sum(distriB) + sum(a) - numClusters);
    Umap(3,k) = (distriC(k) + a(k) - 1) / (sum(distriC) + sum(a) - numClusters);
end

%%
testlength = 100;

for i =  trainlength + 1 :trainlength + testlength
    filename = strcat('/Users/apple/Desktop/GUITARS/guitars/',srcFiles1(i).name);
    I = imread(filename);
    if size(I,3) == 3
        I = single(rgb2gray(I));
    end
    I = single(I);
    [f,d] = vl_sift(I) ;
    d = single(d);
    assignmentstest = leastdistance(d,centers);
    for k = 1:numClusters
        distritest(k) = length(find(assignmentstest == k));
        q(k) = (distritest(k) + a(k) - 1)/(sum(distritest) + sum(a) - numClusters);
    end
    
    scoreMAP = ones(3,1);
    for k = 1:3   %%Compute similarity score for MAP
        for j = 1:numClusters
            scoreMAP(k) = scoreMAP(k) * Umap(k,j) ^ q(j);
        end
    end
    label1(i - trainlength) = find(scoreMAP == max(scoreMAP));
    accuracy1 = length(find(label1 == 1)) / testlength;
end

for i =  trainlength + 1 :trainlength + testlength
    filename = strcat('/Users/apple/Desktop/MOTORBIKES/motorbikes_side/',srcFiles2(i).name);
    I = imread(filename);
    if size(I,3) == 3
        I = single(rgb2gray(I));
    end;
    I = single(I);
    [f,d] = vl_sift(I) ;
    d = single(d);
    assignmentstest = leastdistance(d,centers);
    for k=1:numClusters
        distritest(k) = length(find(assignmentstest == k));
        q(k) = (distritest(k) + a(k) - 1) / (sum(distritest) + sum(a) - numClusters);
    end
    
    scoreMAP = ones(3,1);
    for k = 1:3   %%Compute similarity score for MAP
        for j = 1:numClusters
            scoreMAP(k) = scoreMAP(k) * Umap(k,j) ^ q(j);
        end
    end
    label2(i - trainlength) = find(scoreMAP == max(scoreMAP));
    accuracy2 = length(find(label2 == 2)) / testlength;
end

for i =  trainlength + 1 :trainlength + testlength
    filename = strcat('/Users/apple/Desktop/HOUSES/houses/',srcFiles3(i).name);
    I = imread(filename);
    if size(I,3) == 3
        I = single(rgb2gray(I));
    end
    I = single(I);
    [f,d] = vl_sift(I) ;
    d = single(d);
    assignmentstest = leastdistance(d,centers);
    for k = 1:numClusters
        distritest(k) = length(find(assignmentstest == k));
        q(k) = (distritest(k) + a(k) - 1) / (sum(distritest) + sum(a) - numClusters);
    end
    
    scoreMAP = ones(3,1);
    for k = 1:3   %%Compute similarity score for MAP
        for j = 1:numClusters
            scoreMAP(k) = scoreMAP(k) * Umap(k,j) ^ q(j);
        end
    end
    label3(i - trainlength) = find(scoreMAP == max(scoreMAP));
    accuracy3 = length(find(label3 == 3)) / testlength;
end

