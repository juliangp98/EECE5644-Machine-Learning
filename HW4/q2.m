% Julian Perez %
% EECE5644 Assignment 4 %
% Question 2 %

clear all;
close all;
warning off;
files = {'3096_color.jpg';'42049_color.jpg'};
dTypes={'c3096' 'c42049'};

% Cross-validation parameters
nGMM=10; % Number of GMMs
k=10; % Number of folds

for ind=1:length(files)
    % Read in data
    imageData = imread(files{ind});
    figure(1);
    subplot(length(files),3,(ind-1)*3+1);
    imshow(imageData);
    
    [R,C,D]=size(imageData);
    N=R*C;
    imageData=double(imageData);
    rows=(1:R)'*ones(1,C);
    columns=ones(R,1)*(1:C);
    featureData=[rows(:)';columns(:)'];
    
    for ind2=1:D
        imdatad=imageData(:,:,ind2);
        featureData=[featureData; imdatad(:)'];
    end
    
    minf=min(featureData,[],2);
    maxf=max(featureData,[],2);
    ranges=maxf-minf;
    
    % Normalize data
    x=(featureData-minf)./ranges;
    
    % Assess for GMM with 2 Gaussians
    GMM2=fitgmdist(x',2,'RegularizationValue',0.025);
    post2=posterior(GMM2,x')';
    decision=post2(2,:)>post2(1,:);
    
    % Plot GMM=2 case
    labelImage2=reshape(decision,R,C);
    subplot(length(files),3,(ind-1)*3+2);
    imshow(uint8(labelImage2*255/2));
    
    % Perform k-fold cross-validation to determine optimal num Gaussians
    N=length(x);
    numValIterations=10;
    
    % Set up cross validation on training data
    partSize=floor(N/k);
    partInd=[1:partSize:N length(x)];
    
    % Perform cross-validation on each number of perceptrons
    for numGMM=1:nGMM
        for numK=1:k
            index.val=partInd(numK):partInd(numK+1);
            index.train=setdiff(1:N,index.val);
            
            % Create object with M perceptrons in hidden layer
            GMMk_loop=fitgmdist(x(:,index.train)',numGMM,'Replicates',5);
            if GMMk_loop.Converged
                probX(numK)=sum(log(pdf(GMMk_loop,x(:,index.val)')));
            else
                probX(numK)=0;
            end
        end
        
        % Determine average probability of error for a number of perceptrons
        avgProb(ind,numGMM)=mean(probX);
        stat(ind).NumGMMs=1:nGMM;
        stat(ind).avgProb=avgProb;
        stat(ind).mProb(:,numGMM)=probX;
        fprintf('File: %1.0f, NumGMM: %1.0f\n',ind,numGMM);
    end
    
    % Select GMM with maximum probability
    [~,optNumGMM]=max(avgProb(ind,:));
    GMMk=fitgmdist(x',optNumGMM,'Replicates',10);
    postk=posterior(GMMk,x')';
    lossMatrix=ones(optNumGMM,optNumGMM)-eye(optNumGMM);
    expectedRisk =lossMatrix*postk;
    [~,decision] = min(expectedRisk,[],1);
    
    % Plot segmented image for GMM max-likelihood
    labelImageK=reshape(decision-1,R,C);
    subplot(length(files),3,(ind-1)*3+3);
    imshow(uint8(labelImageK*255/2));
    save(['HW4Q2' num2str(ind) '.mat']);
end