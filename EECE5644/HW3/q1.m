% Julian Perez %
% EECE5644 Assignment 3 %
% Question 1 %

% Expected risk minimization with 2 classes
clear all; close all; clc;

dim=3; %Dimensions
numLabel=4;
Lx={'L0','L1','L2','L3'};

% For min-Perror use 0-1 loss
lossMatrix = ones(numLabel,numLabel)-eye(numLabel);
muScale=2.5;
SigScale=0.2;

% Define data
D.d100.N=100;
D.d200.N=200;
D.d500.N=500;
D.d1k.N=1e3;
D.d2k.N=2e3;
D.d5k.N=5e3;
D.d100k.N=100e3;
dType=fieldnames(D);

p=ones(1,numLabel)/numLabel; %Prior

% Label datasets
mu.L0=muScale*[1 1 0]';
RandSig=SigScale*rand(dim,dim);
Sigma.L0(:,:,1)=RandSig*RandSig'+eye(dim);
mu.L1=muScale*[1 0 0]';
RandSig=SigScale*rand(dim,dim);
Sigma.L1(:,:,1)=RandSig*RandSig'+eye(dim);
mu.L2=muScale*[0 1 0]';
RandSig=SigScale*rand(dim,dim);
Sigma.L2(:,:,1)=RandSig*RandSig'+eye(dim);
mu.L3=muScale*[0 0 1]';
RandSig=SigScale*rand(dim,dim);
Sigma.L3(:,:,1)=RandSig*RandSig'+eye(dim);


% Generate Data
for ind=1:length(dType)
    D.(dType{ind}).x=zeros(dim,D.(dType{ind}).N); %Initialize Data
    [D.(dType{ind}).x,D.(dType{ind}).labels,D.(dType{ind}).N_l,D.(dType{ind}).p_hat]=genData(D.(dType{ind}).N,p,mu,Sigma,Lx,dim);
end

% Plot training data
figure;
for ind=1:length(dType)-1
    subplot(3,2,ind);
    plotData(D.(dType{ind}).x,D.(dType{ind}).labels,Lx);
    legend 'show';
    title([dType{ind}]);
end

% Plot validation data
figure;
plotData(D.(dType{ind}).x,D.(dType{ind}).labels,Lx);
legend 'show';
title([dType{end}]);


% Determine theoretically O=optimal classifier
for ind=1:length(dType)
    [D.(dType{ind}).opt.PFE, D.(dType{ind}).opt.decisions]=optClass(lossMatrix,D.(dType{ind}).x,mu,Sigma,p,D.(dType{ind}).labels,Lx);
    opPFE(ind)=D.(dType{ind}).opt.PFE;
    fprintf('Optimal pFE, N=%1.0f: Error=%1.2f%%\n',D.(dType{ind}).N,100*D.(dType{ind}).opt.PFE);
end


% Train and Validate Data

numPerceptron=15; %Max number of perceptrons to attempt to train
k=10; %number of folds for kfold validation
for ind=1:length(dType)-1
    %kfold validation is in this function
    [D.(dType{ind}).net,D.(dType{ind}).minPFE,D.(dType{ind}).optM,valData.(dType{ind}).stats]=kfoldMLP_NN(numPerceptron,k,D.(dType{ind}).x,D.(dType{ind}).labels,numLabel);
    %Produce validation data from test dataset
    valData.(dType{ind}).yVal=D.(dType{ind}).net(D.d100k.x);
    [~,valData.(dType{ind}).decisions]=max(valData.(dType{ind}).yVal);
    valData.(dType{ind}).decisions=valData.(dType{ind}).decisions-1;
    %Probability of Error is wrong decisions/num data points
    valData.(dType{ind}).pFE=sum(valData.(dType{ind}).decisions~=D.d100k.labels)/D.d100k.N;
    outpFE(ind,1)=D.(dType{ind}).N;
    outpFE(ind,2)=valData.(dType{ind}).pFE;
    outpFE(ind,3)=D.(dType{ind}).optM;
    fprintf('Neural Net pFE, N=%1.0f: Error=%1.2f%%\n',D.(dType{ind}).N,100*valData.(dType{ind}).pFE);
end

% Get cross validation results
for ind=1:length(dType)-1
    [~,select]=min(valData.(dType{ind}).stats.mPFE);
    
    M(ind)=(valData.(dType{ind}).stats.M(select));
    N(ind)=D.(dType{ind}).N;
end

% Plot number of perceptrons vs. pFE
for ind=1:length(dType)-1
    figure;
    stem(valData.(dType{ind}).stats.M,valData.(dType{ind}).stats.mPFE);
    xlabel('Number of Perceptrons');
    ylabel('pFE');
    title(['Error Probability vs. Perceptron Count for ' dType{ind}]);
end

% Number of perceptrons vs. training dataset size
figure,semilogx(N(1:end-1),M(1:end-1),'o','LineWidth',2)
grid on;
xlabel('Number of Data Points')
ylabel('Optimal Perceptron Count')
ylim([0 10]);
xlim([50 10^4]);
title('Optimal Perceptron Count vs. Number of Data Points');
% Probability of error vs. training dataset size
figure,semilogx(outpFE(1:end-1,1),outpFE(1:end-1,2),'o','LineWidth',2)
xlim([90 10^4]);
hold all;semilogx(xlim,[opPFE(end) opPFE(end)],'r--','LineWidth',2)
legend('Neural Net pFE','Optimal pFE')
grid on
xlabel('Number of Data Points')
ylabel('pFE')
title('Probability of Error vs. Data Points in Training Data');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,label,N_l,p_hat]= genData(N,p,mu,Sigma,Lx,d)
%Generates data and labels for random variable x from multiple gaussian
%distributions
numDist = length(Lx);
cumul_p = [0,cumsum(p)];
u = rand(1,N);
x = zeros(d,N);
label = zeros(1,N);
for ind=1:numDist
    pts = find(cumul_p(ind)<u & u<=cumul_p(ind+1));
    N_l(ind)=length(pts);
    x(:,pts) = mvnrnd(mu.(Lx{ind}),Sigma.(Lx{ind}),N_l(ind))';
    label(pts)=ind-1;
    p_hat(ind)=N_l(ind)/N;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotData(x,label,Lx)
%Plots data
for ind=1:length(Lx)
    p_index=label==ind-1;
    plot3(x(1,p_index),x(2,p_index),x(3,p_index),'.','DisplayName',Lx{ind});
    hold all;
end
grid on;
xlabel('x1');
ylabel('x2');
zlabel('x3');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSig = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSig)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSig*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [minPFE,decision]=optClass(lossMatrix,x,mu,Sigma,p,label,Lx)
% Determine optimal probability of error
symbols='ox+*v';
numLabels=length(Lx);
N=length(x);
for ind = 1:numLabels
    pxgivenl(ind,:) = evalGaussian(x,mu.(Lx{ind}),Sigma.(Lx{ind})); % Evaluate p(x|L=l)
end
px = p*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(p',1,N)./repmat(px,numLabels,1);
% Expected Risk for each label (rows) for each sample (columns)
expectedRisks =lossMatrix*classPosteriors;
% Minimum expected risk decision with 0-1 loss is the same as MAP
[~,decision] = min(expectedRisks,[],1);
decision=decision-1; %Adjust to account for L0 label
fDecision_ind=(decision~=label);%Incorrect classificiation vector
minPFE=sum(fDecision_ind)/N;
%Plot Decisions with Incorrect Results
figure;
for ind=1:numLabels
    class_ind=decision==ind-1;
    plot3(x(1,class_ind & ~fDecision_ind),x(2,class_ind & ~fDecision_ind),x(3,class_ind & ~fDecision_ind),symbols(ind),'Color',[0.39 0.83 0.07],'DisplayName',['Class ' num2str(ind) ' Correct Classification']);
    hold on;
    plot3(x(1,class_ind & fDecision_ind),x(2,class_ind & fDecision_ind),x(3,class_ind & fDecision_ind),['r' symbols(ind)],'DisplayName',['Class ' num2str(ind) ' Incorrect Classification']);
    hold on;
end
xlabel('x1');
ylabel('x2');
grid on;

title('X Vector - Incorrect Classifications');
legend 'show';
if 0
    % Plot Decisions with incorrect decisions
    figure;
    for ind2=1:numLabels
        subplot(3,2,ind2);
        for ind=1:numLabels
            class_ind=decision==ind-1;
            plot3(x(1,class_ind),x(2,class_ind),x(3,class_ind),'.','DisplayName',['Class ' num2str(ind)]);
            hold on;
        end
        plot3(x(1,fDecision_ind & label==ind2),x(2,fDecision_ind & label==ind2),x(3,fDecision_ind & label==ind2),'kx','DisplayName','Incorrectly Classified','LineWidth',2);
        ylabel('x2');
        grid on;
        title(['X Vector with Incorrect Decisions for Class ' num2str(ind2)]);
        if ind2==1
            legend 'show';
        elseif ind2==4
            xlabel('x1');
        end
    end
end
end

% Cross validation and model selection
function [outNet,outPFE, optM, stats]=kfoldMLP_NN(numPerc,k,x,label,numLabel)
%Assumes data is evenly divisible by partition choice which it should be
N=length(x);
numValIterations=10;
%Create output matrices from labels
y=zeros(numLabel,length(x));
for ind=1:numLabel
    y(ind,:)=(label==ind-1);
end
%Setup cross validation on training data
partSize=N/k;
partInd=[1:partSize:N length(x)];
%Perform cross validation to select number of perceptrons
for M=1:numPerc
    for ind=1:k
        
        index.val=partInd(ind):partInd(ind+1);
        index.train=setdiff(1:N,index.val);
        %Create object with M perceptrons in hidden layer
        net=patternnet(M);
        %Train using training data
        net=train(net,x(:,index.train),y(:,index.train));
        %Validate with remaining data
        yVal=net(x(:,index.val));
        [~,labelVal]=max(yVal);
        labelVal=labelVal-1;
        pFE(ind)=sum(labelVal~=label(index.val))/partSize;
    end
    %Determine average probability of error for a number of perceptrons
    avgPFE(M)=mean(pFE);
    stats.M=1:M;
    stats.mPFE=avgPFE;
end
%Determine optimal number of perceptrons
[~,optM]=min(avgPFE);
%Train one final time on all the data
for ind=1:numValIterations
    netName(ind)={['net' num2str(ind)]};
    finalnet.(netName{ind})=patternnet(optM);
    finalnet.(netName{ind})=train(net,x,y);
    yVal=finalnet.(netName{ind})(x);
    [~,labelVal]=max(yVal);
    labelVal=labelVal-1;
    pFEFinal(ind)=sum(labelVal~=label)/length(x);
end
[minPFE,outInd]=min(pFEFinal);
stats.finalPFE=pFEFinal;
outPFE=minPFE;
outNet=finalnet.(netName{outInd});
end