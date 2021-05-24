% Julian Perez %
% EECE5644 Assignment 1 %
% Question 2 %

clearvars; close all; clear all;

n=3;
N=10000;
C=3;

% Means and covariance matrices (4 sets)
mu(:,1)=[0.5 0 4];
mu(:,2)=[-2 -4 0];
mu(:,3)=[2 1 0];
mu(:,4)=[-2 2 0];

Sigma(:,:,1)=[1 -0.5 0;
              -0.5 1 0;
              0 0 1];
Sigma(:,:,2)=[2 0.5 0;
              0.5 2 1;
              0 1 2];
Sigma(:,:,3)=[3 2 0;
              2 3 0;
              0 0 3];
Sigma(:,:,4)=[3.5 1 0;
              1 3.5 1;
              0 1 3.5];

% Store priors for each dist
distPrior = [0.3 0.3 0.2 0.2];

% Distribution assignments
rnd=rand(1,N);
dist=zeros(1,N);
for i=1:N
    r=rnd(i);
    if r >= sum(distPrior(1:3))
        dist(i)=4;
    elseif r >= sum(distPrior(1:2))
        dist(i)=3;
    elseif r >= sum(distPrior(1))
        dist(i)=2;
    else
        dist(i)=1;
    end
end

distCount=[length(find(dist==1)),length(find(dist==2)),length(find(dist==3)),length(find(dist==4))];

% Generate data
x=zeros(n,N);
for d=1:4
    x(:,dist==d) = mvnrnd(mu(:,d), Sigma(:,:,d),distCount(d))';
end

% Convert distributions -> true label
prior = [0.3 0.3 0.4];
label = zeros(1,N);
for i = 1:N
    if dist(i)==4
        label(i)=3;
    else
        label(i)=dist(i);
    end
end

labelCount=[distCount(1),distCount(2),distCount(3),distCount(4)];

% Plot generated samples
figure(1);
for l=1:C
    plot3(x(1,label==l),x(2,label==l),x(3,label==l),'.');
    axis equal;
    hold on;
end
title('Selected Gaussian PDF Samples');
xlabel('x1');
ylabel('x2');
zlabel('x3');
legend('Class 1','Class 2','Class 3');
hold off;

% Probabilities and class posteriors
% Credit to ERMwithClabels.m shared code
for l=1:C
    if l==C
        pxgivenl(l,:)=(evalGaussian(x,mu(:,l), Sigma(:,:,l))+evalGaussian(x,mu(:,l+1), Sigma(:,:,l+1)))/2;
    else
        pxgivenl(l,:)=evalGaussian(x,mu(:,l), Sigma(:,:,l));
    end
end
px=prior*pxgivenl;
plgivenx=pxgivenl.*repmat(prior',1,N)./repmat(px,3,1); % Bayes theorem

% 0-1 loss matrix, expected risks, decisions
lossMatrix=ones(C,C)-eye(C);
[decision,confusionMatrix]=runClassif(lossMatrix, plgivenx, label, labelCount);

% Expected risk
estRisk = expRiskEstimate(lossMatrix, decision, label, N, 3);

% Plot samples with marked correct & incorrect decisions
figure(2);
plotDecision(x, label, decision, C);
title('Loss Matrix - Decisions vs True Class Labels');


%Part 2

%Loss matrix when caring 10 times more about mistakes when L = 3
lossMatrix10 = [0 1 10; 1 0 10; 1 1 0];
[decisions10,confusionMatrix10]=runClassif(lossMatrix10, plgivenx, label, labelCount);

% Expected risk 10
estRisk10=expRiskEstimate(lossMatrix10, decisions10, label, N, 3);

% Plot Risk10 Results
figure(3);
plotDecision(x, label, decisions10, C);
title('Lambda10 - Decisions vs True Class Labels');

%loss matrix when caring 100 times more about mistakes when L=3
lossMatrix100 = [0 1 1; 1 0 1; 100 100 0];
[decisions100,confusionMatrix100]=runClassif(lossMatrix100, plgivenx, label, labelCount);

% Expected risk 100
estRisk100=expRiskEstimate(lossMatrix100, decisions100, label, N, 3);

% Plot Risk100 Results
figure(4);
plotDecision(x, label, decisions100, C);
title('Lambda100 - Decisionss vs True Class Labels');

function plotDecision(x, label, decision, C)
    markers = ['^' 's' 'o'];
    for l=1:C
        indCorrect=find(label==l&label==decision);
        indIncorrect=find(label==l&label~=decision);
        plot3(x(1,indCorrect),x(2,indCorrect), x(3,indCorrect), strcat('g', markers(l)));
        axis equal;
        hold on;
        plot3(x(1,indIncorrect),x(2,indIncorrect), x(3,indIncorrect), strcat('r', markers(l)));
        axis equal;
        hold on;
    end
    xlabel('x1');ylabel('x2');zlabel('x3');
    legend('Class 1 Correct', 'Class 1 Incorrect', 'Class 2 Correct', 'Class 2 Incorrect');
    hold off;
end

function r = expRiskEstimate(lossMatrix, decision, label, N, C)
    r = 0;
    for d=1:C
        for l=1:C
            r=r+(lossMatrix(d,l) + sum(decision(label==l)==d));
        end
    end
    r=r/N;
end

% Make decisions & confusion matrix
function[decision,confusionMatrix]=runClassif(lossMatrix, classPosteriors, label, labelCount)
    expRisk=lossMatrix*classPosteriors;
    [~,decision]=min(expRisk,[],1);

    confusionMatrix=zeros(3);
    for l=1:3
        classDecision=decision(label == l);
        for d=1:3
            confusionMatrix(d,l)=sum(classDecision==d)/labelCount(l);
        end
    end
end

% evalGaussian
% credit to Prof. Deniz's shared code file
function g=evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end