% Julian Perez %
% EECE5644 Assignment 1 %
% Question 1 %

clearvars; close all; clear all;

% Setup Initial Data

n=4;
N=10000;
p=[0.7,0.3]; % Class priors for P(L=0, P(L=1)

% Mean and covariance matrices
m(:,1) = [-1 1 -1 1];
m(:,2) = [1 1 1 1];

C(:,:,1) = [3 -0.6 0.2 0;
            -0.6 1 -0.6 0;
            0.2 -0.6 1 0;
            0 0 0 3];
C(:,:,2) = [2 0.2 -0.2 0;
            0.2 2 0.2 0;
            -0.2 0.2 2 0;
            0 0 0 3];

% True class labels for N samples
label=rand(1,N) >= p(1);
totLabel=[length(find(label==0)),length(find(label==1))];

% Generate the samples
x=zeros(n,N);
for l=0:1
    x(:,label==l)=mvnrnd(m(:,l+1),C(:,:,l+1),totLabel(l+1))';
end

% Fisher LDA Visual
% Credit to fisherLDA.m code

% Between/within scatter matrices
Sb = (m(:,1)-m(:,2))*(m(:,1)-m(:,2))';
Sw = C(:,:,1)+C(:,:,2);

% Fisher LDA projection vector
[V,D]=eig(inv(Sw)*Sb);
[~,ind]=sort(diag(D),'descend');
w=V(:,ind); %fisher LDA projection vector
y=w'*x;

% LDA Plot (3 Dimensions)
figure(1);
plot3(y(1, label==0), y(2, label==0), y(3, label==0), '.');
hold on;
plot3(y(1, label==1), y(2, label==1), y(3, label==1), '.');
axis equal;
title('Fisher Projection: 1st 3 Dimensions');
xlabel('y1'); ylabel('y2'); zlabel('y3');
legend('Class 0 Fisher LDA', 'Class 1 Fisher LDA');

%Part 1: ERM Classification

% Discriminant scores
discrimScore=log(evalGaussian(x,m(:,2),C(:,:,2)))-log(evalGaussian(x,m(:,1),C(:,:,1)));

% ROC Curve - gamma list
[sortScores]=sort(discrimScore,'ascend');
threshList = [min(sortScores)-eps,(sortScores(1:end-1)+sortScores(2:end))/2,max(sortScores)];
    
% Error vectors, make decisions, and calculate error probs
pFP=zeros(1,length(threshList));
pTP=zeros(1,length(threshList));
pFN=zeros(1,length(threshList));
pTN=zeros(1,length(threshList));
pErr=zeros(1,length(threshList));

for i=1:length(threshList)
    decisions=discrimScore >= threshList(i);
    pFP(i)=sum(decisions==1&label==0)/totLabel(1);
    pTP(i)=sum(decisions==1&label==1)/totLabel(2);
    pFN(i)=sum(decisions==0&label==1)/totLabel(2);
    pTN(i)=sum(decisions==0&label==0)/totLabel(1);
    pErr(i)=pFP(i)*p(1)+pFN(i)*p(2);
end

% experimental
[minError,minInd]=min(pErr);
minGamma=exp(threshList(minInd));
minDecision=(discrimScore >= threshList(minInd));
minFalseAlarm=pFP(minInd);
minDetection = pTP(minInd);

% theoretical
t_decision=(discrimScore>=log(p(1)/p(2))); % use class priors for threshold
t_pFalseAlarm=sum(t_decision==1&label==0)/totLabel(1);
t_pDetection=sum(t_decision == 1&label==1)/totLabel(2);
t_pErr=t_pFalseAlarm*p(1)+t_pDetection*p(2);

figure(2);
plot(pFP, pTP, '-', minFalseAlarm, minDetection, 'go', t_pFalseAlarm, t_pDetection, 'm+');
title('ROC Curve - Proportion Error vs. Gamma'); legend('ROC Curve', 'Experimental Minimum Error', 'Theoretical Minimum Error');
xlabel('pFalseAlarm');
ylabel('pDetection');



%Part 2: Naive Bayesian Classifier

% New NBC covariance matrices
C_NB(:,:,1)=[2 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2];
C_NB(:,:,2)=[1 0 0 0; 0 2 0 0; 0 0 1 0; 0 0 0 3];

% Discriminant scores
discrimScore_NB=log(evalGaussian(x, m(:,2),C_NB(:,:,2)))-log(evalGaussian(x,m(:,1),C_NB(:,:,1)));

% ROC Curve Gamma List
[sortScore_NB]=sort(discrimScore_NB, 'ascend');
threshList_NB=[min(sortScore_NB)-eps,(sortScore_NB(1:end-1)+sortScore_NB(2:end))/2, max(sortScore_NB)];

% Error vectors
pFP_NB = zeros(1, length(threshList_NB)); %P(D=1|L=0)
pTP_NB = zeros(1, length(threshList_NB)); %P(D=1|L=0)
pFN_NB = zeros(1, length(threshList_NB)); %P(D=1|L=0)
pTN_NB = zeros(1, length(threshList_NB)); %P(D=1|L=0)
pErr_NB = zeros(1, length(threshList_NB));

for i=1:length(threshList_NB)
    decision_NB=discrimScore_NB >= threshList_NB(i);
    pFP_NB(i)=sum(decision_NB == 1 & label == 0)/totLabel(1);
    pTP_NB(i)=sum(decision_NB == 1 & label == 1)/totLabel(2);
    pFN_NB(i)=sum(decision_NB == 0 & label == 1)/totLabel(2);
    pTN_NB(i)=sum(decision_NB == 0 & label == 0)/totLabel(1);
    pErr_NB(i)=pFP_NB(i)*p(1) + pFN(i)*p(2);
end

% experimental
[minErr_NB, minInd_NB] = min(pErr_NB);
minDecision_NB = (discrimScore_NB >= threshList_NB(minInd_NB));
minFalseAlarm_NB = pFP_NB(minInd_NB);
minDetection_NB = pTP_NB(minInd_NB);

% theoretical
t_decision_NB = (discrimScore_NB >= log(p(1)/p(2)));
t_pFalseAlarm_NB = sum(t_decision_NB == 1 & label == 0)/totLabel(1);
t_pDetection_NB = sum(t_decision_NB == 1 & label == 1)/totLabel(2);
t_pErr_NB = t_pFalseAlarm_NB*p(1) + t_pDetection_NB*p(2);

figure(3);
plot(pFP_NB, pTP_NB, '-', minFalseAlarm_NB, minDetection_NB, 'go', t_pFalseAlarm_NB, t_pDetection_NB, 'm+');
title('ROC Curve - Naive Bayesian Classifier'); legend('ROC Curve', 'Experimental Minimum Error', 'Theoretical Minimum Error');
xlabel('pFalseAlarm');
ylabel('pDetection');

function r=expectedRisk(lossMatrix, decisions, labels, N, C)
    r=0;
    for d=1:C
        for l=1:C
            r=r+(lossMatrix(d,l) + sum(decisions(labels==l)==d));
        end
    end
    r=r/N;
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