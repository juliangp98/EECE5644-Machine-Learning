% Julian Perez %
% EECE5644 Assignment 4 %
% Question 1 %

% MLP Curve
clear all;
close all;
dTypes={'train'; 'test'};
D.train.N=1e3;
D.test.N=10e3;

% Generate Data
figure;
for ind=1:length(dTypes)
    [D.(dTypes{ind}).x,D.(dTypes{ind}).y] = hw4q1dataGen(D.(dTypes{ind}).N);
    
    % Plot validation data
    subplot(2,1,ind);
    plot(D.(dTypes{ind}).x,D.(dTypes{ind}).y,'.');
    xlabel('x');ylabel('y'); grid on;
    xlim([0 30]); ylim([0 15]);
    title([dTypes{ind}]);
end

% Train and Validate Data
nPerceptrons=15; % Max number of perceptrons
k=10; % fold count - kfold validation
[D.train.net,D.train.MSEtrain,optM,stats]=kfoldMLP_Fit(nPerceptrons,k,D.train.x, D.train.y);

% Produce validation data
yVal=D.train.net(D.test.x);

% Calculate MSE
MSE=mean((yVal-D.test.y).^2);

% Plot results
fprintf('MSE=%1.2f%\n',MSE);
figure;
plot(D.test.x,D.test.y,'o','DisplayName','Test Data');
hold all;

plot(D.test.x,yVal,'.','DisplayName','Estimated Data');
xlabel('x');ylabel('y');grid on;
title('Actual and Estimated Data');
legend 'show';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [outputNet,outputMSE, optM, stat]=kfoldMLP_Fit(numPerceptron,k,x,y)
N=length(x);
numIterations=10;

% Set up cross-validation on training data
partSize=N/k;
partIndex=[1:partSize:N length(x)];

% Perform cross-validation on certain perceptrons
for M=1:numPerceptron
    figure;
    for i=1:k
        % Separate training and validation data
        index.val=partIndex(i):partIndex(i+1)-1;
        index.train=setdiff(1:N,index.val);
        % Create object with M perceptrons in hidden layer
        net=feedforwardnet(M);
        % Train using training data
        net=train(net,x(:,index.train),y(:,index.train));
        % Validate with remaining data
        yVal=net(x(:,index.val));
        % MSE for evaluation
        MSE(i)=mean((yVal-y(:,index.val)).^2);
        % Plot overlay of model and validation data
        subplot(5,2,i);
        plot(x(:,index.val),y(:,index.val),'o')
        hold all;
        plot(x(:,index.val),yVal,'.')
        if i==1
            title([num2str(M) ' Perceptrons']);
        end
    end
    
    % Determine average probability of error for each number of perceptrons
    avgMSE(M)=mean(MSE);
    stat.M=1:M;
    stat.avgMSE=avgMSE;
    stat.mMSE(:,M)=MSE;
end

% Determine optimal number of perceptrons
[~,optM]=min(avgMSE);

% Train all the data again
for i=1:numIterations
    netName(i)={['net' num2str(i)]};
    finalnet.(netName{i})=patternnet(optM);
    finalnet.(netName{i})=train(net,x,y);
    yVal=finalnet.(netName{i})(x);
    MSEFinal(i)=mean((yVal-y).^2);
end

% Set data to be output for eval purposes
[minMSE,outInd]=min(MSEFinal);
stat.finalMSE=MSEFinal;
outputMSE=minMSE;
outputNet=finalnet.(netName{outInd});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Other function to generate data
% function [x,y] = generateData(N)
% close all,
% x = gamrnd(3,2,1,N);
% z = exp((x.^2).*exp(-x/2));
% v = lognrnd(0,0.1,1,N);
% y = v.*z;
% figure(1), plot(x,y,'.'),
% xlabel('x'); ylabel('y');
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%