function [  ] = NCL
% Managing diversity in regression ensembles
% Journal of machine learning research 6.Sep (2005): 1621-1650.
% Brown, Gavin, Jeremy L. Wyatt, and Peter Ti?o. 


[x,t] = house_dataset();

x = mapminmax(x,0,1);
t = mapminmax(t,0,1);

[tr,va,ts] = dividerand(size(x,1),0.5,0,0.5);

data.xTrain = x(tr,:);
data.tTrain = t(tr,:);
data.xTest = x(ts,:);
data.tTest = t(ts,:);

data.nTrain = length(tr);
data.nTest = length(ts);
data.nInput = size(x,2);
data.nOutput = size(t,2);

MaxIteration = 100; % maximum number of iterations
nEnsemble = 5;  % ensemble size
nHidden = 20;   % number of hidden nodes

lambda = 0.5;
eta = 0.1;

T = tic;
model = trainingFcn(data,nEnsemble,nHidden,MaxIteration,lambda,eta);
toc(T)

sqrt(mse(predictor(data.xTest,model) - data.tTest))
end

function model = trainingFcn(data,nEnsemble,nHidden,MaxIteration,lambda,eta)
model = struct('base',struct('net',[]),'nEnsemble',[],'lambda',[]);
model.nEnsemble = nEnsemble;
model.MaxIteration = MaxIteration;
model.lambda = lambda;
for i = 1:model.nEnsemble
    model.base(i).net = initial(data,nHidden);
end

Curve = inf(1,model.MaxIteration);
for iter = 1:model.MaxIteration
    fBar = predictor(data.xTrain,model);
    for j = 1:model.nEnsemble
        penalty = -lambda * (forward(data.xTrain,model.base(j).net) - fBar);
        model.base(j).net = backward(data.xTrain, data.tTrain, model.base(j).net, penalty, eta);
    end
    
    Curve(iter) = sqrt(mse(predictor(data.xTrain,model) - data.tTrain));
end
plot(Curve)

end

function output = predictor(X,model)
Y = zeros(size(X,1),model.nEnsemble);
for i = 1:model.nEnsemble
    Y(:,i) = forward(X,model.base(i).net);
end
output = mean(Y,2);
end

function net = initial(data,nHidden)
net = struct('InputWeight',[],'HiddenBias',[],'OutputWeight',[],'OutputBias',[]);
RND = @(R,C)rands(R,C)./10;
net.InputWeight = RND(data.nInput,nHidden);
net.HiddenBias = RND(1,nHidden);
net.OutputWeight = RND(nHidden,data.nOutput);
net.OutputBias = RND(1,data.nOutput);
net.nHidden = nHidden;
end

function net = backward(X,T,net,penalty,eta)
[OutputLayer,HiddenLayer] = forward(X,net);

NC = OutputLayer-T + penalty;

delta_OutputWeight  = OutputLayer.*(1-OutputLayer)'*HiddenLayer;
delta_OutputBias    = OutputLayer.*(1-OutputLayer);
delta_InputWeight   = net.OutputWeight.*(HiddenLayer.*(1-HiddenLayer))'*X;
delta_HiddenBias    = net.OutputWeight.*(HiddenLayer.*(1-HiddenLayer))';

net.OutputWeight    = net.OutputWeight  -eta.*(delta_OutputWeight'*NC)';
net.OutputBias      = net.OutputBias    -eta.*(delta_OutputBias'*NC)';
net.InputWeight     = net.InputWeight   -eta.*NC*delta_InputWeight';
net.HiddenBias      = net.HiddenBias    -eta.*NC*delta_HiddenBias';

end

function [OutputLayer,HiddenLayer] = forward(X,net)
[N,~] = size(X);
HiddenLayer = logsig(X*net.InputWeight+repmat(net.HiddenBias,N,1));
OutputLayer = logsig(HiddenLayer*net.OutputWeight+repmat(net.OutputBias,N,1));
end