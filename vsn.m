clc;
clear;
close all;

%% Problem Definition

model=SelectModel();        % Select Model of the Problem

model.eta=0.1;

CostFunction=@(q) MyCost(q,model);       % Cost Function

%% EDA Parameters

MaxIt=1200;     % Maximum Number of Iterations
MaxPop = 10000;    % Maximum population individuals
numParents=3000;
bestCost = inf;
maxK = 10;
%% Initialization

% Create Initial Solution
x.Position=CreateRandomSolution(model);
[x.Cost, x.Sol]=CostFunction(x.Position);

% Update Best Solution Ever Found
BestSol=x;

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

for it=1:MaxIt  
 k = 1;
 while k<=maxK
        xnew.Position=CreateNeighbor(x.Position);
        [xnew.Cost, xnew.Sol]=CostFunction(xnew.Position);
        
        if xnew.Cost<=x.Cost
            % xnew is better, so it is accepted
            x=xnew; 
            break;
        else
          k = k + 1;  
        end      
 end

%Update solutions
 if x.Cost<=BestSol.Cost
    BestSol=x;
 end

  % Store Best Cost
    BestCost(it)=BestSol.Cost;

% Display Iteration Information
    if BestSol.Sol.IsFeasible
        FLAG=' *';
    else
        FLAG='';
    end
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it)) FLAG]);
  
  % Plot Solution
    figure(1);
    PlotSolution(BestSol.Sol,model);
    pause(0.01);
end


%% Results

figure;
plot(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;