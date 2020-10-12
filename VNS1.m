%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function:         [FVr_bestmem,S_bestval,I_nfeval] = deopt(fname,S_struct)
% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017

function [Fit_and_p,FVr_bestmemit, fitMaxVector] = ...
    VNS(deParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)


%-----This is just for notational convenience and to keep the code uncluttered.--------
I_NP         = deParameters.I_NP;
%F_weight     = deParameters.F_weight;
%F_CR         = deParameters.F_CR;
I_D          = numel(up_habitat_limit); %Number of variables or dimension
deParameters.nVariables=I_D;
FVr_minbound = low_habitat_limit;
FVr_maxbound = up_habitat_limit;
I_itermax    = deParameters.I_itermax;

%Repair boundary method employed
BRM=deParameters.I_bnd_constr; %1: bring the value to bound violated
                               %2: repair in the allowed range

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_strategy   = deParameters.I_strategy; %important variable
fnc= otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-----Check input variables---------------------------------------------
%if (I_NP < 5)
%   I_NP=5;
  % fprintf(1,' I_NP increased to minimal value 5\n');
%end
%if ((F_CR < 0) || (F_CR > 1))
 %  F_CR=0.5;
  % fprintf(1,'F_CR should be from interval [0,1]; set to default value 0.5\n');
%end
if (I_itermax <= 0)
   I_itermax = 200;
   fprintf(1,'I_itermax should be > 0; set to default value 200\n');
end



%-----Initialize population and some arrays-------------------------------
%FM_pop = zeros(I_NP,I_D); %initialize FM_pop to gain speed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-allocation of loop variables
%fitMaxVector = nan(1,I_itermax);
%objMaxVector = nan(1,I_itermax);
Niter= I_itermax;
iter = 0;
% limit iterations by threshold
fitMaxVector = nan(1,I_itermax);
gen = 1; %iterations

%Niter= 74;
%fitMaxVector = nan(1,Niter+1);
%objMaxVector = nan(2,Niter+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----FM_pop is a matrix of size I_NPx(I_D+1). It will be initialized------
%----with random values between the min and max values of the-------------
%----parameters-----------------------------------------------------------
% FLC modification - vectorization
minPositionsMatrix=repmat(FVr_minbound,I_NP,1);
maxPositionsMatrix=repmat(FVr_maxbound,I_NP,1);
deParameters.minPositionsMatrix=minPositionsMatrix;
deParameters.maxPositionsMatrix=maxPositionsMatrix;

% generate initial population.
%FM_pop=genpop(I_NP,I_D,minPositionsMatrix,maxPositionsMatrix);

%% Random Initial Solution
%X = zeros(1,I_D);
%for i = 1:I_D
%    X(i) = low_habitat_limit(i) + rand*(up_habitat_limit(i) - low_habitat_limit(i));
%end
rand('state',otherParameters.iRuns) %Guarantee same initial population
X = unifrnd(minPositionsMatrix,maxPositionsMatrix,I_NP,I_D); 
%for i = 1:I_D
   % if X(i) < minPositionsMatrix(i)
      %  X(i) = minPositionsMatrix(i);
   % elseif X(i) > maxPositionsMatrix(i)
   %     X(i) = maxPositionsMatrix(i);  
   % end
%end

nEvals=0;
%FM_pop = Neighborghood(1,I_D,minPositionsMatrix,maxPositionsMatrix,X);
%I_D
%FM_pop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------Evaluate the best member after initialization----------------------
svalue = 0;
if gen < Niter
   % otherParameters.No_eval_Scenarios=deParameters.Scenarios;
    %FM_ui=update1(FM_pop,minPositionsMatrix,maxPositionsMatrix,BRM);
    [S_val, ~]=feval(fnc,X,caseStudyData,otherParameters);

     xrep=X;  
       
end

xbest = xrep;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The user should decide which is the criterion to optimize. 
% In this example, we optimize worse performance
%% Worse performance criterion
[S_bestval,I_best_index] = min(S_val); % This mean that the best individual correspond to the best worst performance
FVr_bestmemit = X(I_best_index,:); % best member of current iteration
fitMaxVector(1,gen) = S_bestval;
sbest = S_bestval;
%gen=gen+1;
% The user can decide to save the mean, best, or any other value here

%------DE-Minimization---------------------------------------------
%------FM_popold is the population which has to compete. It is--------
%------static through one iteration. FM_pop is the newly--------------
%------emerging population.----------------------------------------

maxK = 5;

while gen<=I_itermax %%&&  fitIterationGap >= threshold
  % svalueT = svalue; 
  k = 1;
   while k<=maxK
     X1 = Neighborghood(k,I_NP,I_D,minPositionsMatrix,maxPositionsMatrix,xbest(I_best_index,:));
    % k
  %  X1 = update_eda(X1,minPositionsMatrix,maxPositionsMatrix,BRM);
    %Evaluation of new Pop
    [S_val_temp, ~]=feval(fnc,X1,caseStudyData, otherParameters);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% update best results
    [S_bestval,I_best1_index] = min(S_val_temp);
   
    % store fitness evolution and obj fun evolution as well
      if  (S_bestval) < (sbest)
        sbest = S_bestval;
       % k = 1;   
        I_best_index = I_best1_index;   
        X=X1;
        xrep=X;  
      xbest = xrep; 
      FVr_bestmemit = xbest(I_best_index,:);
      break;
     else
       k = k + 1;  
     end    
    
   end
           
   
 
  fprintf('Fitness value: %f\n',fitMaxVector(1,gen) )
  fprintf('Generation: %d\n',gen)
    
  gen=gen+1;
  fitMaxVector(1,gen) =  sbest;
end %---end while ((I_iter < I_itermax) ...

Fit_and_p=[fitMaxVector(1,gen) 0]; %;p2;p3;p4]


function pop=Neighborghood(K,I_NP,I_D,lowMatrix,upMatrix,Xt)
pop_u(1,:)= Xt(1,:);
for k = 2:K+1
 %X = zeros(1,I_D);
%for i = 1:I_D
 %   X(i) = lowMatrix(i) + rand*(upMatrix(i) - lowMatrix(i));
%end
X = unifrnd(lowMatrix,upMatrix); 
%for i = 1:I_D
  %  if X(i) < lowMatrix(i)
   %     X(i) = lowMatrix(i);
   % elseif X(i) > upMatrix(i)
    %    X(i) = upMatrix(i);  
  %  end
%end   
% 
 pop_u(k,:) =  X(1,:);
end 
 mu = mean(pop_u);
 sd = std(pop_u);
 for j=1:I_NP
 Np = normrnd(mu,sd).*(mu - sd*tan(pi*(rand(1,1))-0.5));
  for i = 1:I_D
    if Np(i) < lowMatrix(i)
      Np(i) = lowMatrix(i);
   elseif Np(i) > upMatrix(i)
      Np(i) = upMatrix(i);  
   end
  end 
  pop(j,:)=Np;
 end
% Np = mean(Nk)+std(Nk);
 
function pop_eda = learningEDA(pop_u,I_NP)
 %Cauchy's distribution
 %var_means = mean(pop_u);
 %var_sigma = std(pop_u);
 %m = length(var_means);
 %pop_eda = sin(pop_u)-exp(pop_u);%28.864
 mu = mean(pop_u);
 sd = std(pop_u);
for i=1:I_NP
 %pop_eda(i,:) = -normrnd(mu,sd).*exp(pop_u(i,:));
 %pop_eda(i,:) = - lognrnd(mu,sd).*exp(pop_u(i,:));
 %pop_eda(i,:) = - chi2rnd(2)*exp(pop_u(i,:));
 pop_eda(i,:) = -normrnd(mu,sd).*(mu - sd*tan(pi*(rand(1,1))-0.5));
 %pop_eda(i,:) = - lognrnd(mu,sd).*(mu - sd*tan(pi*(rand(1,1))-0.5));
 %pop_eda(i,:) = chi2rnd(2)*(mu - sd*tan(pi*(rand(1,1))-0.5));
end
 %pop_eda = cos(pop_u)-exp(pop_u);31.65
 %pop_eda = log(pop_u);
 %sigma = std(pop_u);
 %mu = mean(pop_u);
 %sd = std(pop_u);
 %m = length(sd);
 %mu = mean(pop_u);
 %fu = sin(mu+sd)-exp(mu+sd);
 %fu = -exp(mu+sd);
 %va = var(pop_u);
 %m = length(mu);
 %xmin = min(pop_u);
 %xmax = max(pop_u);
 %Gaussian multiva
 %vars_cov = cov(sin(pop_u)-exp(pop_u));%34.057  
 %for i=1:I_NP
  %pop_eda(i,:) = vars_cov(i,:);
  %pop_eda(i,:) = var_means - var_sigma*tan(pi*(rand(m,1))-0.5); %estimation of distribution of cauchy--- 24.862
  %pop_eda(i,:) = fu*rand(1,1); %24.262
  %pop_eda(i,:)= fu*rand(1,1);%24.193
 %end
 % VECTORIZED THE CODE INSTEAD OF USING FOR
 function p=update_eda(p,lowMatrix,upMatrix,BRM)
  switch BRM
    case 1 % max and min replace
        [idx] = find(p<lowMatrix);
        p(idx)=lowMatrix(idx);
        [idx] = find(p>upMatrix);
        p(idx)=upMatrix(idx);
    case 2 %Random reinitialization
        [idx] = [find(p<lowMatrix);find(p>upMatrix)];
        replace=unifrnd(lowMatrix(idx),upMatrix(idx),length(idx),1);
        p(idx)=replace;
    case 3 %Bounce Back
        [idx] = find(p<lowMatrix);
      p(idx)=unifrnd(lowMatrix(idx),p(idx),length(idx),1);
        [idx] = find(p>upMatrix);
      p(idx)=unifrnd(p(idx), upMatrix(idx),length(idx),1);
   end



 
