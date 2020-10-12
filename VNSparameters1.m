% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017

vnsParameters.I_NP= 10; % population in DE
vnsParameters.F_weight= 0.3; %Mutation factor
vnsParameters.F_CR= 0.5; %Recombination constant
vnsParameters.I_itermax= 50; % number of max iterations/gen
vnsParameters.I_strategy   = 1; %DE strategy
%deParameters.Scenarios = 14;
vnsParameters.I_bnd_constr = 1; %Using bound constraints 
% 1 repair to the lower or upper violated bound 
% 2 rand value in the allowed range
% 3 bounce back

%
