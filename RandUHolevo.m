function h = RandUHolevo(NK)
s = size(NK{1});
d = s(1);

%% optimization
num_p = d^2; % number of pure state

elem.p = sympositivedefinitesimplexfactory(1,num_p); % probability
elem.psi = stiefelcomplexfactory(d, 1, num_p); % pure state ensemble

problem.M = productmanifold(elem);
problem.cost = @(X) Holevo_cost(X, NK, num_p);
problem.egrad = @(X) struct('p', gradient_p(X, NK, num_p), 'psi', gradient_psi(X, NK, num_p)); % gradient
options.tolgradnorm = 1e-6;
options.maxiter = 100000;

% choose optimization method
[Xopt, f, info] = barzilaiborwein(problem, [], options);
% [Xopt, f, info] = trustregions(problem, [], options);

h = -f;
end


