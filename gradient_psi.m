function grad = gradient_p(X, K, num_p)
% Input:
% X: product manifold with pi and psi_i
% K: A cell array containing the Kraus operators {K_i}_{i=1}^k
% num_p: The number of p_i
% Output:
% Euclidean gradient of p

% linear combination of N and fully depolarizing channel if Kraus rank < d
delta = 1e-9;
sK = size(K);
ss = size(K{1});
d = ss(1); % dimension of the input state
Kdepo = chanconv(eye(d^2)/d,'choi','kraus',[d d]);
if length(K)<d % map is not full rank
    for i = 1:length(K)
        K{i} = K{i} * sqrt(1-delta);
    end
    for j = 1:length(Kdepo)
        Kdepo{j} = Kdepo{j} * sqrt(delta);
    end
    K = [K;Kdepo]; % new Kraus with full-rank output
else
end

% get probability and input state
p = X.p;
psi = X.psi;
% sum of pi psi_i
rho = 0;
for j=1:num_p
    rho = rho + p(:,:,j) * psi(:,:,j)*psi(:,:,j)';
end
% N(rho)
sK = size(K);
if(sK(2) == 1 || (sK(1) == 1 && sK(2) > 2)) % map is CP
    K = K(:);
    K(:,2) = cellfun(@ctranspose,K(:,1),'UniformOutput',false);
else
    K(:,2) = cellfun(@ctranspose,K(:,2),'UniformOutput',false);
end
Nrho = cell2mat(K(:,1).')*kron(speye(size(K,1)), rho) * cell2mat(K(:,2)); % apply kraus on rho

for I=1:num_p
    NpsiI = cell2mat(K(:,1).')*kron(speye(size(K,1)), psi(:,:,I)*psi(:,:,I)') * cell2mat(K(:,2)); % apply kraus on rho
    t = logm(Nrho) - logm(NpsiI);
    t = t/log(2);
    Q1 = 0;
    for i=1:length(K)
        Q1 = Q1 + K{i}' * t * K{i};
    end
    grad(:,:,I) = 2 .* p(:,:,I) .* Q1 * psi(:,:,I);
end
end
