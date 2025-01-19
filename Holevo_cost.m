function f = Holevo_cost(X, NK, num_p)
% Input:
% X: product manifold with pi and psi_i
% NK: A cell array containing the Kraus operators {K_i}_{i=1}^k
% num_p: The number of p_i
% Output:
% Holevo cost function w.r.t. NK

p = X.p;
psi = X.psi;
rho = 0;
for j=1:num_p
    rho = rho + p(:,:,j) * psi(:,:,j)*psi(:,:,j)';
end
% N(rho)
sK = size(NK);
if(sK(2) == 1 || (sK(1) == 1 && sK(2) > 2)) % map is CP
    NK = NK(:);
    NK(:,2) = cellfun(@ctranspose,NK(:,1),'UniformOutput',false);
else
    NK(:,2) = cellfun(@ctranspose,NK(:,2),'UniformOutput',false);
end
Nrho = cell2mat(NK(:,1).')*kron(speye(size(NK,1)), rho) * cell2mat(NK(:,2)); % apply kraus on rho

% compute Holevo cost function
f = quantum_entr(Nrho);
for j=1:num_p
    psiin = psi(:,:,j)*psi(:,:,j)';
    % Npsi = 0;
    % for i=1:length(NK)
    %     Npsi = Npsi + NK{i}*psiin*NK{i}';
    % end
    Npsi = cell2mat(NK(:,1).')*kron(speye(size(NK,1)), psiin) * cell2mat(NK(:,2)); % apply kraus on psi
    f = f - p(:,:,j) * quantum_entr(Npsi);
end
f = -f/log(2);
end
