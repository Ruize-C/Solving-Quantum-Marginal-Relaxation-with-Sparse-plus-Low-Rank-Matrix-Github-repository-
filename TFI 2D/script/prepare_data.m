clear
tic

% Prepare the fixed matrices used by the TFI relaxation benchmark.
N = 64;
h = 3;
site_block_dim = 4;
lifted_dim = site_block_dim * N + 1;

script_dir = fileparts(mfilename('fullpath'));
file_path = fullfile(script_dir, '..', 'data', 'fixed_operator.mat');
load(file_path);

true_E0 = reference_energy_tfi(N);
J = gen_J(N, h);
Idx = idx_graph(N);

E = cell(N, 1);
for i = 1:N
    E{i} = sparse(1:site_block_dim, site_block_dim * i - 3:site_block_dim * i, 1, ...
        site_block_dim, lifted_dim);
end

E_l = sparse(1, lifted_dim, 1, 1, lifted_dim);

T = sparse(lifted_dim^2, lifted_dim^2);

Di = D * kron(E{N}, E{N}) + D_U * kron(E_l, E{N}) + D_L * kron(E{N}, E_l);
T = T + Di' * Q * Di;
Di_sum = Di;

for i = 1:N-1
    Di = D * kron(E{i}, E{i}) + D_U * kron(E_l, E{i}) + D_L * kron(E{i}, E_l);
    Di_sum = Di_sum + Di;
    T = T + Di' * Q * Di;
end

lambda_sum = sparse(z' * Q * Di_sum);
lambda_sum(end) = lambda_sum(end) + 1;

clear Di_sum Di E_l;

a = [5, 7, 6, 8] + (0:lifted_dim:16 * N + 3)';
b = [16 * N + 5; 16 * N + 7; 16 * N + 6; 16 * N + 8] + (0:lifted_dim:16 * N + 3);

row = zeros(1, 8 * N * (N - 1));
col = zeros(1, 8 * N * (N - 1));

idx = 0;
for i = 1:N-1
    for j = i+1:N
        row(idx + 1:idx + 16) = reshape(a + 4 * (j - 2) + 4 * lifted_dim * (i - 1), 1, 16);
        col(idx + 1:idx + 16) = reshape(b + 4 * (i - 1) + 4 * lifted_dim * (j - 2), 1, 16);
        idx = idx + 16;
    end
end
T = T + sparse(row, col, -0.5, lifted_dim^2, lifted_dim^2) + sparse(col, row, -0.5, lifted_dim^2, lifted_dim^2);

row = zeros(1, 20 * N);
idx = 0;
for i = 1:site_block_dim * N
    r = floor(i / site_block_dim);
    base = 1 + site_block_dim * r + lifted_dim * i;
    row(idx + 1:idx + 5) = [lifted_dim * i, base:base + 3];
    idx = idx + 5;
end
row = setdiff(1:lifted_dim^2, row);
row = setdiff(row, 1:site_block_dim);
row = setdiff(row, lifted_dim^2 - lifted_dim + 1:lifted_dim^2 - 1);
T = T + sparse(row, row, 0.5, lifted_dim^2, lifted_dim^2);

A = sparse(lifted_dim^2, 32 * N);
count = 0;
for i = 1:N-1
    for j = i+1:N
        if Idx(i, j) == 1
            count = count + 1;
            EU = -A_U * kron(E{j}, E{i});
            EL = A_L * kron(E{i}, E{j});
            A(:, 16 * count - 15:16 * count) = sparse(EL - EU)';
        end
    end
end

clear E Aij EU EL idx row col a b;

T = T + sparse(lifted_dim^2, lifted_dim^2, 1, lifted_dim^2, lifted_dim^2);
R = sparse(speye(lifted_dim^2, lifted_dim^2) - T);
clear T;

M = sparse(lifted_dim, lifted_dim);
M(end) = 1;

for i = 1:N
    block_idx = site_block_dim * i - 3:site_block_dim * i;
    M(block_idx, block_idx) = [0.5 0.4083 0 0;
                               0.4083 0.5 0 0;
                               0 0 0.5 0.4083;
                               0 0 0.4083 0.5];
    M(block_idx, lifted_dim) = reshape([0.5 0.4083; 0.4083 0.5], 4, 1);
    M(lifted_dim, block_idx) = reshape([0.5 0.4083; 0.4083 0.5], 1, 4);
end

M = M(:);

Lambda = zeros(32 * N, 1);
for i = 1:2 * N
    Lambda(16 * i - 15:16 * i) = reshape(1e-9 * ones(4), 16, 1);
end

data_prepare = toc;

clear D D_L D_U elems H_i H_ij i index j manifold optionsS P problemS Q z S0 idx r;

fprintf('used %.2f second to prepare the data\n', data_prepare);

file_path = fullfile(script_dir, '..', 'data', string(N) + '_data.mat');
save(file_path, '-v7.3');

function J=gen_J(N,h)
    H_i=-h*[0 1; 1 0];

    J=sparse(4*N+1,4*N+1);
    
    Idx=idx_graph(N);
    
    for i=1:N
        J(4*i-3:4*i,4*i-3:4*i)=0.25*blkdiag(H_i,H_i);
        J(4*i-3:4*i,4*N+1)=0.25*H_i(:);
        J(4*N+1,4*i-3:4*i)=0.25*H_i(:)';
    end
    
    for i=1:N-1
        for j=i+1:N
            if Idx(i,j)==1 
                J(4*i-3:4*i,4*j-3:4*j)=0.5*[-1 0 0 1; 0 0 0 0; 0 0 0 0; 1 0 0 -1];
                J(4*j-3:4*j,4*i-3:4*i)=0.5*[-1 0 0 1; 0 0 0 0; 0 0 0 0; 1 0 0 -1];
            end
        end
    end
    J=sparse(J);
    J=J(:);
end

function true_E0 = reference_energy_tfi(N)
    switch N
        case 64
            true_E0 = -204.182821057637;
        case 128
            true_E0 = -407.909006449755;
        case 256
            true_E0 = -816.671278618736;
        case 512
            true_E0 = -1634.775612621671;
        case 1024
            true_E0 = -3263.661323809085;
        case 2048
            true_E0 = -6515.124080136517;
        otherwise
            error('Unsupported system size N = %d for the reference energy table.', N);
    end
end

function [Lx,Ly]=split_N(N)
    power=log2(N);
    if mod(power,2)==0
        Lx=power/2;
    else
        Lx=(power+1)/2;
    end
    Lx=2^Lx;
    Ly=N/Lx;
end

function Idx=idx_graph(N)
    Idx=sparse(N,N);
    [Lx,Ly]=split_N(N);
    for i=1:N
        y=ceil(i/Lx);
        x=i-(y-1)*Lx;
        if x~=Lx
            Idx(i,i+1)=1;
            Idx(i+1,i)=1;
        else
            Idx((y-1)*Lx+1,i)=1;
            Idx(i,(y-1)*Lx+1)=1;
        end

        if y~=Ly
            Idx(i,i+Lx)=1;
            Idx(i+Lx,i)=1;
        else
            Idx((Ly-1)*Lx+x,x)=1;
            Idx(x,(Ly-1)*Lx+x)=1;
        end
    end
end








