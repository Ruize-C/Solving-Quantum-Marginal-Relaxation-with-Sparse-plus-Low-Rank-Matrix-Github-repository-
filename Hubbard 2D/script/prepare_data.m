clear

% Prepare the fixed matrices used by the 2D Hubbard relaxation benchmark.
N = 8 * 8;
t = 1;
U = 2;
site_block_dim = 16;
lifted_dim = site_block_dim * N + 1;

script_dir = fileparts(mfilename('fullpath'));
file_path = fullfile(script_dir, '..', 'data', 'fixed_operator.mat');
load(file_path);
file_path = fullfile(script_dir, '..', 'data', 'diff_idx.mat');
load(file_path);

tic
true_E0 = reference_energy_hubbard(N);

Idx=idx_graph(N);

J=gen_J(N,U,t,Idx,A_U);

E = cell(N, 1);
for i = 1:N
    E{i} = sparse(1:site_block_dim, site_block_dim * i - 15:site_block_dim * i, 1, ...
        site_block_dim, lifted_dim);
end

E_l = sparse(1, lifted_dim, 1, 1, lifted_dim);

A = sparse(lifted_dim^2, 256 * 2 * N);

count = 0;
for i = 1:N-1
    for j = i+1:N
        if Idx(i, j) == 1
            count = count + 1;
            EU = -A_U * kron(E{j}, E{i});
            EL = A_L * kron(E{i}, E{j});
            A(:, 256 * count - 255:256 * count) = sparse(EL - EU)';
        end
    end
end

T = sparse(lifted_dim^2, lifted_dim^2, 1, lifted_dim^2, lifted_dim^2);

Di = D * kron(E{N}, E{N}) + D_U * kron(E_l, E{N}) + D_L * kron(E{N}, E_l);
T = T + Di' * Q * Di;
zQ = z' * Q;
lambda_sum = sparse(zQ * Di);

for i = 1:N-1
    Di = D * kron(E{i}, E{i}) + D_U * kron(E_l, E{i}) + D_L * kron(E{i}, E_l);
    lambda_sum = lambda_sum + zQ * Di;
    T = T + Di' * Q * Di;
end

clear E EL EU;

lambda_sum(end) = lambda_sum(end) + 1;

Crow_idx = setdiff(1:256, row_idx);
Ccol_idx = setdiff(1:256, col_idx);

a = [17, 21, 25, 29, 18, 22, 26, 30, 19, 23, 27, 31, 20, 24, 28, 32] + (0:lifted_dim:256 * N + 15)';
b = 256 * N + [17; 21; 25; 29; 18; 22; 26; 30; 19; 23; 27; 31; 20; 24; 28; 32] + (0:lifted_dim:256 * N + 15);

rowp = zeros(1, 96 * N * (N - 1));
rowm = zeros(1, 32 * N * (N - 1));
colp = zeros(1, 96 * N * (N - 1));
colm = zeros(1, 32 * N * (N - 1));

idxp = 0;
idxm = 0;
for i = 1:N-1
    for j = i+1:N
        row = reshape(a + site_block_dim * (j - 2) + site_block_dim * lifted_dim * (i - 1), 1, 256);
        rowp(idxp + 1:idxp + 192) = row(Crow_idx);
        rowm(idxm + 1:idxm + 64) = row(row_idx);
        col = reshape(b + site_block_dim * (i - 1) + site_block_dim * lifted_dim * (j - 2), 1, 256);
        colp(idxp + 1:idxp + 192) = col(Ccol_idx);
        colm(idxm + 1:idxm + 64) = col(col_idx);
        idxp = idxp + 192;
        idxm = idxm + 64;
    end
end
T = T + sparse(rowp, colp, -0.5, lifted_dim^2, lifted_dim^2) + sparse(rowm, colm, 0.5, lifted_dim^2, lifted_dim^2);
T = T + sparse(colp, rowp, -0.5, lifted_dim^2, lifted_dim^2) + sparse(colm, rowm, 0.5, lifted_dim^2, lifted_dim^2);

row = zeros(1, 272 * N);
idx = 0;
for i = 1:site_block_dim * N
    r = floor(i / site_block_dim);
    base = 1 + site_block_dim * r + lifted_dim * i;
    row(idx + 1:idx + 17) = [lifted_dim * i, base:base + 15];
    idx = idx + 17;
end
row = setdiff(1:lifted_dim^2, row);
row = setdiff(row, 1:site_block_dim);
row = setdiff(row, lifted_dim^2 - lifted_dim + 1:lifted_dim^2 - 1);
T = T + sparse(row, row, 0.5, lifted_dim^2, lifted_dim^2);

R = speye(size(T)) - sparse(T);
clear T;

data_prepare = toc;
fprintf('used %.2f second to prepare the data\n', data_prepare);

M = sparse(lifted_dim, lifted_dim);
M(end) = 1;

for i = 1:N
    block_idx = site_block_dim * i - 15:site_block_dim * i;
    M(block_idx, block_idx) = kron(eye(4), diag([0.1 0.4 0.4 0.1]));
    M(block_idx, lifted_dim) = [0.1 0 0 0 0 0.4 0 0 0 0 0.4 0 0 0 0 0.1]';
    M(lifted_dim, block_idx) = [0.1 0 0 0 0 0.4 0 0 0 0 0.4 0 0 0 0 0.1];
end

M = M(:);

Lambda = zeros(2 * N * 256, 1);

for i = 1:2 * N
    Lambda(256 * i - 255:256 * i, 1) = reshape(1e-9 * ones(16), 256, 1);
end

clear a A_L b band_width base col D D_L D_U Di Di_sum E_l h i idx j Lambda_block M_ii P Q r row sigma zQ

file_path = fullfile(script_dir, '..', 'data', string(N) + '_data.mat');
save(file_path, '-v7.3');


function J=gen_J(N,U,t,Idx,A_U)
I4  = speye(4);
cu  = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0];  
cud = cu.';                                   
cd  = [0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0]; 
cdd = cd.';                                   
nu  = cud*cu;                                 
nd  = cdd*cd;                                 
P   = diag([1 -1 -1 1]);                     

Hi=U*nu*nd;
J = sparse(16 * N + 1, 16 * N + 1);

J_left        = @(O) kron(O, I4);   
J_right_odd   = @(O) kron(P , O);   

Hij_up = J_left(cud) * J_right_odd(cu);  
Hij_dn = J_left(cdd) * J_right_odd(cd);  
Hij=-t*(Hij_up+Hij_up'+Hij_dn+Hij_dn');


for i=1:N
    J(16*i-15:16*i,16*i-15:16*i)=0.125*blkdiag(Hi,Hi,Hi,Hi);
    J(16*i-15:16*i,16*N+1)=0.25*Hi(:);
    J(16*N+1,16*i-15:16*i)=0.25*Hi(:)';
end

for i=1:N-1
    for j=i+1:N
        if Idx(i,j)==1
            J(16*i-15:16*i,16*j-15:16*j)=0.5*reshape(A_U'*Hij(:),16,16);
            J(16*j-15:16*j,16*i-15:16*i)=J(16*i-15:16*i,16*j-15:16*j)';
        end
    end
end
J=sparse(J);
J=J(:);
end

function true_E0 = reference_energy_hubbard(N)
    switch N
        case 16
            true_E0 = -18.02;
        case 32
            true_E0 = -36.48;
        case 64
            true_E0 = -74.483688237442;
        case 128
            true_E0 = -150.008938663942;
        case 256
            true_E0 = -300.369921167455;
        case 512
            true_E0 = -599.362039177223;
        case 1024
            true_E0 = -1203.167320077821;
        case 1600
            true_E0 = -1878.725;
        otherwise
            error('Unsupported system size N = %d for the reference energy table.', N);
    end
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
