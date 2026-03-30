clear

% Initialize the low-rank plus banded starting point used by `alg4.m`.
% Run this script after `prepare_data.m`.
N = 8 * 8;
r = 200;
band_width = 32;
site_block_dim = 16;
lifted_dim = site_block_dim * N + 1;

script_dir = fileparts(mfilename('fullpath'));
file_path = fullfile(script_dir, '..', 'data', string(N) + '_data.mat');
load(file_path);

U = gen_U(N, site_block_dim);
B = build_B_periodic(site_block_dim * N, 0:band_width);
V = U * B;

S0 = reshape(J, lifted_dim, lifted_dim) + speye(lifted_dim);
S0 = S0(:);
c = norm(S0)^2;

elems.y = euclideanfactory(size(V, 2));
elems.t = euclideanfactory(lifted_dim, r);
manifold = productmanifold(elems);
problem.M = manifold;

problem.cost = @(yt) norm(S0 - construct_H(V, yt.y, yt.t, lifted_dim))^2 / c;
problem.egrad = @(yt) dLoss(yt.y, yt.t, V, S0, lifted_dim, r);

options.maxiter = 100;
[yt, ~] = rlbfgs(problem, [], options);

y = yt.y;
t = yt.t;

file_path = fullfile(script_dir, '..', 'data', string(N) + '_S0.mat');
save(file_path, 'y', 't', 'band_width', 'V', 'r', '-v7.3');

function val = construct_H(V, y, t, lifted_dim)
    X = reshape(V * y, lifted_dim, lifted_dim);
    H = X' * X + t * t';
    val = H(:);
end

function val = dLoss(y, t, V, X0, lifted_dim, r)
    dLdX = reshape(X0 - construct_H(V, y, t, lifted_dim), lifted_dim, lifted_dim);
    X = reshape(V * y, lifted_dim, lifted_dim);
    v = X * dLdX;

    dLdy = -2 * reshape(v(:)' * V, size(V, 2), 1);
    dLdt = -2 * reshape(dLdX * t, lifted_dim, r);

    val = struct('y', dLdy, 't', dLdt);
end

function B = build_B_periodic(n, offsets)
    % Build a banded circulant basis with the requested diagonal offsets.
    num_offsets = numel(offsets);

    S = spdiags(ones(n, 1), 1, n, n);
    S(n, 1) = 1;

    D = sparse(n^2, n);
    for i = 1:n
        D((i - 1) * n + i, i) = 1;
    end

    B = sparse(n^2, num_offsets * n);
    I = speye(n);
    for k = 1:num_offsets
        shift = mod(offsets(k), n);
        Sk = S^shift;
        Bk = kron(Sk.', I) * D;
        cols = (k - 1) * n + (1:n);
        B(:, cols) = Bk;
    end
end

function U = gen_U(N, site_block_dim)
    % Lift an n x n matrix to the affine `(n + 1) x (n + 1)` embedding.
    n = site_block_dim * N;
    S = [speye(n); sparse(1, n)];
    U = kron(S, S);
end
