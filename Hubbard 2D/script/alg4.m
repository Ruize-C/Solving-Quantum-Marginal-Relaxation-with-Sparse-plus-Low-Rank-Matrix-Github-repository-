clear

% Main benchmark driver for the 2D Hubbard relaxation.
% Recommended run order:
%   1) prepare_data.m
%   2) init_S0.m
%   3) alg4.m
N = 8 * 8;
site_block_dim = 16;
lifted_dim = site_block_dim * N + 1;

script_dir = fileparts(mfilename('fullpath'));
file_path = fullfile(script_dir, '..', 'data', string(N) + '_data.mat');
load(file_path);
file_path = fullfile(script_dir, '..', 'data', string(N) + '_S0.mat');
load(file_path);

sigma = 1;
sigma0 = sigma;
cD = 1 / (1 + norm(J, 'fro'));
mu = 5;
tau = 2;
maxiter = 200;
maxiter_opt = 20;
record = 1000;
alpha1 = 0.6;
alpha2 = 0.8;
alpha3 = 0.1;

Idx = idx_graph(N);

eta_P = zeros(maxiter, 1);
eta_D = zeros(maxiter, 1);
eta_g = zeros(maxiter, 1);
energy = zeros(maxiter, 1);
converge = zeros(maxiter, 1);

manifoldy = euclideanfactory(size(V, 2));
problemy.M = manifoldy;

manifold_t = euclideanfactory(lifted_dim, r);
problemt.M = manifold_t;

options.maxiter = maxiter_opt;

k = 1;

C = JmH(V, y, t, J, lifted_dim);
X = R * C - A * Lambda;
M_mat = reshape(M, lifted_dim, lifted_dim);

[eta_P(k), eta_D(k), eta_g(k), primal_obj, dual_obj] = evaluate_iteration_metrics( ...
    M, M_mat, X, C, J, cD, lambda_sum, N, A_U, Idx);

converge(k) = max([eta_P(k), eta_D(k), eta_g(k)]);
energy(k) = primal_obj;

time = 0;

tic
while converge(k) >= 1e-3 && k <= maxiter && time <= record
    converge0 = converge(k);
    l = 0;
    no_restart = 1;

    while no_restart && (converge(k) >= 1e-3 && k <= maxiter && time <= record)
        Lambda = local_projection(X, sigma, M, N, A_U, Lambda, Idx);

        [y, t] = get_yt(y, t, N, sigma, J, M, R, A, Lambda, lambda_sum, V, ...
            options, problemy, problemt, r, lifted_dim);

        C = JmH(V, y, t, J, lifted_dim);
        X = R * C - A * Lambda;
        M = M - sigma * X;

        k = k + 1;
        l = l + 1;

        M_mat = reshape(M, lifted_dim, lifted_dim);
        [eta_P(k), eta_D(k), eta_g(k), primal_obj, dual_obj] = evaluate_iteration_metrics( ...
            M, M_mat, X, C, J, cD, lambda_sum, N, A_U, Idx);

        % Preserve the original stopping metric used in the Hubbard script.
        converge(k) = max([eta_P(k), eta_g(k)]);
        energy(k) = primal_obj;
        time = toc;

        print_iteration_status(k, eta_P(k), eta_D(k), eta_g(k), time);

        if time >= record
            break
        end

        if converge(k) <= alpha1 * converge0
            no_restart = 0;
        elseif converge(k) <= alpha2 * converge0 && converge(k) > converge(k - 1)
            no_restart = 0;
        elseif l >= alpha3 * k
            no_restart = 0;
        end
    end

    rho = eta_P(k) / eta_D(k);
    if rho > mu
        sigma = sigma / tau;
    elseif rho < 1 / mu
        sigma = sigma * tau;
    end
end
time = toc;

relative_err = abs(true_E0 - primal_obj) / -true_E0;

fprintf('Time: %f\n', time);
fprintf('Iteration: %f\n', k);
fprintf('Estimated Ground Energy : %f\n', primal_obj);
fprintf('Relative Error: %f\n', relative_err);
fprintf('Primal Feasibility Measure: %f\n', log10(eta_P(k)));
fprintf('Dual Feasibility Measure: %f\n', log10(eta_D(k)));
fprintf('Duality Gap: %f\n', log10(eta_g(k)));

energy_per_site = zeros(maxiter, 1);
for i = 2:k
    energy_per_site(i) = abs(energy(i) - energy(i - 1)) / N;
end

close all;
figure;
hold on;
plot(log10(eta_P))
plot(log10(eta_D))
plot(log10(eta_g))
plot(log10(energy_per_site))
legend('\eta_P', '\eta_D', '\eta_g', 'energy_per_site')
hold off;

file_path = fullfile(script_dir, '..', 'benchmark_result', string(N) + '_benchmark.mat');
save(file_path, "eta_P", "eta_D", "eta_g", "energy", "energy_per_site", ...
    "tau", "maxiter_opt", "mu", "k", "sigma0", "relative_err", "time", 'r', 'band_width');

function val = JmH(V, y, t, J, lifted_dim)
    X = reshape(V * y, lifted_dim, lifted_dim);
    H = X' * X + t * t';
    val = J - H(:);
end

function store = prepareCGAvec(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store)
    if ~isfield(store, 'CGAvec')
        C = JmH(V, y, t, J, lifted_dim);
        G = R * C - A * Lambda - 1 / sigma * M;
        Avec = reshape(-sigma * G' * R + lambda_sum, lifted_dim, lifted_dim);
        store.CGAvec = struct('C', C, 'G', G, 'Avec', Avec);
    end
end

function [loss, store] = Loss(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store)
    store = prepareCGAvec(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store);
    C = store.CGAvec.C;
    G = store.CGAvec.G;
    loss = -lambda_sum * C + sigma / 2 * norm(G)^2 - norm(M)^2 / (2 * sigma);
end

function [dLdy, store] = dLossy(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store)
    store = prepareCGAvec(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store);
    Avec = store.CGAvec.Avec;
    X = reshape(V * y, lifted_dim, lifted_dim);
    v = X * Avec;

    dLdy = 2 * reshape(v(:)' * V, size(V, 2), 1);
end

function [dLdt, store] = dLosst(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, r, lifted_dim, store)
    store = prepareCGAvec(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store);
    Avec = store.CGAvec.Avec;
    dLdt = 2 * reshape(Avec * t, lifted_dim, r);
end

function Lambda = local_projection(X, sigma, M, N, A_U, Lambda, Idx)
    site_block_dim = round(sqrt(size(A_U, 1)));
    pair_vec_dim = size(A_U, 1);
    lifted_dim = site_block_dim * N + 1;
    mat = reshape(X - 1 / sigma * M, lifted_dim, lifted_dim);
    count = 0;

    for i = 1:N-1
        row_idx = block_range(i, site_block_dim);
        for j = i+1:N
            if Idx(i, j) == 1
                count = count + 1;
                col_idx = block_range(j, site_block_dim);
                m0 = reshape(A_U * reshape(mat(row_idx, col_idx), pair_vec_dim, 1), site_block_dim, site_block_dim);

                m0 = (m0 + m0') / 2;
                [V_mat, Eg] = eig(m0);

                d = diag(Eg);
                d(d < 0) = 0;

                Lambda(pair_vec_dim * count - pair_vec_dim + 1:pair_vec_dim * count) = ...
                    reshape(V_mat * diag(d) * V_mat', pair_vec_dim, 1);
            end
        end
    end
end

function [y, t] = get_yt(y, t, ~, sigma, J, M, R, A, Lambda, lambda_sum, V, ...
    options, problemy, problemt, r, lifted_dim)
    problemt.cost = @(t, store) Loss(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store);
    problemt.egrad = @(t, store) dLosst(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, r, lifted_dim, store);

    [t, ~] = rlbfgs(problemt, t, options);

    problemy.cost = @(y, store) Loss(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store);
    problemy.egrad = @(y, store) dLossy(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store);

    [y, ~] = rlbfgs(problemy, y, options);

    problemt.cost = @(t, store) Loss(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, lifted_dim, store);
    problemt.egrad = @(t, store) dLosst(y, t, sigma, J, M, R, A, Lambda, lambda_sum, V, r, lifted_dim, store);

    [t, ~] = rlbfgs(problemt, t, options);
end

function [eta_P, eta_D, eta_g, primal_obj, dual_obj] = evaluate_iteration_metrics( ...
    M, M_mat, X, C, J, cD, lambda_sum, N, A_U, Idx)
    eta_P = primal_feasibility(M_mat, N, A_U, Idx);
    eta_D = norm(X, 'fro') * cD;

    primal_obj = real(J' * M);
    dual_obj = lambda_sum * C;
    eta_g = abs(primal_obj - dual_obj) / (1 + abs(primal_obj) + abs(dual_obj));
end

function eta_P = primal_feasibility(M_mat, N, A_U, Idx)
    site_block_dim = round(sqrt(size(A_U, 1)));
    pair_vec_dim = size(A_U, 1);

    eta_PM = relative_psd_violation(M_mat);
    eta_Pij = 0;

    for i = 1:N-1
        row_idx = block_range(i, site_block_dim);
        for j = i+1:N
            if Idx(i, j) == 1
                col_idx = block_range(j, site_block_dim);
                rho_ij = reshape(A_U * reshape(M_mat(row_idx, col_idx), pair_vec_dim, 1), ...
                    site_block_dim, site_block_dim);
                eta_Pij = max(eta_Pij, relative_psd_violation(rho_ij));
            end
        end
    end

    eta_P = max(eta_PM, eta_Pij);
end

function eta = relative_psd_violation(mat)
    [~, lambda_max] = eigs(mat, 1, 'largestreal');
    [~, lambda_min] = eigs(mat, 1, 'smallestreal');
    eta = max(0, -real(lambda_min)) / (1 + max(0, real(lambda_max)));
end

function idx = block_range(site_idx, site_block_dim)
    start_idx = site_block_dim * site_idx - (site_block_dim - 1);
    idx = start_idx:(start_idx + site_block_dim - 1);
end

function print_iteration_status(k, eta_P, eta_D, eta_g, time)
    fprintf('Iteration: %f\n', k);
    fprintf('Primal feasibility measure: %f\n', log10(eta_P));
    fprintf('Dual feasibility measure: %f\n', log10(eta_D));
    fprintf('Duality Gap: %f\n', log10(eta_g));
    fprintf('Time: %f\n', time);
end

function Idx = idx_graph(N)
    Idx = sparse(N, N);
    [Lx, Ly] = split_N(N);
    for i = 1:N
        y = ceil(i / Lx);
        x = i - (y - 1) * Lx;
        if x ~= Lx
            Idx(i, i + 1) = 1;
            Idx(i + 1, i) = 1;
        else
            Idx((y - 1) * Lx + 1, i) = 1;
            Idx(i, (y - 1) * Lx + 1) = 1;
        end

        if y ~= Ly
            Idx(i, i + Lx) = 1;
            Idx(i + Lx, i) = 1;
        else
            Idx((Ly - 1) * Lx + x, x) = 1;
            Idx(x, (Ly - 1) * Lx + x) = 1;
        end
    end
end

function [Lx, Ly] = split_N(N)
    power = log2(N);
    if mod(power, 2) == 0
        Lx = power / 2;
    else
        Lx = (power + 1) / 2;
    end
    Lx = 2^Lx;
    Ly = N / Lx;
end








