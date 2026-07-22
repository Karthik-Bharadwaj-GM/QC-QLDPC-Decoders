clc
clear all

% ───────────────────────────────────────────────
% Simulation parameters
% ───────────────────────────────────────────────
p = 17; % permutation matrix size
pr = [0.01:0.02:0.1]; % physical depolarizing error probabilities
max_err = [1e7, 10000, 1000, 1000, 1000]; % max trials per pr
max_itr = 100; % max iterations for soft decoding
meu = 0.5; % correlation parameter for Markov chain
warning off

% Parity-check matrices definitions
cw3_H_x = [0:1:16; mod(2*([0:1:16]),17); mod(3*([0:1:16]),17)];
cw3_H_z = [mod(4*([0:1:16]),17); mod(5*([0:1:16]),17); mod(6*([0:1:16]),17)];

cw4_H_x = [0:1:16; mod(2*([0:1:16]),17); mod(3*([0:1:16]),17); mod(4*([0:1:16]),17)];
cw4_H_z = [mod(5*([0:1:16]),17); mod(6*([0:1:16]),17); mod(7*([0:1:16]),17); mod(8*([0:1:16]),17)];

cw5_H_x = [0:1:16; mod(2*([0:1:16]),17); mod(3*([0:1:16]),17); mod(4*([0:1:16]),17); mod(5*([0:1:16]),17)];
cw5_H_z = [mod(6*([0:1:16]),17); mod(7*([0:1:16]),17); mod(8*([0:1:16]),17); mod(9*([0:1:16]),17); mod(10*([0:1:16]),17)];

cw6_H_x = [0:1:16; mod(2*([0:1:16]),17); mod(3*([0:1:16]),17); mod(4*([0:1:16]),17); mod(5*([0:1:16]),17); mod(6*([0:1:16]),17)];
cw6_H_z = [mod(7*([0:1:16]),17); mod(8*([0:1:16]),17); mod(9*([0:1:16]),17); mod(10*([0:1:16]),17); mod(11*([0:1:16]),17); mod(12*([0:1:16]),17)];

cw7_H_x = [0:1:16; mod(2*([0:1:16]),17); mod(3*([0:1:16]),17); mod(4*([0:1:16]),17); mod(5*([0:1:16]),17); mod(6*([0:1:16]),17); mod(7*([0:1:16]),17)];
cw7_H_z = [mod(8*([0:1:16]),17); mod(9*([0:1:16]),17); mod(10*([0:1:16]),17); mod(11*([0:1:16]),17); mod(12*([0:1:16]),17); mod(13*([0:1:16]),17); mod(14*([0:1:16]),17)];

cw8_H_x = [0:1:16; mod(2*([0:1:16]),17); mod(3*([0:1:16]),17); mod(4*([0:1:16]),17); mod(5*([0:1:16]),17); mod(6*([0:1:16]),17); mod(7*([0:1:16]),17); mod(8*([0:1:16]),17)];
cw8_H_z = [mod(9*([0:1:16]),17); mod(10*([0:1:16]),17); mod(11*([0:1:16]),17); mod(12*([0:1:16]),17); mod(13*([0:1:16]),17); mod(14*([0:1:16]),17); mod(15*([0:1:16]),17); mod(16*([0:1:16]),17)];

% Prepare result storage
res = zeros(6, length(pr));

% ───────────────────────────────────────────────
% Main loop over 6 constructions
% ───────────────────────────────────────────────
constructions = {cw3_H_x, cw3_H_z; cw4_H_x, cw4_H_z; cw5_H_x, cw5_H_z; cw6_H_x, cw6_H_z; cw7_H_x, cw7_H_z; cw8_H_x, cw8_H_z};
legend_names = {'[[289,192;1]]_2 code with column weight 3',...
'[[289,160;1]]_2 code with column weight 4',...
'[[289,128;1]]_2 code with column weight 5',...
'[[289,96;1]]_2 code with column weight 6',...
'[[289,64;1]]_2 code with column weight 7',...
'[[289,32;1]]_2 code with column weight 8'};

for code_idx = 1:6
    switch code_idx
        case 1 | 2 | 3
            alpha = 0.8;
            beta = 0.95;
        case 4 |5 
            alpha = 0.75;
            beta = 0.75;
        otherwise
            alpha = 0.7;
            beta = 0.65;
    end
    % Select current parity-check matrices
    H_x_base = constructions{code_idx, 1};
    H_z_base = constructions{code_idx, 2};

    % Generate shifted/expanded parity-check matrices
    H_x = Encoder_shift(p, H_x_base);
    H_z = Encoder_shift(p, H_z_base);

    % Build full CSS check matrix
    [~, N] = size(H_x);
    H_css = [H_x zeros(size(H_x)); zeros(size(H_z)) H_z];

    log_err_int = zeros(1, length(pr) + length(meu) - 1);

    corr = 1;
    for i = 1:length(pr)

        p_phys = pr(i);
        n_trials = max_err(i);
        log_err = 0;

        % Build 4-state Markov transition matrix
        Trans_mat = zeros(4);
        Trans_mat(:,1) = (1 - meu(corr)) * (1 - p_phys);
        Trans_mat(:,[2 3 4]) = (1 - meu(corr)) * p_phys / 3;

        for mm = 1:4
            Trans_mat(mm,mm) = Trans_mat(mm,mm) + meu(corr);
        end

        Mc = dtmc(Trans_mat);

        % ====================== REPRODUCIBLE RANDOMNESS SETUP ======================
        % Create a fixed random stream for all workers (Threefry generator + fixed seed)
        sc = parallel.pool.Constant(RandStream('Threefry', 'Seed', 0));
        % ===========================================================================

        % Preallocate failure counter for parfor slicing
        failure_count = zeros(n_trials, 1);

        % ───────────────────────────────────────────────
        % Monte Carlo trials
        % ───────────────────────────────────────────────
        parfor j = 1:n_trials
            warning('off','all')

            % ====================== SET SUBSTREAM FOR REPRODUCIBILITY ======================
            % Each iteration j gets its own independent but fixed random sequence
            stream = sc.Value;
            stream.Substream = j;
            RandStream.setGlobalStream(stream);
            % =============================================================================

            % Generate random depolarizing errors
            Err = randsrc(1, N, [0 1 2 3; 1-p_phys p_phys/3 p_phys/3 p_phys/3]);

            % Convert to binary X and Z error vectors
            error = zeros(1, 2*N);
            for l = 1:N
                error([l, N+l]) = dec2bin(Err(l), 2) - '0';
            end

            % Compute syndromes
            Syn_x = mod(H_x * error(1:N)', 2)';
            Syn_z = mod(H_z * error(N+1:2*N)', 2)';

            if sum(Syn_x) == 0 && sum(Syn_z) == 0
                continue
            end

            % Soft-decision decoding
            S = [Syn_x Syn_z];
            [E, flag] = Soft_Dec_combine_BLNMS_dyn(H_css, pr(i), S, max_itr, p, 1, 0.625, 0.7, 0.98);
            Err_z = E(N+1:2*N);
            Err_x = E(1:N);

            % Estimated correction
            error_E = [Err_z Err_x];

            % Residual error after correction
            stab = mod(error + error_E, 2);
            XXs = mod(H_css * stab', 2);

            if ~isempty(find(XXs))
                failure_count(j) = 1;
            else
                if isempty(gflineq(H_css', [stab(N+1:2*N) stab(1:N)]'))
                    failure_count(j) = 1;
                end
            end

        end % end parfor

        % Sum failures
        log_err = sum(failure_count);

        % Store result
        log_err_int(i + corr - 1) = log_err / n_trials;

    end % end pr loop

    % Assign to result
    res(code_idx, :) = log_err_int(1:length(pr));

end % end construction loop

% ───────────────────────────────────────────────
% Final plotting
% ───────────────────────────────────────────────
figure;
hold on;

plot(pr, res(1,:), 'o-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', legend_names{1});
plot(pr, res(2,:), 's-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', legend_names{2});
plot(pr, res(3,:), '<-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', legend_names{3});
plot(pr, res(4,:), '|-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', legend_names{4});
plot(pr, res(5,:), '*-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', legend_names{5});
plot(pr, res(6,:), 'x-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', legend_names{6});

set(gca, 'XScale', 'log', 'YScale', 'log');
grid on;
box on;

xlabel('Depolarizing error probability p_d');
ylabel('Logical error rate (LER)');
title(sprintf('Logical error rate vs. Depolarizing error probability', p));
legend('Location', 'northwest', 'FontSize', 11);

hold off;