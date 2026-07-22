clc
clear all

% ───────────────────────────────────────────────
% Simulation parameters
% ───────────────────────────────────────────────
pr = [1e-3 0.01:0.02:0.1]; % physical depolarizing error probabilities
max_err = [1e9, 1e6, 1000, 1000, 100, 10]; % max trials per pr
max_itr = 100; % max iterations for soft decoding
meu = 0.5; % correlation parameter 

% Parity-check matrices definitions
M_H_x=[1 1 1 1 1 1 1 1;1 2 3 4 5 6 7 8;1 3 5 7 9 11 13 15]; % min-hsieh, p=16
M_H_z=[1 1 1 1 1 1 1 1;1 2 3 4 5 6 7 8;1 3 5 7 9 11 13 15];
P_H_x=[0 0 0 0 0 0 0 0 0 0 0; 0 1 2 3 4 5 6 7 8 9 10; mod(2*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(3*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(4*([0 1 2 3 4 5 6 7 8 9 10]),11)]; % pavan, p=11
P_H_z=[0 0 0 0 0 0 0 0 0 0 0; 0 1 2 3 4 5 6 7 8 9 10; mod(2*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(3*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(4*([0 1 2 3 4 5 6 7 8 9 10]),11)]; % pavan, p=11

% Prepare result storage: 2 constructions × number of pr points
res = zeros(2, length(pr));

% ───────────────────────────────────────────────
% Main loop over two constructions
% ───────────────────────────────────────────────
constructions = {M_H_x, M_H_z; P_H_x, P_H_z};
legend_names = {'[[128,58;18]]_2 code with random errors (QBLNMS decoder)', '[[121,70;51]]_2 code with random errors (QBLNMS decoder)'};

for code_idx = 1:2
    warning off
    if code_idx==1
        p = 16;
    else
        p = 11;
    end
   
    % Select current parity-check matrices
    H_x_base = constructions{code_idx, 1};
    H_z_base = constructions{code_idx, 2};
   
    % Generate shifted/expanded parity-check matrices
    H_x = Encoder_shift(p, H_x_base);
    H_z = Encoder_shift(p, H_z_base);
   
    % Build full CSS check matrix (stacked X and Z parts)
    [~, N] = size(H_x);
    H_css = [H_x zeros(size(H_x)); zeros(size(H_z)) H_z];
   
    % ───────────────────────────────────────────────
    % Correlation loop (original structure, even though meu is scalar)
    % ───────────────────────────────────────────────
    log_err_int = zeros(1, length(pr) + length(meu) - 1); % preallocate as in original
   
    corr = 1; % since length(meu)=1
    for i = 1:length(pr)
       
        p_phys = pr(i);
        n_trials = max_err(i);
        log_err = 0; % scalar accumulator (will use sliced for parfor compatibility)
       
        % Build 4-state Markov transition matrix (correlated depolarizing)
        Trans_mat = zeros(4);
        Trans_mat(:,1) = (1 - meu(corr)) * (1 - p_phys);
        Trans_mat(:,[2 3 4]) = (1 - meu(corr)) * p_phys / 3;
       
        % Add self-transition probability (correlation)
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
        % Monte Carlo trials (parallelized, with sliced accumulator)
        % ───────────────────────────────────────────────
        parfor j = 1:n_trials
            warning('off','all')
            
            % ====================== SET SUBSTREAM FOR REPRODUCIBILITY ======================
            % Each iteration j gets its own independent but fixed random sequence
            stream = sc.Value;
            stream.Substream = j;
            RandStream.setGlobalStream(stream);
            % =============================================================================
            
            % Generate random depolarizing errors (0=I,1=X,2=Z,3=Y)
            Err = randsrc(1, N, [0 1 2 3; 1-p_phys p_phys/3 p_phys/3 p_phys/3]);
           
            % Convert to binary X and Z error vectors (original assignment style)
            error = zeros(1, 2*N); % preallocate row vector
            for l = 1:N
                error([l, N+l]) = dec2bin(Err(l), 2) - '0';
            end
           
            % ─── Compute syndromes ───
            Syn_x = mod(H_x * error(1:N)', 2)';
            Syn_z = mod(H_z * error(N+1:2*N)', 2)';
           
            % Trivial syndrome → success (no logical error, original continue)
            if sum(Syn_x) == 0 && sum(Syn_z) == 0
                continue
            end
           
            % ─── Soft-decision decoding (combined quaternary version) ───
            S = [Syn_x Syn_z];
            [E, flag] = Soft_Dec_combine_BLNMS_dyn(H_css, pr(i), S, max_itr, p, 1, 0.625, 0.7, 0.98); % original flag
            Err_z = E(N+1:2*N);
            Err_x = E(1:N);

            % Estimated correction (exact original order: [Err_z Err_x])
            error_E = [Err_z Err_x];
           
            % Residual error after correction
            stab = mod(error + error_E, 2);
            XXs = mod(H_css * stab', 2);
           
            % Check if decoding failed (non-zero syndrome, original logic)
            if ~isempty(find(XXs))
                failure_count(j) = 1;
            else
                % Additional logical error check via linear dependence (exact original)
                if isempty(gflineq(H_css', [stab(N+1:2*N) stab(1:N)]'))
                    failure_count(j) = 1;
                end
            end
           
        end % end parfor
       
        % Sum failures (replaces original scalar accumulation for parfor compatibility)
        log_err = sum(failure_count);
       
        % Store as in original (i + corr - 1)
        log_err_int(i + corr - 1) = log_err / n_trials;
       
    end % end pr loop
   
    % Assign to result (original log_err_plot equivalent)
    res(code_idx, :) = log_err_int(1:length(pr)); % take first length(pr) elements
   
end % end construction loop

% ───────────────────────────────────────────────
% Final plotting (both curves)
% ───────────────────────────────────────────────
figure;
hold on;
plot(pr, res(1,:), 'o-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', legend_names{1});
plot(pr, res(2,:), 's-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', legend_names{2});
set(gca, 'XScale', 'log', 'YScale', 'log');
grid on;
box on;
xlabel('Depolarizing error probability p_d');
ylabel('Logical error rate (LER)');
title(sprintf('Logical error rate vs. Depolarizing error probability'));
legend('Location', 'northwest', 'FontSize', 11);
hold off;