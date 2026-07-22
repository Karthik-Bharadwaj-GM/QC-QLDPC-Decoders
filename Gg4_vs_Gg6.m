clc
clear all

% ───────────────────────────────────────────────
% Simulation parameters
% ───────────────────────────────────────────────
pr = [0.01:0.01:0.1]; % physical depolarizing error probabilities
max_err = [12000,12000,12000,12000,12000,12000,12000,12000,12000,12000]; % max trials per pr
max_itr = 100; % max iterations for soft decoding
meu = 0.5; % correlation parameter for Markov chain
warning off

% Parity-check matrices definitions
Gg6_H_x=[0 0 0 0 0 0;2 4 8 16 32 64;63 61 57 49 33 1]; % Girth > 6 code, p=65, [[390,132;128]]
Gg6_H_z=[0 0 0 0 0 0;2 4 8 16 32 64;63 61 57 49 33 1];
Gg4_H_x=[0:1:22; mod(2*(0:1:22),23); mod(3*(0:1:22),23); mod(4*(0:1:22),23); mod(5*(0:1:22),23); mod(6*(0:1:22),23);...
    mod(7*(0:1:22),23); mod(8*(0:1:22),23); mod(9*(0:1:22),23); mod(10*(0:1:22),23); mod(11*(0:1:22),23); mod(12*(0:1:22),23);...
    mod(13*(0:1:22),23); mod(14*(0:1:22),23); mod(15*(0:1:22),23); mod(16*(0:1:22),23)]; % Girth > 4, p=23, [[529,176;353]]
Gg4_H_z=[0:1:22; mod(2*(0:1:22),23); mod(3*(0:1:22),23); mod(4*(0:1:22),23); mod(5*(0:1:22),23); mod(6*(0:1:22),23);...
    mod(7*(0:1:22),23); mod(8*(0:1:22),23); mod(9*(0:1:22),23); mod(10*(0:1:22),23); mod(11*(0:1:22),23); mod(12*(0:1:22),23);...
    mod(13*(0:1:22),23); mod(14*(0:1:22),23); mod(15*(0:1:22),23); mod(16*(0:1:22),23)];

% Prepare result storage: 2 constructions (with burst, 2 without burst) × number of pr points
res = zeros(4, length(pr));

% ───────────────────────────────────────────────
% Main loop over two constructions
% ───────────────────────────────────────────────
constructions = {Gg6_H_x, Gg6_H_z; Gg4_H_x, Gg4_H_z};

for Bust_req = 0:1
    for code_idx = 1:2
        warning off
        if code_idx==1
            p = 65;
            Bust_len = p;
        else
            p = 23;
            Bust_len = p;
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
   
                % Burst Error (only if Bust_req==1)
                if Bust_req == 1
                    Burst = zeros(1, 2*N);
                    Int_feed = zeros(1, 4);
                    indx = randsrc(1, N, [0 1 2 3; 1-pr(i) pr(i)/3 pr(i)/3 pr(i)/3]);
                    Int_feed(indx+1) = 1;
                    Bus_s = randi([1 N-Bust_len]);
                    Vat = simulate(Mc, Bust_len, "X0", Int_feed);
                    for k = Bus_s:Bus_s+Bust_len
                        Burst([k N+k]) = dec2bin(Vat(k-Bus_s+1,1)-1) - '0';
                    end
                    error = mod(Burst + error, 2);
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
                [E, flag] = Soft_Dec_combine_BLNMS_dyn(H_css, pr(i), S, max_itr, p, 1, 0.625, 0.7, 0.98); 
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
        res(code_idx + 2*Bust_req, :) = log_err_int(1:length(pr)); % take first length(pr) elements
   
    end % end construction loop
end % end burst loop
   
% ───────────────────────────────────────────────
% Final plotting (both curves)
% ───────────────────────────────────────────────
figure;
hold on;
plot(pr, res(1,:), 'o-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', '[[390,132;128]]_2 code with random errors, girth>6');
plot(pr, res(2,:), 's-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', '[[529,176;353]]_2 code with random errors, girth>4');
plot(pr, res(3,:), 'o-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', '[[390,132;128]]_2 code with random and length 65 burst error, girth>6');
plot(pr, res(4,:), 's-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', '[[529,176;353]]_2 code with random and length 23 burst error, girth>4');
set(gca, 'XScale', 'log', 'YScale', 'log');
grid on;
box on;
xlabel('Depolarizing error probability p_d');
ylabel('Logical error rate (LER)');
title(sprintf('Logical error rate vs. Depolarizing error probability'));
legend('Location', 'northwest', 'FontSize', 11);
hold off;