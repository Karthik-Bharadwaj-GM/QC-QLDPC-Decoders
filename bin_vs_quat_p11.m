clc
clear all
max_err=[1e7,120000,12000,12000,12000,12000,12000,12000,12000,12000];
% max_err=[12000,12000,12000,12000,12000,12000,12000,12000];
prime=true;
pr=[0.01:0.01:0.1];
% pr=[0.03:0.01:0.1];
p=11;
max_itr=100;
Bust_len=p;
EA_H_x=[0 1 2 3 4 5 6 7 8 9 10; mod(2*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(3*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(4*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(5*([0 1 2 3 4 5 6 7 8 9 10]),11)];
EA_H_z=[mod(6*[0 1 2 3 4 5 6 7 8 9 10],11); mod(7*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(8*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(9*([0 1 2 3 4 5 6 7 8 9 10]),11); mod(10*([0 1 2 3 4 5 6 7 8 9 10]),11)];
H_x = Encoder_shift(p, EA_H_x);
H_z = Encoder_shift(p, EA_H_z);
[~, N] = size(H_x);
H_css = [H_x zeros(size(H_x)); zeros(size(H_z)) H_z];
meu = 0.5; % Correlation parameter
warning off

% Storage for all result curves
all_results = zeros(10, length(pr));
% Row 1: Bust_req=0, BLSP
% Row 2: Bust_req=0, QMS
% Row 3: Bust_req=0, QNMS
% Row 4: Bust_req=0, QBLNMS
% Row 5: Bust_req=0, QLNMS
% Row 6: Bust_req=1, BLSP
% Row 7: Bust_req=1, QMS
% Row 8: Bust_req=1, QNMS
% Row 9: Bust_req=1, QBLNMS
% Row 10: Bust_req=1, QLNMS
result_idx = 0;

for Bust_req = 0:1
    for dec_type = 1:4 % 1=BLSP, 2=QMS, 3=QNMS, 4=QBLNMS, 5=QLNMS
        result_idx = result_idx + 1;
        log_err_int = zeros(1, length(pr));
        for i = 1:length(pr)
            if dec_type == 1
                dec_name = 'BLSP';
            elseif dec_type == 2
                dec_name = 'QMS';
            elseif dec_type == 3
                dec_name = 'QNMS';
            elseif dec_type == 4
                dec_name = 'QBLNMS';
            else
                dec_name = 'QLNMS';
            end
            fprintf('Bust_req=%d, DecType=%s, pr=%.2f\n', Bust_req, dec_name, pr(i));
            log_err = 0;
            
            %% Transition Matrix for the Markov chain
            Trans_mat = zeros(4);
            Trans_mat(:,1) = (1-meu) * (1-pr(i));
            Trans_mat(:,[2 3 4]) = (1-meu) * pr(i) / 3;
            for mm = 1:4
                Trans_mat(mm,mm) = Trans_mat(mm,mm) + meu;
            end
            Mc = dtmc(Trans_mat);
            
            % ====================== REPRODUCIBLE RANDOMNESS SETUP ======================
            % Create a fixed random stream for all workers (Threefry generator + fixed seed)
            sc = parallel.pool.Constant(RandStream('Threefry', 'Seed', 0));
            % ===========================================================================
            
            parfor j = 1:max_err(i)
                warning off
                
                % ====================== SET SUBSTREAM FOR REPRODUCIBILITY ======================
                % Each iteration j gets its own independent but fixed random sequence
                stream = sc.Value;
                stream.Substream = j;
                RandStream.setGlobalStream(stream);
                % =============================================================================
                
                error = zeros(1, 2*N);
                
                %% Generate depolarizing errors
                Err = randsrc(1, N, [0 1 2 3; 1-pr(i) pr(i)/3 pr(i)/3 pr(i)/3]);
                for l = 1:N
                    error([l, N+l]) = dec2bin(Err(l), 2) - '0';
                end
                
                %% Burst Error (only if Bust_req==1)
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
                
                %% Syndrome Computation
                Syn_x = mod(H_x * error(1:N)', 2)';
                Syn_z = mod(H_z * error(1+N:2*N)', 2)';
                if sum(Syn_x) == 0 && sum(Syn_z) == 0
                    continue
                end
                
                %% Decoding
                if dec_type == 1
                    %% BINARY DECODING
                    [Err_z, flag_x] = Soft_Dec_bsp_lay(H_x, p, pr(i), Syn_x, max_itr);
                    [Err_x, flag_z] = Soft_Dec_bsp_lay(H_z, p, pr(i), Syn_z, max_itr);
                elseif dec_type == 2
                    %% QUATERNARY MIN-SUM DECODING
                    S = [Syn_x Syn_z];
                    [E, flag] = Soft_Dec_combine(H_css, pr(i), S, max_itr);
                    Err_z = E(N+1:2*N);
                    Err_x = E(1:N);
                elseif dec_type == 3
                    %% QUATERNARY NORMALIZED MIN-SUM DECODING
                    S = [Syn_x Syn_z];
                    [E, flag] = Soft_Dec_combine_NMS(H_css, pr(i), S, max_itr);
                    Err_z = E(N+1:2*N);
                    Err_x = E(1:N);
                elseif dec_type == 4
                    %% QUATERNARY BLOCK LAYERED NORMALIZED MIN-SUM DECODING
                    S = [Syn_x Syn_z];
                    [E, flag] = Soft_Dec_combine_BLNMS_dyn(H_css, pr(i), S, max_itr, p, 1, 0.625, 0.7, 0.98);
                    Err_z = E(N+1:2*N);
                    Err_x = E(1:N);
                else
                    %% QUATERNARY LAYERED NORMALIZED MIN-SUM DECODING
                    S = [Syn_x Syn_z];
                    [E, flag] = Soft_Dec_combine_LNMS(H_css, pr(i), S, max_itr);
                    Err_z = E(N+1:2*N);
                    Err_x = E(1:N);
                end
                
                %% Logical error check
                error_E = [Err_z Err_x];
                stab = mod(error + error_E, 2);
                XXs = mod(H_css * stab', 2);
                if ~isempty(find(XXs))
                    log_err = log_err + 1;
                else
                    if isempty(gflineq(H_css', [stab(N+1:2*N) stab(1:N)]'))
                        log_err = log_err + 1;
                    end
                end
            end % parfor
            
            log_err_int(i) = log_err / max_err(i);
        end % pr loop
        
        all_results(result_idx, :) = log_err_int;
    end % dec_type loop
end % Bust_req loop

%% Plot all eight results
figure;
legend_labels = { ...
    '[[121,20,10;1]]_2 code with random errors (BLSP decoder)', ...
    '[[121,20,10;1]]_2 code with random errors (QMS decoder)', ...
    '[[121,20,10;1]]_2 code with random errors (QNMS decoder)', ...
    '[[121,20,10;1]]_2 code with random errors (QBLNMS decoder)', ... % '[[121,20,10;1]]_2 code with random errors (QLNMS decoder)', ...
    '[[121,20,10;1]]_2 code with random and length 11 burst error (BLSP decoder)', ...
    '[[121,20,10;1]]_2 code with random and length 11 burst error (QMS decoder)', ...
    '[[121,20,10;1]]_2 code with random and length 11 burst error (QNMS decoder)', ...
    '[[121,20,10;1]]_2 code with random and length 11 burst error (QBLNMS decoder)', ... %'[[121,20,10;1]]_2 code with random and length 11 burst error (QLNMS decoder)', ...
    };
line_styles = {'-o', '-s', '-d', '-x', '--o', '--s', '--d', '--x'};
colors = [
    0.0000 0.0000 1.000;   % blue
    1.000 0.0000 0.0000;   % red
    0.0000 0.7500 0.7500;   % teal
    0.7500 0.0000 0.7500;   % purple
    0.0000 0.0000 1.0000;   % blue
    1.0000 0.0000 0.0000;   % red
    0.0000 0.7500 0.7500;   % teal
    0.7500 0.0000 0.7500;   % purple
];

hold on;
for k = 1:8
    loglog(pr, all_results(k,:), line_styles{k}, ...
        'Color', colors(k,:), ...
        'LineWidth', 1.5, ...
        'DisplayName', legend_labels{k});
end
hold off;
set(gca, 'XScale', 'log', 'YScale', 'log');
grid on;
legend('show', 'Location', 'best');
title('Logical error rate vs. Depolarizing noise probability');
xlabel('Depolarizing noise probability p_d');
ylabel('Logical error rate (LER)');