function [Err_b, flag] = Soft_Dec_combine_BLNMS_dyn(H_css, p, S, max_itr, blk_size, ...
                                                  alpha_start, alpha_end, ...
                                                  beta_start,  beta_end,  ...
                                                  ~)
% Block-Layered Normalized Quaternary Min-Sum Decoder
% with Syndrome-Driven Per-Block Dynamic alpha/beta
%
% alpha and beta are now adapted per-block per-iteration based on the
% fraction of UNSATISFIED syndromes within that block, measured using
% the current live Lambda BEFORE the block snapshot is taken.
%
%   sat_ratio = (number of unsatisfied syns in block) / blk_size   in [0,1]
%
%   alpha(block) = alpha_end + (alpha_start - alpha_end) * sat_ratio
%   beta(block)  = beta_end  - (beta_end   - beta_start) * sat_ratio
%
% Interpretation:
%   sat_ratio = 1  (all syns unsatisfied) -> alpha = alpha_start, beta = beta_start
%               Aggressive normalization, less damping: accept large moves.
%   sat_ratio = 0  (all syns satisfied)   -> alpha = alpha_end,   beta = beta_end
%               Tight normalization, full damping: stabilize near convergence.
%
% Arguments:
%   H_css       : [R x 2C] CSS parity-check matrix
%   p           : channel error probability
%   S           : syndrome vector (length R)
%   max_itr     : maximum number of decoding iterations
%   blk_size    : rows per block          (default: 1  -> row-layered)
%   alpha_start : alpha when block fully unsatisfied  (default: 1.0)
%   alpha_end   : alpha when block fully satisfied    (default: 0.625)
%   beta_start  : beta  when block fully unsatisfied  (default: 0.7)
%   beta_end    : beta  when block fully satisfied    (default: 0.98)


    %% Defaults
    if nargin < 5  || isempty(blk_size),     blk_size    = 1;     end
    if nargin < 6  || isempty(alpha_start),  alpha_start = 1.0;   end
    if nargin < 7  || isempty(alpha_end),    alpha_end   = 0.625; end
    if nargin < 8  || isempty(beta_start),   beta_start  = 0.7;   end
    if nargin < 9  || isempty(beta_end),     beta_end    = 0.98;  end
    if nargin < 10, decay_rate = []; end  % accepted, not used

    flag = false;
    [R, C_full] = size(H_css);
    C = C_full / 2;

    % Transform H_css to quaternary {0,1,2,3}
    H = H_css(:, 1:C) + 2 * H_css(:, C+1:2*C);

    % Channel prior
    gamma = log(p / (3 - 3 * p));
    Lb    = [0, gamma, gamma, gamma];
    llr_v = max_exp(0, gamma) - max_exp(gamma, gamma);

    % neu(r,vn): scalar VN-to-CN messages
    neu = ones(R, C) * llr_v;
    neu(H == 0) = inf;

    % meu(r,vn,:): CN-to-VN 4-component offsets
    meu = zeros(R, C, 4);

    % Lambda(vn,:): sum of meu only (Lb excluded)
    Lambda = zeros(C, 4);

    % Block structure
    num_blocks = ceil(R / blk_size);

    %% Iterative block-layered decoding
    for w = 1:max_itr

        block_perm = randperm(num_blocks);

        for bb = 1:num_blocks

            % ------------------------------------------------------------------
            % Identify rows in this block
            % ------------------------------------------------------------------
            b          = block_perm(bb);
            row_start  = (b - 1) * blk_size + 1;
            row_end    = min(b * blk_size, R);
            block_rows = row_start:row_end;
            nb         = length(block_rows);   

            % ------------------------------------------------------------------
            % SYNDROME-DRIVEN alpha/beta
            % ------------------------------------------------------------------
            unsat = 0;
            for idx = 1:nb
                i = block_rows(idx);
                [~, Neigh_Chk] = find(H(i, :));
                if isempty(Neigh_Chk), continue; end

                % Refresh neu for syndrome check using live Lambda
                % (same formula as Step 1 below, but using live Lambda)
                neu_signs = zeros(1, length(Neigh_Chk));
                for j = 1:length(Neigh_Chk)
                    vn = Neigh_Chk(j);
                    ex = Lb + Lambda(vn, :) - squeeze(meu(i, vn, :))';
                    if H(i, vn) == 1
                        val = max_exp(ex(1), ex(2)) - max_exp(ex(3), ex(4));
                    elseif H(i, vn) == 2
                        val = max_exp(ex(1), ex(3)) - max_exp(ex(2), ex(4));
                    elseif H(i, vn) == 3
                        val = max_exp(ex(1), ex(4)) - max_exp(ex(2), ex(3));
                    end
                    neu_signs(j) = sign(val);
                end

                % Sign product of all neu in this row, flipped by syndrome bit
                % If (-1)^S(i) * prod(signs) < 0 => check unsatisfied
                check_sign = (-1)^(S(i)) * prod(neu_signs);
                if check_sign < 0
                    unsat = unsat + 1;
                end
            end

            sat_ratio = unsat / nb;   % in [0,1]; 1=fully unsat, 0=fully sat

            % Map sat_ratio to alpha and beta
            alpha = alpha_end + (alpha_start - alpha_end) * sat_ratio;
            beta  = beta_end  - (beta_end   - beta_start) * sat_ratio;

            % ------------------------------------------------------------------
            % SNAPSHOT: Lambda before touching anything in this block.
            % All neu refreshes inside the block read from Lambda_snap ->
            % flooding semantics within block; between blocks Lambda is live.
            % ------------------------------------------------------------------
            Lambda_snap  = Lambda;
            delta_Lambda = zeros(C, 4);
            new_meu_blk  = meu(block_rows, :, :);

            % ------------------------------------------------------------------
            % Step 1: Refresh neu for ALL rows in block using Lambda_snap
            % ------------------------------------------------------------------
            for idx = 1:nb
                i = block_rows(idx);
                [~, Neigh_Chk] = find(H(i, :));
                nc = length(Neigh_Chk);

                for j = 1:nc
                    vn = Neigh_Chk(j);
                    ex = Lb + Lambda_snap(vn, :) - squeeze(meu(i, vn, :))';

                    if H(i, vn) == 1
                        neu(i, vn) = max_exp(ex(1), ex(2)) - max_exp(ex(3), ex(4));
                    elseif H(i, vn) == 2
                        neu(i, vn) = max_exp(ex(1), ex(3)) - max_exp(ex(2), ex(4));
                    elseif H(i,vn) == 3
                        neu(i, vn) = max_exp(ex(1), ex(4)) - max_exp(ex(2), ex(3));
                    end
                end
            end

            % ------------------------------------------------------------------
            % Steps 2+3: Normalized min-sum with syndrome-driven alpha +
            %            damped meu update with syndrome-driven beta
            % ------------------------------------------------------------------
            for idx = 1:nb
                i = block_rows(idx);
                [~, Neigh_Chk] = find(H(i, :));
                nc = length(Neigh_Chk);

                neu_chk = neu(i, Neigh_Chk);

                for j = 1:nc
                    vn = Neigh_Chk(j);

                    %% Optimized min1/min2
                    abs_neu        = abs(neu_chk);
                    abs_neu(j)     = inf;              % exclude self
                    min1           = min(abs_neu);
                    idx_m          = find(abs_neu == min1, 1);
                    abs_neu(idx_m) = inf;
                    min2           = min(abs_neu);

                    sign_prod = (-1)^(S(i)) * prod(sign(neu_chk([1:j-1, j+1:end])));

                    if min1 == 0
                        x = sign_prod * 1;
                    else
                        if abs(neu_chk(j)) == min1
                            x = sign_prod * min2;      % use second minimum
                        else
                            x = sign_prod * min1;      % use first minimum
                        end
                    end

                    x = alpha * x;                     % syndrome-driven normalization

                    %% Symplectic mapping to 4-component meu
                    if H(i, vn) == 1
                        new_meu = [0,  0, -x, -x];
                    elseif H(i, vn) == 2
                        new_meu = [0, -x,  0, -x];
                    elseif H(i,vn) == 3
                        new_meu = [0, -x, -x,  0];
                    end

                    %% Syndrome-driven damped update
                    old_meu  = squeeze(meu(i, vn, :))';
                    meu_damp = beta * new_meu + (1 - beta) * old_meu;

                    new_meu_blk(idx, vn, :) = meu_damp;
                    delta_Lambda(vn, :) = delta_Lambda(vn, :) - old_meu + meu_damp;
                end
            end

            % ------------------------------------------------------------------
            % Commit: flush accumulated delta into Lambda; write back meu
            % ------------------------------------------------------------------
            Lambda = Lambda + delta_Lambda;
            meu(block_rows, :, :) = new_meu_blk;

        end  % block loop

        %% Hard decision & syndrome check (after full iteration)
        Err   = zeros(1, C);
        Err_b = zeros(1, 2 * C);
        for i = 1:C
            [~, Err(i)] = max(Lb + Lambda(i, :));
            if Err(i) == 1
                Err_b([i, C + i]) = [0, 0];
            elseif Err(i) == 2
                Err_b([i, C + i]) = [1, 0];
            elseif Err(i) == 3
                Err_b([i, C + i]) = [0, 1];
            else
                Err_b([i, C + i]) = [1, 1];
            end
        end

        Til_S = mod(H_css * [Err_b(C + 1:2*C), Err_b(1:C)]', 2);
        if all(S == Til_S')
            flag = true;
            Err  = Err - 1;
            break;
        end

    end  % iteration loop
end

function A = max_exp(a, b)
    A = max(a, b) + log(1 + exp(-abs(a - b)));
end