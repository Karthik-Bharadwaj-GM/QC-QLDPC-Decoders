function [wwc, flag] = Soft_Dec_bsp_lay(H, p_circ, p, Syn, max_itr)
% Layered Sum-Product decoder where ONE LAYER = ONE BLOCK ROW of p_circ rows
%
% Inputs:
%   H        : parity check matrix [R x C]
%   p_circ   : circulant size (number of rows per block-row)
%   p        : crossover probability (BSC channel)
%   Syn      : syndrome vector [1 x R]
%   max_itr  : maximum number of iterations
%
% Outputs:
%   wwc      : hard decision codeword estimate [1 x C]
%   flag     : true if valid codeword found

    flag = false;
    [R, C] = size(H);

    % Number of block-rows (layers)
    num_layers = R / p_circ;
    if mod(R, p_circ) ~= 0
        error('R must be divisible by p_circ (circulant size).');
    end

    % Syndrome sign: +1 if Syn=0, -1 if Syn=1
    gamma = 1 - 2 * Syn;                      % [1 x R]

    % Channel LLR for BSC
    llr_v = log((1 - (2/3)*p) / ((2/3)*p));

    % Check-to-variable messages, initialised to zero
    meu = zeros(R, C);

    % Lambda(v): running belief for each variable node
    %   = channel LLR + sum over ALL check rows of meu(row, v)
    Lambda = llr_v * ones(1, C);              % [1 x C]

    for w = 1:max_itr
        %% ---- Iterate over block-rows (layers) ---------------------------
        for blk = 1:num_layers

            % Row indices belonging to this block-row layer
            row_start = (blk - 1) * p_circ + 1;
            row_end   = blk * p_circ;
            blk_rows  = row_start:row_end;    % [1 x p_circ]

            % --------------------------------------------------------------
            % Step 1: Subtract OLD check-to-variable messages for ALL rows
            %         in this block-row from Lambda BEFORE any recomputation.
            %         This gives each check row in the block a consistent
            %         extrinsic input that excludes the entire block's
            %         old contribution simultaneously.
            % --------------------------------------------------------------
            neu_block = zeros(p_circ, C);     % will index by Neigh per row

            for i_local = 1:p_circ
                i = blk_rows(i_local);
                [~, Neigh] = find(H(i, :));
                % Subtract this row's old messages from Lambda to get extrinsic
                neu_block(i_local, Neigh) = Lambda(Neigh) - meu(i, Neigh);
            end

            % --------------------------------------------------------------
            % Step 2: Check-to-variable update for ALL rows in the block
            %         using the extrinsic inputs captured in Step 1.
            %         Lambda is updated after ALL new messages are computed
            %         so rows within the same block don't interfere.
            % --------------------------------------------------------------
            meu_new_block = zeros(R, C);      % stores new messages for block

            for i_local = 1:p_circ
                i = blk_rows(i_local);
                [~, Neigh] = find(H(i, :));

                neu_i     = neu_block(i_local, Neigh);   
                tanh_vals = tanh(neu_i / 2);             

                for j = 1:length(Neigh)
                    tanh_prod = prod(tanh_vals([1:j-1, j+1:end]));
                    meu_new_block(i, Neigh(j)) = 2 * sign(gamma(i)) * atanh(tanh_prod);
                end
            end

            % --------------------------------------------------------------
            % Step 3: Update Lambda and meu using the newly computed block
            %         messages. All rows in the block commit simultaneously,
            %         so the next block-layer sees a fully updated Lambda.
            % --------------------------------------------------------------
            for i_local = 1:p_circ
                i = blk_rows(i_local);
                [~, Neigh] = find(H(i, :));

                Lambda(Neigh) = Lambda(Neigh) ...
                                - meu(i, Neigh) ...
                                + meu_new_block(i, Neigh);

                meu(i, Neigh) = meu_new_block(i, Neigh);
            end

            % --------------------------------------------------------------
            % Step 4: Early termination after each block-layer
            % --------------------------------------------------------------
            wwc     = double(Lambda <= 0);
            Syn_Est = mod(H * wwc', 2)';
            if all(Syn_Est == Syn)
                flag = true;
                return;
            end
        end
    end

    % Return final hard decision if max iterations reached
    wwc = double(Lambda <= 0);
end