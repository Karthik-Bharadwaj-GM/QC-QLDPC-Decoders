function [Err_b,flag]= Soft_Dec_combine_NMS(H_css,p,S,max_itr,alpha)
% Normalized Quaternary Min-Sum Decoder for Quantum LDPC Codes
% alpha: normalization factor in (0,1]
% If not provided, defaults to 0.75
    if nargin < 5
        alpha = 0.75;
    end

    flag = false;
    [R, C_full] = size(H_css);
    C = C_full / 2;
    
    % Transform H_css to a quaternary matrix
    H =  H_css(:, 1:C) + 2* H_css(:, C+1:2*C);
    
    % Define log likelihood ratios and initial values
    gamma = log(p / (3 - 3 * p));
    Lb = [0, gamma, gamma, gamma];
    llr_v = max_exp(0,gamma)-max_exp(gamma,gamma);
    
    % Initialize neu and meu matrices
    neu = ones(R, C) * llr_v;
    neu(H == 0) = inf;
    meu = zeros(R, C, 4);

    % Iterative decoding process
    for w = 1:max_itr        
        %% Check node to variable node message update
        for i = 1:R
            [~, Neigh_Chk] = find(H(i, :));  % Find connected variable nodes
            neu_chk = neu(i, Neigh_Chk);
            
            for j = 1:length(neu_chk)
                    Minimum=sort(abs(neu_chk(1:end ~=j)));
                    if Minimum(1)==0
                        x=(-1)^(S(i))*prod(sign(neu_chk(1:end ~=j)))* 1;
                    else
                        if Minimum(1)==abs(neu_chk(j))
                         x=(-1)^(S(i))*prod(sign(neu_chk(1:end ~=j)))*Minimum(2);
                        else
                         x=(-1)^(S(i))*prod(sign(neu_chk(1:end ~=j)))*Minimum(1);
                        end
                    end

                % --- Normalization: scale check-node message by alpha ---
                x = alpha * x;
                
                % Compute potential new meu values
                new_meu_values = zeros(1, 4);
                if H(i, Neigh_Chk(j)) == 1
                    new_meu_values = [0, 0, -x, -x];
                elseif H(i, Neigh_Chk(j)) == 2
                    new_meu_values = [0, -x, 0, -x];
                elseif H(i, Neigh_Chk(j)) == 3
                    new_meu_values = [0, -x, -x, 0];
                end

                % Update only if the new values differ from current ones
                if ~isequal(squeeze(meu(i, Neigh_Chk(j), :))', new_meu_values)
                    meu(i, Neigh_Chk(j), :) = new_meu_values;
                end
            end
        end
        
        %% Variable node to check node message update and hard decision
        Err = zeros(1, C);
        Err_b = zeros(1, 2 * C);
        
        for i = 1:C
            [Neigh_var, ~] = find(H(:, i));  % Find connected check nodes
            
            % Aggregate meu values
            meu_var = meu(Neigh_var, i, :);
            
            % Hard decision: Find error based on Lb and meu sum
            [~, Err(i)] = max(Lb + sum(meu_var(1:end,:), 1));
            
            % Update Err_b based on quaternary hard decision
            if Err(i) == 1
                Err_b([i, C + i]) = [0, 0];
            elseif Err(i) == 2
                Err_b([i, C + i]) = [1, 0];
            elseif Err(i) == 3
                Err_b([i, C + i]) = [0, 1];
            else
                Err_b([i, C + i]) = [1, 1];
            end
            
            % Message passing to check nodes
            for j = 1:length(Neigh_var)
                meu_excluded = meu_var(1:end ~= j, :);  % Exclude jth neighbor
                
                % Compute updated neu value with log-sum-exp for stability
                x = Lb + sum(meu_excluded, 1);
                
                if H(Neigh_var(j), i) == 2
                    neu(Neigh_var(j), i) = max_exp(x(1), x(3)) - max_exp(x(2), x(4));
                elseif H(Neigh_var(j), i) == 1
                    neu(Neigh_var(j), i) = max_exp(x(1), x(2)) - max_exp(x(3), x(4));
                else
                    neu(Neigh_var(j), i) = max_exp(x(1), x(4)) - max_exp(x(2), x(3));
                end
            end
        end
        
        % Check stopping condition
        Til_S = mod(H_css * [Err_b(C + 1:2 * C), Err_b(1:C)]', 2);
        if all(S == Til_S')
            flag = true;
            Err = Err - 1;  % Adjust Err values to match original format
            break;
        end
    end
end

function [A] = max_exp(a, b)
    % Stable max-exp calculation
    A = max(a, b) + log(1 + exp(-abs(a - b)));
end