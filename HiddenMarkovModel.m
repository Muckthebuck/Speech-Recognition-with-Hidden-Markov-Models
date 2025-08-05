classdef HiddenMarkovModel < handle
    properties (SetAccess = private)
        % Fixed properties of the HMM
        NumStates
        NumCoeffs

        % Trainable parameters
        Prior
        LogPrior
        TransitionMatrix
        LogTransitionMatrix
        Means
        Covariances
    end

    methods
        % Constructor
        function obj = HiddenMarkovModel(num_coeffs, num_states)
            % Set the size of the model
            obj.NumStates = num_states;
            obj.NumCoeffs = num_coeffs;
            
            % Initialize with dummy parameters
            obj.reset(zeros(1, num_coeffs), eye(num_coeffs));
        end
        
        % Reset the trainable parameters
        function reset(obj, dataset_mean, dataset_covariance)
            % Use a randomized prior and transition matrix
            obj.Prior = rand(1, obj.NumStates);
            obj.Prior = obj.Prior ./ sum(obj.Prior);
            obj.LogPrior = log(obj.Prior);

            obj.TransitionMatrix = rand(obj.NumStates);
            obj.TransitionMatrix = obj.TransitionMatrix ./ sum(obj.TransitionMatrix, 2);
            obj.LogTransitionMatrix = log(obj.TransitionMatrix);

            % Initially assume state distributions are equal to the overall distribution
            obj.Means = repmat(dataset_mean.', 1, obj.NumStates);
            obj.Covariances = repmat(dataset_covariance, 1, 1, obj.NumStates);
        end

        % Compute the log probability of an observation in a given state
        function log_prob = log_obs_prob(obj, obs, state)
            log_prob = logmvnpdf(obs, obj.Means(:, state).', obj.Covariances(:, :, state));
        end

        % Forward recursion algorithm
        function [log_a, log_b] = forward(obj, obs)
            N = size(obs, 1);

            % Storage for the sequences a_t = p(x_t|y^t) and b_t = p(x_t|y^{t-1})
            log_a = zeros(N, obj.NumStates);
            log_b = zeros(N, obj.NumStates);

            log_b(1, :) = obj.LogPrior;

            % Forward recursion loop
            for t = 1:N
                % Measurment update
                for k = 1:obj.NumStates
                    % Compute log probability of the observation in state k
                    log_prob = obj.log_obs_prob(obs(t, :), k);

                    % Update log probabilities
                    log_a(t, k) = log_b(t, k) + log_prob;
                end

                % Normalize probabilities
                log_a(t, :) = log_a(t, :) - logsumexp(log_a(t, :), 2);

                % Time update
                if t < N
                    for k = 1:obj.NumStates
                        log_b(t + 1, k) = logsumexp(log_a(t, :).' + obj.LogTransitionMatrix(:, k), 1);
                    end
                end
            end
        end

        % Backward recursion algorithm
        function [log_y, log_z] = backward(obj, log_a, log_b)
            N = size(log_a, 1);

            % Storage for sequences y_t = p(x_t|y^n) and z_t = p(x_t, x_{t+1}|y^n)
            log_y = zeros(N, obj.NumStates);
            log_z = zeros(N - 1, obj.NumStates, obj.NumStates);
            
            % Initialization using results of the forward recursion
            log_y(N, :) = log_a(N, :);
            
            % Backward recursion loop
            for t = (N - 1):-1:1
                for k = 1:obj.NumStates
                    log_z(t, k, :) = log_y(t + 1, :) + log_a(t, k)...
                        + obj.LogTransitionMatrix(k, :) - log_b(t + 1, :);
                    log_y(t, k) = logsumexp(log_z(t, k, :), 3);
                end
            end
        end

        % Calculate the log likelihood of an observation given the results of the 
        % forward-backward algorithm
        function llf = log_likelihood(obj, obs, y, z)
            N = size(obs, 1);

            term1 = sum(y(1, :) .* obj.LogPrior);

            term2 = 0.0;
            for t = 1:(N - 1)
                term2 = term2 + sum(squeeze(z(t, :, :)) .* obj.LogTransitionMatrix, 'all');
            end

            term3 = 0.0;
            for t = 1:N
                for k = 1:obj.NumStates
                    term3 = term3 + y(t, k) * obj.log_obs_prob(obs(t, :), k);
                end
            end
            
            llf = term1 + term2 + term3;
        end

        % Test a sample by calculating its log likelihood
        function llf = test(obj, sample)
            obs = load_features(sample, obj.NumCoeffs);

            % Run forward-backward algorithm
            [log_a, log_b] = obj.forward(obs);
            [log_y, log_z] = obj.backward(log_a, log_b);

            y = exp(log_y);
            z = exp(log_z);

            % Return log likelihood
            llf = obj.log_likelihood(obs, y, z);
        end

        % Train the model on a series of samples
        function llf = train(obj, samples, max_iterations)
            M = length(samples);

            % Load observations into memory
            obs = [];
            for i = 1:M
                obs(i, :, :) = load_features(samples{i}, obj.NumCoeffs);         %#ok<AGROW>
            end

            % Find dataset mean and covariance, then reset the model
            [mu, sigma] = get_stats(obs, obj.NumCoeffs);
            obj.reset(mu, sigma);

            N = size(obs, 2);
            L = size(obs, 3);

            reverse_string = '';

            prev_llf = 0.0;
            for iteration = 1:max_iterations
                % Run forward-backward algorithm for each observation
                log_a = zeros(M, N, obj.NumStates);
                log_b = zeros(M, N, obj.NumStates);
    
                log_y = zeros(M, N, obj.NumStates);
                log_z = zeros(M, N - 1, obj.NumStates, obj.NumStates);
    
                for i = 1:M
                    [tmp_a, tmp_b] = obj.forward(squeeze(obs(i, :, :)));
                    [tmp_y, tmp_z] = obj.backward(tmp_a, tmp_b);
    
                    log_a(i, :, :) = tmp_a;
                    log_b(i, :, :) = tmp_b;
                    log_y(i, :, :) = tmp_y;
                    log_z(i, :, :, :) = tmp_z;
                end

                y = exp(log_y);
                z = exp(log_z);

                % Compute log likelihood
                llf = 0.0;
                for i = 1:M
                    llf = llf + obj.log_likelihood(squeeze(obs(i, :, :)),...
                        squeeze(y(i, :, :)), squeeze(z(i, :, :, :)));
                end
                msg = sprintf('Iteration %3d, NLL: %.3e\n', iteration, -1.0 * llf);
                fprintf([reverse_string, msg]);
                reverse_string = repmat('\b', 1, length(msg));

                if abs(llf - prev_llf) / abs(llf) < 1e-6
                    break
                end
                prev_llf = llf;
                
                % Update prior
                obj.LogPrior = squeeze(logsumexp(log_y(:, 1, :), 1)).' - log(M);
                obj.Prior = exp(obj.LogPrior);
                
                % Update transition probability
                for k = 1:obj.NumStates
                    obj.LogTransitionMatrix(k, :) = logsumexp(log_z(:, :, k, :), [1, 2])...
                        - logsumexp(log_y(:, 1:(N - 1), k), [1, 2]);
                end
                obj.TransitionMatrix = exp(obj.LogTransitionMatrix);
    
                % Update mean for each state
                for k = 1:obj.NumStates
                    den = exp(logsumexp(log_y(:, :, k), [1, 2])) + 1e-12;
                    num = zeros(1, L);
                    for l = 1:M
                        cur_obs = squeeze(obs(l, :, :));
                        num = num + sum(y(l, :, k).' .* cur_obs, 1);
                    end
                    obj.Means(:, k) = num ./ den;
                end
    
                % Update covariance for each state
                for k = 1:obj.NumStates
                    den = exp(logsumexp(log_y(:, :, k), [1, 2])) + 1e-12;
                    num = zeros(L);
                    for l = 1:M
                        for t = 1:N
                            deviation = squeeze(obs(l, t, :)) - obj.Means(:, k);
                            outerprod = deviation * deviation.';
                            num = num + y(l, t, k) * outerprod;
                        end
                    end
                    obj.Covariances(:, :, k) = (num ./ den) + 1e-6 * eye(L);
                end
            end
        end
    end
end

% Compute log-sum-exp of a vector of log probabilities x
function y = logsumexp(x, dim)

c = max(x, [], dim);
if c == -Inf
    y = -Inf;
else
    y = c + log(sum(exp(x - c), dim));
end

end

% Calculate sample mean and covariance of the class samples
function [mean, cov] = get_stats(obs, num_coeffs)

M = size(obs, 1);

% Compute mean
mean = zeros(1, num_coeffs);
num_samples = 0;
for i = 1:M
    cur_obs = squeeze(obs(i, :, :));
    num_samples = num_samples + size(cur_obs, 1);
end
mean = mean ./ num_samples;

% Compute covariance
cov = zeros(num_coeffs);
for i = 1:M
    cur_obs = squeeze(obs(i, :, :));
    for k = 1:size(cur_obs, 1)
        cov = cov + (cur_obs(k, :).' * cur_obs(k, :));
    end
end
cov = cov ./ (num_samples - 1);
    
end

