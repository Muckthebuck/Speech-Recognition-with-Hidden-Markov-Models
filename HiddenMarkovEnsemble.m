classdef HiddenMarkovEnsemble < handle
    properties (SetAccess = private)
        NumClasses
        NumCoeffs
        Models
    end

    methods
        % Constructor
        function obj = HiddenMarkovEnsemble(num_classes, num_coeffs, num_states)
            % Set the size of the model
            obj.NumClasses = num_classes;
            obj.NumCoeffs = num_coeffs;

            obj.Models = cell(1, num_classes);
            for i = 1:num_classes
                obj.Models{i} = HiddenMarkovModel(num_coeffs, num_states{i});
            end
        end

        % Train the hidden markov models on the training dataset and return the
        % log likelihood achieved
        function llf = train(obj, samples, max_iterations)
            assert(length(samples) == obj.NumClasses);

            llf = zeros(1, obj.NumClasses);
            for i = 1:obj.NumClasses
                fprintf("Training class %d:\n", i);
                llf(i) = obj.Models{i}.train(samples{i}, max_iterations);
            end
        end

        % Run the models on a sample and choose the class according to the 
        % Maximum Likelihood principle
        function pred_label = run_models(obj, sample)
            llf = zeros(1, obj.NumClasses);
            for i = 1:obj.NumClasses
                llf(i) = obj.Models{i}.test(sample);
            end
            [~, pred_label] = max(llf);
        end

        % Test the HMM classifier and report results as [true_labels, pred_labels]
        function results = test(obj, samples)
            assert(length(samples) == obj.NumClasses);
            
            results = [];
            for i = 1:obj.NumClasses
                cur_samples = samples{i};
                for j = 1:length(cur_samples)
                    pred_label = obj.run_models(cur_samples{j});
                    results(:, end + 1) = 0;                %#ok<AGROW>
                    results(1, end) = i;                 
                    results(2, end) = pred_label;        
                end
            end
        end
    end
end