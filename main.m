clearvars;
close all;

num_coeffs = 14;
num_states = 9;
train_ratio = 0.75;
train_fresh = true;

constant = false;

% Data from lecturer
dataset_1 = prepare_data('.\audio_1', '.\features_1', num_coeffs, train_ratio, '.m4a');
% Data recorded individually
dataset_2 = prepare_data('.\audio_2', '.\features_2', num_coeffs, train_ratio, '.wav');

num_classes = size(dataset_1, 1);

% Combine the datasets into dataset 3
dataset_3 = dataset_1;
for i = 1:num_classes
    dataset_3{i, 2} = [dataset_1{i, 2}; dataset_2{i, 2}];
    dataset_3{i, 3} = [dataset_1{i, 3}; dataset_2{i, 3}];
end

if constant
    num_states_cell_1 = num2cell(ones(1,numel(dataset_1(:,1)))*num_states);
    num_states_cell_2 = num2cell(ones(1,numel(dataset_1(:,1)))*num_states);
    num_states_cell_3 = num2cell(ones(1,numel(dataset_1(:,1)))*num_states);
else
    num_states_cell_1 =  dataset_1(:,4);
    num_states_cell_2 =  dataset_2(:,4);
    num_states_cell_3 =  dataset_3(:,4);
end


% Train the HMP on the recordings from lecturer
if train_fresh
    hmm_1 = HiddenMarkovEnsemble(num_classes, num_coeffs,  num_states_cell_1);
    llf_1 = hmm_1.train(dataset_1(:, 2), 500);
    save('model_1.mat', 'hmm_1');

  %  hmm_2 = HiddenMarkovEnsemble(num_classes, num_coeffs, num_states_cell_1);
  %  llf_2 = hmm_2.train(dataset_2(:, 2), 500);
  % save('model_2.mat', 'hmm_2');

    hmm_3 = HiddenMarkovEnsemble(num_classes, num_coeffs, num_states_cell_1);
    llf_3 = hmm_3.train(dataset_3(:, 2), 500);
    save('model_3.mat', 'hmm_3');
else
    load('model_1.mat', 'hmm_1');
    %load('model_2.mat', 'hmm_2');
    load('model_3.mat', 'hmm_3');
end

% Evaluate the first model, trained on provided recordings only, on the recordings from lecturer
fprintf("\nProvided data, trained with provided recordings:\n");
fprintf("================================================\n");
test_results = hmm_1.test(dataset_1(:, 3));

figure;
confusion_matrix = confusionmat(test_results(1, :), test_results(2, :));
confusionchart(confusion_matrix);

[precision, recall, f1, accuracy] = classification_metrics(confusion_matrix);
for i = 1:num_classes
    fprintf("Class: %-10s  Precision: %.3f  Recall: %.3f  F1 Score: %.3f\n",...
        dataset_1{i, 1}, precision(i), recall(i), f1(i));
end
fprintf("Overall Accuracy: %.3f\n", accuracy);

% Evaluate the first model on the custom recordings
fprintf("\nCustom data, trained with provided recordings:\n");
fprintf("==============================================\n");
test_results = hmm_1.test(dataset_2(:, 3));

figure;
confusion_matrix = confusionmat(test_results(1, :), test_results(2, :));
confusionchart(confusion_matrix);

[precision, recall, f1, accuracy] = classification_metrics(confusion_matrix);
for i = 1:num_classes
    fprintf("Class: %-10s  Precision: %.3f  Recall: %.3f  F1 Score: %.3f\n",...
        dataset_2{i, 1}, precision(i), recall(i), f1(i));
end
fprintf("Overall Accuracy: %.3f\n", accuracy);

% Evaluate the third model, trained with all recordings, on the custom recordings only
fprintf("\nCustom data, trained with all recordings:\n");
fprintf("=========================================\n");
test_results = hmm_3.test(dataset_2(:, 3));

figure;
confusion_matrix = confusionmat(test_results(1, :), test_results(2, :));
confusionchart(confusion_matrix);

[precision, recall, f1, accuracy] = classification_metrics(confusion_matrix);
for i = 1:num_classes
    fprintf("Class: %-10s  Precision: %.3f  Recall: %.3f  F1 Score: %.3f\n",...
        dataset_2{i, 1}, precision(i), recall(i), f1(i));
end
fprintf("Overall Accuracy: %.3f\n", accuracy);

% Evaluate the third model on all recordings
fprintf("\nAll data, trained with all recordings:\n");
fprintf("======================================\n");
test_results = hmm_3.test(dataset_3(:, 3));

figure;
confusion_matrix = confusionmat(test_results(1, :), test_results(2, :));
confusionchart(confusion_matrix);

[precision, recall, f1, accuracy] = classification_metrics(confusion_matrix);
for i = 1:num_classes
    fprintf("Class: %-10s  Precision: %.3f  Recall: %.3f  F1 Score: %.3f\n",...
        dataset_3{i, 1}, precision(i), recall(i), f1(i));
end
fprintf("Overall Accuracy: %.3f\n", accuracy);


