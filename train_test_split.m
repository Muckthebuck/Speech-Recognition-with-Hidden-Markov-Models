function [train_indices, test_indices] = train_test_split(num_examples, ratio)

% Create a random permutation of the example indices
indices = randperm(num_examples);

% Find size of the train set
train_size = ceil(num_examples * ratio);

% Select training indices
train_indices = indices(1:train_size);
test_indices = indices((train_size + 1):end);

end