function dataset = prepare_data(src_folder, dst_folder, num_coeffs, train_ratio, ext)

features_suffix = sprintf('_%d.bin', num_coeffs);

% Read the subfolders, exclude '.' (current directory) and '..' (parent directory)
subfolders = dir(src_folder);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));
num_classes = length(subfolders);

% The first column of the dataset gives the name of each class
dataset = cell(num_classes, 2);
dataset(:, 1) = {subfolders.name};
for i = 1:numel(dataset(:,1))
    dataset(i, 4) = {ceil(numel(dataset{i, 1})*2)};
end

% Duplicate the directory structure for feature map storage
if ~isfolder(dst_folder)
    status = mkdir(dst_folder);
    if status == 0
        fprintf("Failed to create folder %s\n", dst_folder);
        return
    end
end

% Read files and create features
for i = 1:num_classes
    cur_src_subfolder = subfolders(i);
    cur_dst_subfolder = strcat(dst_folder, '\', cur_src_subfolder.name);
    
    if ~isfolder(cur_dst_subfolder)
        status = mkdir(cur_dst_subfolder);
        if status == 0
            fprintf("Failed to create folder %s\n", cur_dst_subfolder);
            return
        end
    end

    % Read audio from the source directory
    wildcard = strcat('\*', ext);
    src_files = dir(strcat(cur_src_subfolder.folder, '\', cur_src_subfolder.name, wildcard));
    num_examples = length(src_files);

    examples = cell(num_examples, 1);
    for j = 1:num_examples
        cur_src_file = src_files(j);
        audio_filename = strcat(cur_src_file.folder, '\', cur_src_file.name);
        
        features_filename = strcat(cur_dst_subfolder, '\',...
            replace(cur_src_file.name, ext, features_suffix));
        examples{j} = features_filename;

        if ~isfile(features_filename)
            features = get_features(audio_filename, num_coeffs);
            fid = fopen(features_filename, 'w');
            fwrite(fid, features, 'double');
            fclose(fid);
        end
    end

    % Split into train and test sets
    [train_indices, test_indices] = train_test_split(num_examples, train_ratio);
    dataset{i, 2} = examples(train_indices);
    dataset{i, 3} = examples(test_indices);
end

end

function features = get_features(filename, num_coeffs)

% Read the audio file
[waveform, fs] = audioread(filename);

% Crop audio to 1 second or add zero-padding
if length(waveform) > fs
    waveform = waveform(1:fs);
elseif length(waveform) < fs
    waveform(end + 1, fs) = 0.0;
end

assert(length(waveform) == fs);

% If necessary, resample the waveform now
if fs ~= 48000
    [p, q] = rat(48000 / fs);
    waveform = resample(waveform, p, q);
end

% Compute MFCC
features = mfcc(waveform, fs, 'LogEnergy', 'ignore', 'NumCoeffs', num_coeffs);

end