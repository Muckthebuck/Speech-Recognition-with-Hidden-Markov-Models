function features = load_features(filename, num_coeffs)

fid = fopen(filename, 'r');
features = fread(fid, 'double');
features = reshape(features, [], num_coeffs);
features = features(3:end, :);
fclose(fid);

end