function logpdf = logmvnpdf(x, mu, sigma)

[V, D] = eig(sigma);
values = diag(D);
values_inv = 1.0 ./ values;
log_det = sum(log(values));

rank = length(values);
deviation = x - mu;

tmp_1 = V .* sqrt(values_inv).';
tmp_2 = tmp_1 * deviation.';
innerprod = tmp_2.' * tmp_2;

logpdf = -0.5 * (rank * log(2.0 * pi) + innerprod + log_det);

end