function [precision, recall, f1, accuracy] = classification_metrics(confusion_matrix)

precision = diag(confusion_matrix) ./ sum(confusion_matrix, 2);
recall = diag(confusion_matrix) ./ sum(confusion_matrix, 1).';
f1 = 2.0 * precision .* recall ./ (precision + recall);
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix, 'all');

end