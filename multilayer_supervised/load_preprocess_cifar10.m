function [data_train, labels_train, data_test, labels_test] = load_preprocess_cifar10()
  data_train = []; labels_train = [];
  for i=1:5
    name = ['data_batch_', int2str(i), '.mat'];
    data = load(strcat(name));
    data_train = [data_train, double(data.data') / 255];
    labels_train = [labels_train; double(data.labels + 1)];
  end
  
  data6 = load('test_batch.mat');
  data_test = double(data6.data') / 255;
  labels_test = double(data6.labels + 1);