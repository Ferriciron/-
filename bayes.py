#coding=gbk
import pandas as pd
import numpy as np
import math
data_df = pd.read_csv('temporary.csv')
def bayesianPredictOneSample(input_data):
  prior_prob = calulateClassPriorProb(trainset,train_info)
  train_Summary_by_class = summarizeByClass(trainset)
  classprob_dict = calculateClassProb(input_data,train_Summary_by_class)
  result = {}
  for class_value,class_prob in classprob_dict.np.items():
      p = class_prob*prior_prob[class_value]
      result[class_value] = p
  return max(result,key=result.get)
def splitData(data_list,ratio):
  train_size = int(len(data_list)*ratio)
  np.random.shuffle(data_list)
  train_set = data_list[:train_size]
  test_set = data_list[train_size:]
  return train_set,test_set

data_list = np.array(data_df).tolist()
trainset,testset = splitData(data_list,ratio = 0.7)
print('Split {0} samples into {1} train and {2} test samples '.format(len(data_df), len(trainset), len(testset)))
def seprateByClass(dataset):
  seprate_dict = {}
  info_dict = {}
  for vector in dataset:
      if vector[-1] not in seprate_dict:
          seprate_dict[vector[-1]] = []
          info_dict[vector[-1]] = 0
      seprate_dict[vector[-1]].append(vector)
      info_dict[vector[-1]] +=1
  return seprate_dict,info_dict

train_separated,train_info = seprateByClass(trainset)
def calulateClassPriorProb(dataset,dataset_info):
  dataset_prior_prob = {}
  sample_sum = len(dataset)
  for class_value, sample_nums in dataset_info.items():
      dataset_prior_prob[class_value] = sample_nums/float(sample_sum)
  return dataset_prior_prob
def mean(list):
  list = [float(x) for x in list] #字符串转数字
  return sum(list)/float(len(list))
# 方差
def var(list):
  list = [float(x) for x in list]
  avg = mean(list)
  var = sum([math.pow((x-avg),2) for x in list])/float(len(list)-1)
  return var
# 概率密度函数
def calculateProb(x,mean,var):
    exponent = math.exp(math.pow((x-mean),2)/(-2*var))
    p = (1/math.sqrt(2*math.pi*var))*exponent
    return p
def summarizeAttribute(dataset):
    dataset = np.delete(dataset,-1,axis = 1) # delete label
    summaries = [(mean(attr),var(attr)) for attr in zip(*dataset)]
    return summaries
def summarizeByClass(dataset):
  dataset_separated,dataset_info = seprateByClass(dataset)
  summarize_by_class = {}
  for classValue, vector in dataset_separated.items():
      summarize_by_class[classValue] = summarizeAttribute(vector)
  return summarize_by_class
def calculateClassProb(input_data,train_Summary_by_class):
  prob = {}
  for class_value, summary in train_Summary_by_class.items():
      prob[class_value] = 1
      for i in range(len(summary)):
          mean,var = summary[i]
          x = input_data[i]
          p = calculateProb(x,mean,var)
      prob[class_value] *=p
      def bayesianPredictOneSample(input_data):
        prior_prob = calulateClassPriorProb(trainset,train_info)
        train_Summary_by_class = summarizeByClass(trainset)
        classprob_dict = calculateClassProb(input_data,train_Summary_by_class)
        result = {}
        for class_value,class_prob in classprob_dict.items():
         p = class_prob*prior_prob[class_value]
         result[class_value] = p
        return max(result,key=result.get)
input_vector = testset[1]
input_data = input_vector[:-1]
result = bayesianPredictOneSample(input_data)
print("the sameple is predicted to class: {0}.".format(result))
def calculateAccByBeyesian(dataset):
  correct = 0
  for vector in dataset:
      input_data = vector[:-1]
      label = vector[-1]
      result = bayesianPredictOneSample(input_data)
      if result == label:
          correct+=1
  return correct/len(dataset)

acc = calculateAccByBeyesian(testset)