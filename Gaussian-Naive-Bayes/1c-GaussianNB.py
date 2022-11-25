#############################################################################
# 1c-GaussianNB.py
#
# Implements Naive Bayes Classifier to handle continuous data
# This code has been written by Ayokunle J. Abisola
# Contact: 26736158@students.lincoln.ac.uk
############################################################################


import pandas as pd
import math
import time
import numpy as np
from sklearn import metrics
import sys



class Gaussian():
    
    true_values = []
    predicted_values = []
    probability_values = []
    continuous = False
    total_inference = 0
    parameters_estimation = {}
    LL = 0
    penalty = 0
    
    def __init__(self,train,test):
        
        self.train = self.read(train)
        self.test = self.read(test)
        self.full_test = self.read(test)
        self.test = self.test.drop(self.test.columns[-1], axis=1)
        self.columns = list(self.train.columns)
        self.evidence_columns = self.columns[0:-1]
        self.predicted = self.columns[-1]
        self.possible_outcome = list(self.train[self.predicted].unique())
        self.true_values = list(self.full_test[self.predicted])
        self.check_continuous()
        
        
        if self.continuous:
            
            print("Continuous Data...")
            #train
            self.training_time = time.time()
            self.prior()
            self.eval_mean_std()
            self.training_time = time.time()- self.training_time
            
            self.estimation()

            #testing
            print("Testing on test data\n")
            self.count = 0
            self.test.apply(lambda x:self.test_gaussian(x), axis=1)
            
        else:
            print("Discrete Data")
            #train
            self.training_time = time.time()
            self.estimation()
            self.training_time = time.time() - self.training_time
            print("Testing on test data\n")
            self.count = 0
            self.test.apply(lambda x:self.test_discrete(x), axis=1)
        
        #performance
        time.sleep(1)
        self.check_performance()
        score = self.BIC()
        print(f"BIC is {score}")
        
        
    def read(self,file):
        read = pd.read_csv(file)
        return read

    
    def prior(self):
        self.outcome_probability = {}
        for _ in self.possible_outcome:
            _p = len(self.train[self.train[self.predicted]==_])/len(self.train)
            self.outcome_probability[_] = _p

    def check_continuous(self):
        for _ in self.columns:
    
            if len(self.train[_].unique()) >10:
                self.continuous = True
    
    def maximum_likelihood(self,prob, countt):
        if prob<=0:
            prob = (countt+1)/(num+len(evidence_variables))
        return prob
    
    
    def estimation(self):
        
        global num
        global evidence_variables
        
        for outcome in self.possible_outcome:
            num = self.train[self.predicted].value_counts()[outcome]
            den = len(self.train[self.predicted])
            outcome_probability = num/den
#             print(f"\n Probability of getting {self.predicted} as a {outcome} is {outcome_probability} \n")
            self.parameters_estimation['P('+str(self.predicted)+'='+str(outcome)+')'] = outcome_probability

            for evidence in self.evidence_columns:
                evidence_variables = list(self.train[evidence].unique())
#                 print(f"Considering: The variables for evidence {evidence} are {evidence_variables}")

                for variable in evidence_variables:
                    count = self.train[evidence].value_counts()[variable]
                    evidence_probability = count/den
                    key = 'P('+str(evidence)+'='+str(variable)+')'
                    if key not in self.parameters_estimation.keys():
                        self.parameters_estimation['P('+str(evidence)+'='+str(variable)+')'] = evidence_probability
#                     print(f"P({evidence}={variable}) = {evidence_probability}")

                    variable_count = len(self.train[(self.train[evidence]==variable) & (self.train[self.predicted]==outcome)])
                    variable_probability = variable_count/num
                    #Using Maximum Likelihood
                    variable_probability = self.maximum_likelihood(variable_probability,variable_count)
#                     print(f"P({evidence}={variable} | {self.predicted}={outcome}) = {variable_probability}")
                    self.parameters_estimation['P('+str(evidence)+'='+str(variable)+'|'+str(self.predicted)+'='+str(outcome)+')'] = variable_probability

    
    def test_discrete(self,rows):
        
        self.inference_time = time.time()
        unnormalized_prob = {}
        normalized_prob = {}
        
        for out in self.possible_outcome:
            count = 0
            unnormalized = self.parameters_estimation['P('+str(self.predicted)+'='+str(out)+')']
            for column in rows:
                statement = 'P('+str(self.evidence_columns[count])+'='+str(column)+'|'+str(self.predicted)+'='+str(out)+')'
                value = self.parameters_estimation[statement]
                unnormalized = value * unnormalized
                count += 1
            unnormalized_prob[out] = unnormalized

        for possibility in unnormalized_prob.keys():
            normalized_prob[possibility] = unnormalized_prob[possibility]/sum(unnormalized_prob.values())
        # print(f"UNNORMALIZED PROBABILITY: {unnormalized_prob}")
        # print(f"NORMALIZED PROBABILITY: {normalized_prob}")
        # print("\n")
        
        target = self.true_values[self.count]
        needed_prob = normalized_prob[target]
        
        if target in ['no', '0', 0]:
                needed_prob = 1-needed_prob
        self.probability_values.append(needed_prob)
        max_val = max(normalized_prob.values())
        prediction = list(normalized_prob.values()).index(max_val)
        prediction = list(normalized_prob.keys())[prediction]
        self.predicted_values.append(prediction)
        
        self.count +=1
        
        self.inference_time = time.time()- self.inference_time
        self.total_inference += self.inference_time
    
    
    def eval_mean_std(self):
        self.stds = {}
        self.means = {}

        for _ in self.possible_outcome:
            mean = dict(self.train[self.train[self.predicted] == _].drop(columns = [self.predicted]).mean())
            std = dict(self.train[self.train[self.predicted] == _].drop(columns = [self.predicted]).std())
            self.means[_] = mean
            self.stds[_] = std

        print(f"STD values are: \n {self.stds}")
        print(f"Mean Values are: \n {self.means}")


    def gaussian(self,x,mean,std):

        power_val = -0.5*math.pow((x-mean)/std, 2)
        probability = (1/(std*math.sqrt(2*math.pi))) * math.exp(power_val)
        return probability


    def test_gaussian(self,rows):
        
        self.inference_time = time.time()
#         print(rows)
        unnormalized = {}
        normalized = {}
        rows = dict(rows)
        for _ in self.possible_outcome:
            probability = self.outcome_probability[_]

            for row in rows:
                mean = self.means[_][row]
                std = self.stds[_][row]
                x = rows[row]
                prob = self.gaussian(x,mean,std)

                probability = probability*prob
            unnormalized[_] = probability
        
        for possibility in unnormalized.keys():
            normalized[possibility] = unnormalized[possibility]/sum(unnormalized.values())
        
        target = self.true_values[self.count]
        needed_prob = normalized[target]
        
        if target in ['no', '0', 0]:
                needed_prob = 1-needed_prob
        self.probability_values.append(needed_prob)
        max_val = max(normalized.values())
        prediction = list(normalized.values()).index(max_val)
        prediction = list(normalized.keys())[prediction]
        self.predicted_values.append(prediction)

#         print(f"UNNORMALIZED PROBABILITY: {unnormalized}")
#         print(f"NORMALIZED PROBABILITY: {normalized}")
        self.count +=1
        
        self.inference_time = time.time()- self.inference_time
        self.total_inference += self.inference_time

    
    def log_likelihood_score(self,rows):
        
        e_count = 0
        for _ in rows:
            evid = self.columns[e_count]
            if evid != self.predicted:
                value = _
                target = str(rows[-1]).split(".")[0]
                key = "P("+str(evid)+"="+str(value)+"|"+str(self.predicted)+"="+target+")"
                if key in self.parameters_estimation.keys():
                    self.LL = self.LL + math.log(self.parameters_estimation[key])
                else:
                    value = str(_).split(".")[0]
                    key = "P("+str(evid)+"="+str(value)+"|"+str(self.predicted)+"="+target+")"
                    self.LL = self.LL + math.log(self.parameters_estimation[key])
                
            else:
                target = str(rows[-1]).split(".")[0]
                key = "P("+str(self.predicted)+"="+target+")"
                self.LL = self.LL + math.log(self.parameters_estimation[key])
                
            e_count+=1
                
        return self.LL
                
                
    def BIC(self):
        l_score = self.train.apply(lambda x: self.log_likelihood_score(x), axis =1)
        
        print(f"log likelihood score is {l_score.min()}")

        for evid in list(self.train.columns):
            p = len(self.train[evid].unique())
            n = len(self.train)
            evid_penalty = (math.log(n)*p)/2
            self.penalty = self.penalty+ evid_penalty

        return l_score.min() - self.penalty
    
    
    def check_performance(self):
        
        print(f"True value: \n {self.true_values}")
        print(f"Predicted Values: \n {self.predicted_values}")
        print(f"Probability values: \n {self.probability_values}")
        
        divergence_true = np.asarray(self.true_values)+0.00001 
        divergence_prob = np.asarray(self.probability_values)+0.00001
        
        class_accuracy = metrics.accuracy_score(self.true_values, self.predicted_values)
        bal_accuracy = metrics.balanced_accuracy_score(self.true_values, self.predicted_values)
        fpr, tpr, _ = metrics.roc_curve(self.true_values, self.probability_values, pos_label=1)
        area_under_curve = metrics.auc(fpr, tpr)
        brier_score = metrics.brier_score_loss(self.true_values,self.probability_values)
        kl_divergence = np.sum(divergence_true*np.log(divergence_true/divergence_prob))
        print("PERFORMANCE metrics:")
        print(f"Classification Accuracy= {class_accuracy}")
        print(f"Balanced Accuracy= {bal_accuracy}")
        print(f"Area Under Curve="+str(area_under_curve))
        print(f"Brier Score= {brier_score}")
        print(f"KL Divergence= {kl_divergence}")
        print(f"Training time = {self.training_time}")
        print(f"Inference time = {self.total_inference}")
    
    
    
try:
    
    train = sys.argv[1]
    test = sys.argv[2]
    Gaussian_object = Gaussian(train,test)
    
        
except:
    print("Error")
    print("Correct format is : Python Assessment.py train.csv test.csv")