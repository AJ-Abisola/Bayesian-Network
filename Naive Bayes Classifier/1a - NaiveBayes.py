#############################################################################
# 1a - NaiveBayes.py
#
# Implements Naive Bayes Classifier to answer probability queries
# This code has been written by Ayokunle J. Abisola
# Contact: 26736158@students.lincoln.ac.uk
############################################################################


import pandas as pd
import sys
import time
import numpy as np
from sklearn import metrics

class Query:
    
    """" A probabilistic query class
    -----------------------------------
    
    To analyze a data and calculate all possible conditional and joint probability, given a target column
    
    
    """
    true_values = []
    predicted_values = []
    probability_values = []
    total_inference = 0
    inference_time = 0


    def __init__(self,train,test):
        
        self.train = self.read(train_train)
        self.test = self.read(test_train)
        self.full_test = self.read(test_train)
        self.test = self.test.drop(self.test.columns[-1], axis=1)
        self.columns = list(self.train.columns)
        self.evidence_columns = self.columns[0:-1]
        self.predicted = self.columns[-1]
        self.possible_outcome = list(self.train[self.predicted].unique())
        
        self.parameters_estimation = {}
        self.model_prediction = []

        self.true_values = list(self.full_test[self.predicted])
        self.count = 0
        self.training_time = 0
        
        
    def read(self,file):
        read = pd.read_csv(file)
        return read
   

    #To calculate maximum likelihood in the case of zero probability
    def maximum_likelihood(self,prob, countt):
        if prob<=0:
            prob = (countt+1)/(num+len(evidence_variables))
        return prob


    #To make an estimation of all possible parameters, assuming they are all independent of each other
    def estimation(self):
        
        self.training_time = time.time()
        global num
        global evidence_variables
        
        for outcome in self.possible_outcome:
            num = self.train[self.predicted].value_counts()[outcome]
            den = len(self.train[self.predicted])
            outcome_probability = num/den
            print(f"\n Probability of getting {self.predicted} as a {outcome} is {outcome_probability} \n")
            self.parameters_estimation['P('+str(self.predicted)+'='+str(outcome)+')'] = outcome_probability

            for evidence in self.evidence_columns:
                evidence_variables = list(self.train[evidence].unique())
                print(f"Considering: The variables for evidence {evidence} are {evidence_variables}")

                for variable in evidence_variables:
                    count = self.train[evidence].value_counts()[variable]
                    evidence_probability = count/den
                    key = 'P('+str(evidence)+'='+str(variable)+')'
                    if key not in self.parameters_estimation.keys():
                        self.parameters_estimation['P('+str(evidence)+'='+str(variable)+')'] = evidence_probability
                    print(f"P({evidence}={variable}) = {evidence_probability}")

                    variable_count = len(self.train[(self.train[evidence]==variable) & (self.train[self.predicted]==outcome)])
                    variable_probability = variable_count/num
                    #Using Maximum Likelihood
                    variable_probability = self.maximum_likelihood(variable_probability,variable_count)
                    print(f"P({evidence}={variable} | {self.predicted}={outcome}) = {variable_probability}")
                    self.parameters_estimation['P('+str(evidence)+'='+str(variable)+'|'+str(self.predicted)+'='+str(outcome)+')'] = variable_probability

        self.training_time = time.time() - self.training_time
                    
    #Checks if a value is of type string or not
    def check_type(self,entry):
        if type(entry[0]) == str:
            pass
        else:
            c = 0
            for _ in entry:
                entry[c] = str(_)
                c += 1
    
    
    #To get user inputs for the probabiity queries
    def enter_query(self):
        
        self.target_list=[]
        chosen = []
        self.chosen_variable_list = []
        evidence = 0
        target = None

        print(f"The available outcome is {self.predicted} with values {self.possible_outcome}")
        print("Type 'all' to calculate for all possible outcome")

        
        self.check_type(self.possible_outcome)
        while target not in self.possible_outcome:
            target = input("Please choose a target:")
            if target not in self.possible_outcome:
                if target != "all":
                    print("Invalid entry \n")
                else:
                    break
        print(f"You chose {target}\n")
        
        if target == "all":
            self.target_list.extend(self.possible_outcome)
        else:
            self.target_list.append(target)


        print(f"The available evidences are {self.evidence_columns}")

        self.check_type(self.evidence_columns)
        
        while evidence not in self.evidence_columns:
            print("Type in the evidences you want to evaluate one after the other, press enter before typing the next one")
            print( "Type 'done' when you have selected all evidences you want")
            while evidence != "done":
                evidence = input("Please choose an evidence:")
                if evidence not in self.evidence_columns:
                    if evidence != "done":
                        print("Invalid entry/n")
                else:
                    chosen.append(evidence)
            break

        print(f"Your chosen evidence(s) are {chosen}")
        
        for evid in chosen:
            var = list(self.train[evid].unique())
            self.check_type(var)
            print(f"\nPick a variable for {evid}")
            print(f"{evid} has these variables: {var} \n")
            chosen_variable = input("Please type a variable:")
            while chosen_variable not in var:
                print("Invalid \n")
                chosen_variable = input("Please type a variable:")
            text = str(evid)+"="+chosen_variable
            self.chosen_variable_list.append(text)
        
    
    def multiply_prob(self,checks):
        result = 1
        for check in checks:
            result = result * check
        return result
    
    
    #To calculate the probabiity queries
    def get_query(self):
        
        print("This calculates the conditional probability of getting a target given different evidences using Naive Bayes")
        print("--------------------------------------------------------------------------------")
        time.sleep(3)
        self.estimation()
        self.enter_query()
        queries = []
        
        for target in self.target_list:
            chosen_probabilities = []
            check_target = "P("+str(self.predicted)+"="+str(target)+")"

            for v in self.chosen_variable_list:
                check_num = "P("+str(v)+"|"+str(self.predicted)+"="+str(target)+")"
                chosen_probabilities.append(self.parameters_estimation[check_num])

            chosen_probabilities.append(self.parameters_estimation[check_target])
            query_result = self.multiply_prob(chosen_probabilities)
            chosen_probabilities.pop(-1)
            queries.append(query_result)


        print(f"Probability query result is {queries}")
        
    
    #To access accuracy of model on test data   
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


    #To grab all possible probabilities(for new instances) in a test data
    def all_probabilities(self,rows):
        
        self.inference_time = time.time()
        print(rows)
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
        print(f"UNNORMALIZED PROBABILITY: {unnormalized_prob}")
        print(f"NORMALIZED PROBABILITY: {normalized_prob}")
        print("\n")
        
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
    
    
    #To calculate the probabilities on the test data
    def get_probabilities(self):
        
        print("This calculates the Unnormalized and Normalized probability of the entire given test data using Naive Bayes")
        print("--------------------------------------------------------------------------------")
        time.sleep(3)
        self.estimation()
        time.sleep(3)
        self.test.apply(lambda x:self.all_probabilities(x), axis=1)
        self.check_performance()
        
        
        


try:
    
    train_train = sys.argv[1]
    test_train = sys.argv[2]
    Query_object = Query(train_train,test_train)
    print("Welcome \n")
    print("What would you like to calculate?")
    print("1. Probability query 2. Full data test")

    test_entry = 0
    options = ["1","2"]
    while test_entry not in options:
        test_entry = input("Enter your option number:")
        if test_entry not in options:
            print("Invalid Entry")
    if test_entry == "1":
        Query_object.get_query()
    else:
        Query_object.get_probabilities()
        
except:
    print("Error")
    print("Correct format is : Python Assessment.py train.csv test.csv")