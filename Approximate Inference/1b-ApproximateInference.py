#############################################################################
# 1b-ApproximateInference.py
#
# Implements the approximate inference algorithms - Rejection sampling and likelihood weighting
# This code has been written by Ayokunle J. Abisola
# Contact: 26736158@students.lincoln.ac.uk
############################################################################



import random
import ast
import numpy as np
import pandas as pd
import sys
import time


class inference():
    
    
    
    def __init__(self,train,test,cpt_file,structure_file,query,size):
        
        self.train = self.read(train)
        self.test = self.read(test)
        self.cpt = self.read_cpt(cpt_file)
        self.structure = self.read_structure(structure_file)
        self.size = int(size)
        self.query = query
        self.rejection_sample_df = ""
        self.likelihood_sample_df = ""
        self.rejection_inference_time = 0
        self.likelihood_inference_time = 0
        
        self.check_query()
        
        self.get_rejection_probability()
        print("Rejection training time ="+str(self.rejection_inference_time))
        
        self.get_likelihood_probability()
        print("Likelihood Training time ="+str(self.likelihood_inference_time))
        
        
        
    def check_query(self):
        self.query_evid = {}
        for _ in self.query.split("|")[1].split(","):
            self.query_evid[_.split("=")[0]] = _.split("=")[1]
        self.query_main = self.query.split("|")[0]

    
    
    def read(self,file):
        read = pd.read_csv(file)
        return read
    
    def read_structure(self,checking):
        file_list = []
        with open(checking) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line != '\n':
                    file_list.append(line.rstrip())
                    
        return file_list

    
    def read_cpt(self,checking):
        file_list = []
        with open(checking) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line != '\n':
                    structure_line = dict(ast.literal_eval(line.rstrip()))
                    file_list.append(structure_line)
                    
        return file_list


    def generate_holder(self):
        compare = {}
        for _ in list(self.train.columns):
            compare[_] = "x"

        return compare


    
 
    def rejection_get_sample(self):
        
        compare = self.generate_holder()

        compared = compare.copy()

        while True:
            for s in self.structure:

                if s == self.structure[0]:
                    app = []
                    unique_variables = list(self.train[s].unique())
                    picked_variable = random.choice(unique_variables)
                    if s in list(self.query_evid.keys()):
                        if self.query_evid[s] == str(picked_variable):
                            compared[s] = picked_variable
                        else:
                            compared = "rejected"
                            self.rejection_count +=1
                            break
                    else:
                        compared[s] = picked_variable


                else:
                    parents = s.split("|")[1].split(",")

                    nested_parents =""

                    for _ in parents:
                        nested_parents = nested_parents+str(compared[_])+","

                    app = []
                    unique_variables = list(self.train[s.split("|")[0]].unique())
                    for v in unique_variables:
                        nnn = ""
                        nnn = str(v)+"|"+nested_parents
                        app.append(nnn)

                    picked_variable = random.choice(app)

                    if s.split("|")[0] in list(self.query_evid.keys()):
                        if self.query_evid[s.split("|")[0]] == str(picked_variable.split("|")[0]):
                            compared[s.split("|")[0]] = picked_variable.split("|")[0]
                        else:
                            compared = "rejected"
                            self.rejection_count +=1
                            break
                    else:
                        compared[s.split("|")[0]] = picked_variable.split("|")[0]


            break

        return compared



    def generate_rejection_samples(self):
        self.rejection_inference_time = time.time()
        samples = []
        self.rejection_count = 0
        for _ in range(self.size):
            sample = self.rejection_get_sample()
            if sample != "rejected":
                samples.append(sample)

        print(f"rejected {self.rejection_count} samples")
        print(f"good samples: {self.size - self.rejection_count}")
        self.rejection_sample_df = pd.DataFrame(samples)
        
        self.rejection_inference_time = time.time() - self.rejection_inference_time
        return self.rejection_sample_df

    def get_rejection_probability(self):
        print("Calculating using Rejection sampling")
        df = self.generate_rejection_samples()
        probabilities = {}
        for _ in list(df[self.query_main].unique()):
            up = len(df[df[self.query_main]== _])
            down = len(df)
            probabilities[_] = up/down

        print(f"Rejection sampling : {probabilities}")



    def likelihood_get_sample(self):
        
        compare = self.generate_holder()

        compared = compare.copy()
        compared["weight"] = "x" 
        weight = 1
        config_count = 0

        while True:
            for s in self.structure:

                if s == self.structure[0]:
                    app = []
                    unique_variables = list(self.train[s].unique())
                    picked_variable = random.choice(unique_variables)
                    if s in list(self.query_evid.keys()):
                        compared[s] = self.query_evid[s]
                        weight = weight * self.cpt[config_count][s][str(self.query_evid[s])]
                        compared["weight"] = weight

                    else:
                        compared[s] = picked_variable


                else:
                    parents = s.split("|")[1].split(",")
                    nested_parents =""

                    for _ in parents:
                        nested_parents = nested_parents+str(compared[_])+","

                    app = []
                    unique_variables = list(self.train[s.split("|")[0]].unique())
                    for v in unique_variables:
                        nnn = ""
                        nnn = str(v)+"|"+nested_parents
                        app.append(nnn)

                    picked_variable = random.choice(app)


                    if s.split("|")[0] in list(self.query_evid.keys()):
                        full_cp = str(self.query_evid[s.split("|")[0]])+"|"+nested_parents
                        compared[s.split("|")[0]] = self.query_evid[s.split("|")[0]]
                        weight = weight * self.cpt[config_count][s][full_cp]
                        compared["weight"] = weight
                    else:
                        compared[s.split("|")[0]] = picked_variable.split("|")[0]

                config_count +=1


            break

        return compared

    def generate_likelihood_samples(self):
        
        self.likelihood_inference_time =time.time()
        compare = self.generate_holder()
        samples = []
        for _ in range(self.size):
            sample = self.likelihood_get_sample()
            samples.append(sample)
        self.likelihood_sample_df = pd.DataFrame(samples)
        
        self.likelihood_inference_time = time.time() - self.likelihood_inference_time

        return self.likelihood_sample_df


    def get_likelihood_probability(self):
        
        print("Calculating using Likelihood weighting")
        df = self.generate_likelihood_samples()

        probabilities = {}
        for _ in list(df[self.query_main].unique()):
            up = sum(df[df[self.query_main]== _]["weight"])
            down = sum(df["weight"])
            probabilities[_] = up/down

        print(f"Likelihood weighting : {probabilities}")


    def testing(self,rows):
        x = dict(rows[0:-1])

        return x





try:
    
    train = sys.argv[1]
    test = sys.argv[2]
    cpt_file = sys.argv[3]
    structure = sys.argv[4]
    query = sys.argv[5]
    size = sys.argv[6]
    Inference_object = inference(train,test,cpt_file,structure,query,size)
        
except:
    print("Error")
    print("Correct format is : Python Assessment.py train_file test_file your_cpt your_structure your_query sample_size")
    print("EXAMPLE:")
    print("Python 1b-ApproximateInference.py stroke-data-discretized-train.csv stroke-data-discretized-test.csv cpt_file.txt structure_file.txt \"stroke|age=1,gender=Female\" 1000")