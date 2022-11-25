#############################################################################
# StructureLearning.py
#
# Calculates the LL and BIC Score of a structure
# This code has been written by Ayokunle J. Abisola
# Contact: 26736158@students.lincoln.ac.uk
############################################################################


import math
import pandas as pd
import ast
import sys
from sklearn import metrics

class HillClimb():
    
    penalty = 0
    LL = 0
    structure_index = {}
    
    def __init__(self,train,cpt_file,structure_file):
        
        self.train = self.read(train)
        self.cpt = self.read_cpt(cpt_file)
        self.structure = self.read_structure(structure_file)
        self.get_structure_index()

    
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
    
    def get_structure_index(self):
        
        for s in self.structure:
            self.structure_index[s.split("|")[0]] = self.structure.index(s) 
    
    def log_likelihood_score(self,rows):


        for evid in list(self.train.columns):
            val = rows[evid]
            prob_dist = self.cpt[self.structure_index[evid]]
            structure_key = list(prob_dist.keys())[0]
            parents = structure_key.split("|")

            if (len(parents))<2:
                self.LL = self.LL + math.log(prob_dist[structure_key][str(val)])

            else:

                parents_query = ""
                for p in parents[1].split(","):
                    p_val = rows[p]
                    parents_query = parents_query+str(p_val)+","
                full_query = str(val)+"|"+parents_query
                self.LL = self.LL + math.log(prob_dist[structure_key][full_query])


        return self.LL


    def BIC(self):
        l_score = self.train.apply(lambda x: self.log_likelihood_score(x), axis =1)
        
        print(f"log likelihood score is {l_score.min()}")

        for evid in list(self.train.columns):
            prob_dist = self.cpt[self.structure_index[evid]]
            structure_key = list(prob_dist.keys())[0]
            p = len(prob_dist[structure_key])
            n = len(self.train)
            evid_penalty = (math.log(n)*p)/2
            self.penalty = self.penalty+ evid_penalty

        return l_score.min() - self.penalty
    
    def get_score(self):
        score = self.BIC()
        print(f"BIC is {score}")





try:
    
    train = sys.argv[1]
    cpt_file = sys.argv[2]
    structure = sys.argv[3]
    Learning_object = HillClimb(train,cpt_file,structure)
    print("Calculating....")
    Learning_object.get_score()
        
except:
    print("Error")
    print("Correct format is : Python Assessment.py train_file your_cpt your_structure")
    print("EXAMPLE:")
    print("Correct format is : Python StructureLearning.py stroke-data-discretized-train.csv stroke_file-cpt.txt stroke_file-structure.txt")