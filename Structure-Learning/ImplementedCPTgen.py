#############################################################################
# ImplementedCPTgen.py
#
# Generates the CPT given a certain structure and a data
# This code has been written by Ayokunle J. Abisola
# Contact: 26736158@students.lincoln.ac.uk
#############################################################################



import pandas as pd
import itertools
import sys

class CPT():
    
    def __init__(self,train,structure_file):
        
        self.train = self.read(train)
        self.structure = self.read_structure(structure_file)
    
    
    def read(self,file):
        read = pd.read_csv(file)
        return read
    
    
    def read_structure(self,checking):
        train_list = []
        with open(checking) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line != '\n':
                    train_list.append(line.rstrip())
                    
        return train_list
    
    def multiply_prob(self,checks):
        result = 1
        for check in checks:
            result = result * check
        return result


    def normalize(self,query):
        est_keys = list(self.estimation[query].keys())
        est_list = []
        for _ in est_keys:
            est_list.append(_.split("|")[1]) 
        done = []  
        indices = []
        for _ in est_list:
            if _ not in done:
                indices.append([i for i, x in enumerate(est_list) if x == _])
                done.append(_)

        all_instance = []
        for i in indices:
            u=0
            for _ in i:
                val = self.estimation[query][est_keys[_]]
                u = u + val
            all_instance.append(u)

        u_ind = 0
        for i in indices:
            for _ in i:
                self.estimation[query][est_keys[_]] = self.estimation[query][est_keys[_]]/all_instance[u_ind]
            u_ind+=1
        return self.estimation


    def estimating(self):
        self.estimations = []

        for pattern in self.structure:
            other_list = []
            self.estimation = {}
            patt = {}
            structure_line = pattern.split("|")

            if len(structure_line)==1:
                evidence_variables = list(self.train[structure_line[0]].unique())
                for variable in evidence_variables:
                    count = self.train[structure_line[0]].value_counts()[variable]
                    evidence_probability = count/len(self.train)
                    keys = str(structure_line[0])
                    var = str(variable)
                    patt[var] = evidence_probability
                    self.estimation[keys] = patt
                self.estimations.append(self.estimation)
                print(self.estimation)
                print("------------------------------------")

            else:
                #the parents
                others = structure_line[1].split(",")
                #the desired
                evid = structure_line[0]
                evidence_variables = list(self.train[evid].unique())

                #list of the variables of parents
                for other in others:
                    other_variables = list(self.train[other].unique())
                    other_list.append(other_variables)

                #All possible combinations
                possible_comb = []
                for p in itertools.product(*other_list):
                    possible_comb.append(list(p))


                for variable in evidence_variables:
                    count = self.train[evid].value_counts()[variable]
                    evidence_probability = count/len(self.train)
                    var_list = []

                    #looping through all possible combinations
                    for other_var in possible_comb:
                        var = str(variable)+"|"
                        for _ in other_var:
                            var = var+str(_)+","
                        count_e = 0
                        checks = []
                        dd = []
                        checks.append(evidence_probability)
                        for value in other_var:
                            d = self.train[others[count_e]].value_counts()[value]/len(self.train)
                            dd.append(d)
                            variable_count = len(self.train[(self.train[evid]==variable) & (self.train[others[count_e]]==value)])
                            variable_probability = variable_count/count
                            if variable_probability==0:
                                size = len(self.train[evid].unique())
                                variable_probability = (variable_count+1)/(count+size)
                            checks.append(variable_probability)
                            count_e+=1

                        up = self.multiply_prob(checks)
                        down = self.multiply_prob(dd)
                        prob = up/down
                        keys = str(evid)+"|"+structure_line[1]
                        var_list.append(var)
                        patt[var]= prob
                        self.estimation[keys] = patt

                self.estimation = self.normalize(keys)

                self.estimations.append(self.estimation)
                print(self.estimation)
                print("------------------------------------") 


    def save_file(self,save_as):

        with open(save_as, 'w') as f:
            for line in self.estimations:
                f.write(f"{line}\n")
                f.write("\n")




try:
    train = sys.argv[1]
    structure = sys.argv[2]
    save_as = sys.argv[3]
    cpt = CPT(train,structure)
    cpt.estimating()
    cpt.save_file(save_as)
except:
    print("Error")
    print("Correct format is : Python Assessment.py train_file your_structure name_to_save")
    print("EXAMPLE:")
    print("Python ImplementedCPTgen.py stroke-data-discretized-train.csv stroke_file-structure.txt my_cpt.txt")