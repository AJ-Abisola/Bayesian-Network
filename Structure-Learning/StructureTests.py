#############################################################################
# StructureTests.py
#
# Searches a grid for the best structure
# This code has been written by Ayokunle J. Abisola
# Contact: 26736158@students.lincoln.ac.uk
############################################################################

from StructureLearning import HillClimb

cpt_files = ["stroke_file-cpt.txt","st1_file-cpt.txt","st2_file-cpt.txt","st3_file-cpt.txt","st4_file-cpt.txt","st5_file-cpt.txt","st6_file-cpt.txt"]
structure_files = ["stroke_file-structure.txt","st1_file-structure.txt","st2_file-structure.txt","st3_file-structure.txt","st4_file-structure.txt","st5_file-structure.txt","st6_file-structure.txt"]

for _ in range(0,len(structure_files)):
    hc = HillClimb("stroke-data-discretized-train.csv",cpt_files[_],structure_files[_])
    print(f"FOR STRUCTURE {_}")
    hc.get_score()