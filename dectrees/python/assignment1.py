import monkdata as m
import dtree as tree


#DATA
#training
m1 = m.monk1
m2 = m.monk2
m3 = m.monk3
#testing 
m1_test = m.monk1test
m2_test = m.monk2test
m3_test = m.monk3test

attr = m.attributes


#ASSIGNMENT 1

print "\nDataset Entropy"
m1_ent = tree.entropy(m1)
print "Monk 1 entropy:", m1_ent, " "

m2_ent  = tree.entropy(m2)
print "Monk 2 entropy:", m2_ent , " "

m3_ent = tree.entropy(m3)
print "Monk 3 entropy:", m3_ent, " "

