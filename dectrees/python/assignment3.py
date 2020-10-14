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


#ASSIGNMENT 3

#monk-1 
print "\nMONK-1 information gain"
for x in attr: 
    gain = tree.averageGain(m1, x)
    print "", x, ":", gain

#monk-2
print "\nMONK-2 information gain"
for x in attr: 
    gain = tree.averageGain(m2, x)
    print "", x, ":", gain

#monk-3
print "\nMONK-3 information gain"
for x in attr: 
    gain = tree.averageGain(m3, x)
    print "", x, ":", gain