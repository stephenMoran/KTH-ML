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

#ASSIGNMENT 0 - word based

#ASSIGNMENT 4 


#ASSIGNMENT 5
#monk-1 highest info gain is attribute 5 
"Return subset of data samples where the attribute has the given value"
print attr

n1 = tree.select(m1,attr[4], 1)
n2 = tree.select(m1,attr[4], 2)
n3 = tree.select(m1,attr[4], 3)
n4 = tree.select(m1,attr[4], 4)

print "\nnode  1  information gain"
for x in attr: 
    gain = tree.averageGain(n1, x)
    print "", x, ":", gain
#No gain
print tree.mostCommon(n1)

print "\nnode  2  information gain"
for x in attr: 
    gain = tree.averageGain(n2, x)
    print "", x, ":", gain

#attr 4 most gain
n2_1 = tree.select(n1,attr[3], 1)
print tree.mostCommon(n2_1)
n2_2 = tree.select(n1,attr[3], 2)
print tree.mostCommon(n2_2)
n2_3 = tree.select(n1,attr[3], 3)
print tree.mostCommon(n2_3)



print "\nnode  3  information gain"
for x in attr: 
    gain = tree.averageGain(n3, x)
    print "", x, ":", gain

#attr 6 most gain
n3_1 = tree.select(n3,attr[5], 1)
print tree.mostCommon(n3_1)
n3_2 = tree.select(n3,attr[5], 2)
print tree.mostCommon(n3_2)


print "\nnode  4  information gain"
for x in attr: 
    gain = tree.averageGain(n4, x)
    print "", x, ":", gain

#attr 1 most gain
n4_1 = tree.select(n4,attr[0], 1)
print tree.mostCommon(n4_1)
n4_2 = tree.select(n4,attr[0], 2)
print tree.mostCommon(n4_2)
n4_3 = tree.select(n4,attr[0], 3)
print tree.mostCommon(n4_3)



print  "\nMONK-1 TREE"
build_m1 = tree.buildTree(m1, m.attributes)
#print build_m1
print "Training", 1 - tree.check(build_m1, m1)
print "Testing", 1 - tree.check(build_m1, m1_test)

print  "MONK-2 TREE"
build_m2 = tree.buildTree(m2, m.attributes)
#print build_m2
print "Training", 1 - tree.check(build_m2, m2)
print "Testing", 1- tree.check(build_m2, m2_test)

print  "MONK-3 TREE"
build_m3 = tree.buildTree(m3, m.attributes)
#print build_m3
print "Training", 1 - tree.check(build_m3, m3)
print "Testing", 1 - tree.check(build_m3, m3_test)
