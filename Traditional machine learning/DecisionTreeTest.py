from sklearn import tree

treeTest = tree.DecisionTreeClassifier(criterion="entropy", random_state = 0)

#genero i miei dati di input
X = [[0,0,0,0], [0,1,1,1], [0,0,0,1], [1,0,1,0], [1,1,1,1], [0,1,0,0], [0,1,0,1], [1,1,0,1], [0,0,1,0], [0,0,1,1]]
Y = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]

treeTest.fit(X, Y)

print(treeTest.predict([[1,1,1,1]]))
print(treeTest.predict([[0,5,0,1]]))

tree.export_graphviz(treeTest, out_file='tree.dot')
