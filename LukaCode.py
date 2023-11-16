import pygad
import matplotlib.pyplot as plt
import pandas as pd
from trees import *
import random

ds = pd.read_csv('dataset.csv')
eq = np.array(ds['Equation'])
#print(eq)

#število operacij v originalni enačbi (za uporabo v fitness funkciji)
num_operations = []
#ne dela za kotne funkcije ampak jih ni v datasetu
jeKrat = False
for x in eq:
    vsota = 0
    for j in x:
        if ((j == ' ') or (j == '(') or (j == ')')):
            continue
        elif (j == '*'):
            if (jeKrat):
                vsota += 1
                jeKrat = False
                continue
            else:
                jeKrat = True
                continue
        else: vsota += 1

        if (jeKrat):
            vsota += 1
            jeKrat = False
    num_operations.append(vsota)

#print(num_operations)

y = np.array(ds['Ys'])
x = np.array(ds['Xs'])
X = []
Y = []
array_length = 10

#iz stringa ki nam ga da dataset naredimo liste za x,y pare
#z array_length omejimo količino podatkov
for i in range(0, x.size):
    temp = np.fromstring(x[i][1:len(x[i])-1], dtype=float, sep=', ')
    X.append(temp[0:array_length])
X = np.array(X)

for i in range(0, y.size):
    temp = np.fromstring(y[i][1:len(y[i])-1], dtype=float, sep=', ')
    Y.append(temp[0:array_length])
Y = np.array(Y)

#print (Y)


#uporablja se kot globalna spremenljivka, pove številko enačbe, lahko kasneje naredim brez
iteracija = 0;

#print(X[0])
#neki custom input za testing
x = np.array([0,1,2,3,4,5])
y = np.array([0,1,4,9,16,25])

possible_elem_func_tuples = [(addition, 2), (subtraction, 2), (division, 2), (multiplication, 2), (exponentiation, 2), (cosine, 1)]

#fitness function
#lahko probava še kakšne druge načine računanja napak
def fitness(tree):
    st_operacij = len(tree.list_of_nodes)
    diff = abs(st_operacij - num_operations[iteracija])

    calc = tree.calculate()
    if np.isfinite(calc).all():
        #return -np.sum((y-calc)**2)
        return -(np.sum(np.sqrt(np.abs(Y[iteracija]-calc))) + diff)

    else: return -99999

def operation2():
    operations = [addition, subtraction, multiplication, division, exponentiation]
    return operations[random.randint(0, len(operations)-1)]
def operation1():
    operations = [cosine]
    return operations[random.randint(0, len(operations)-1)]
def const(length):
    c = random.randint(1, 10)
    return c * np.ones(length)

#naredi random člana populacije
def create_new_member():
    c = random.randint(0,1)
    if (c == 0):
        return create_new_member1()
    else:
        return create_new_member2()

#drevo velikosti 3
def create_new_member1():
    new_tree = TreeContainer()

    zeros = np.zeros(1)
    ones = np.ones(9)

    op_choice = np.append(zeros, ones)

    o = random.choice(op_choice)
    operation = pass_through

    if (o == 0):
        operation = operation1()
    else:
        operation = operation2()

    const1 = np.zeros(1)
    c = random.randint(0,1)
    if (c == 0):
        const1 = np.copy(X[iteracija])
    else: const1 = const(X[iteracija].size)

    const2 = np.zeros(1)
    c = random.randint(0,1)
    if (c == 0):
        const2 = np.copy(X[iteracija])
    else: const2 = const(X[iteracija].size)

    new_tree.ultimate_parent.set_left_child(TreeNode(False, None, (operation, o+1)))
    new_tree.ultimate_parent.children[0].set_left_child(TreeNode(True, const1))
    if o != 0:
        new_tree.ultimate_parent.children[0].set_right_child(TreeNode(True, const2))

    #print(const1)
    #print(const2)
    #!!! treba kopirat da se update-a list_of_nodes
    return new_tree.copy()

#drevo velikosti 5
def create_new_member2():
    new_tree = TreeContainer()
    op1 = operation2()
    op2 = operation2()

    zeros = np.zeros(1)
    ones = np.ones(3)

    x_choice = np.append(zeros, ones)

    c = random.choice(x_choice)

    const1 = np.zeros(1)
    if (c == 0):
        const1 = np.copy(X[iteracija])
    else: const1 = const(X[iteracija].size)

    const2 = np.zeros(1)
    c = random.choice(x_choice)
    if (c == 0):
        const2 = np.copy(X[iteracija])
    else: const2 = const(X[iteracija].size)

    const3 = np.zeros(1)
    c = random.choice(x_choice)
    if (c == 0):
        const3 = np.copy(X[iteracija])
    else: const3 = const(X[iteracija].size)


    c = random.randint(0, 1)
    if (c == 0):
        new_tree.ultimate_parent.set_left_child(TreeNode(False, None, (op1, 2)))
        new_tree.ultimate_parent.children[0].set_left_child(TreeNode(False, None, (op2, 2)))
        new_tree.ultimate_parent.children[0].set_right_child(TreeNode(True, const1))
        new_tree.ultimate_parent.children[0].children[0].set_left_child(TreeNode(True, const2))
        new_tree.ultimate_parent.children[0].children[0].set_right_child(TreeNode(True, const3))
    else:
        new_tree.ultimate_parent.set_left_child(TreeNode(False, None, (op1, 2)))
        new_tree.ultimate_parent.children[0].set_right_child(TreeNode(False, None, (op2, 2)))
        new_tree.ultimate_parent.children[0].set_left_child(TreeNode(True, const1))
        new_tree.ultimate_parent.children[0].children[1].set_left_child(TreeNode(True, const2))
        new_tree.ultimate_parent.children[0].children[1].set_right_child(TreeNode(True, const3))
    #print(const1)
    #print(const2)
    #!!! treba kopirat da se update-a list_of_nodes
    return new_tree.copy()

#naredi začetno populacijo
def create_starting_population(n):
    pop = []

    for i in range(0, n):
        pop.append(create_new_member())
    return pop

#zračuna fitness vsem v populacije, uporabljal za testing
def evaluate_population(population):
    values = []

    for i in range(0, len(population)):
        values.append(fitness(population[i]))

    return values


def create_next_population(population, mutation_probability):
    #evaluation = evaluate_population(population)
    #print(evaluation)
    size = len(population)
    population.sort(reverse=True, key=fitness)

    for a in population:
            #print(a.calculate())
            print(fitness(a))

    print("____________________________")
    population = population[0:int(len(population)/6)] #znebimo se slabih
    children = []

    for i in range(0, int(len(population))):
        a = random.randint(0, len(population)-1)
        b = random.randint(0, len(population)-1)

        child1, child2 = crossover(population[a], population[b])

        c = random.random()
        if (mutation_probability > c):
            child1.simple_mutation(possible_elem_func_tuples, [const(X[iteracija].size), X[iteracija], const(X[iteracija].size)])

        c = random.random()
        if (mutation_probability > c):
            child2.simple_mutation(possible_elem_func_tuples, [const(X[iteracija].size), X[iteracija], const(X[iteracija].size)])

        children.append(child1)
        children.append(child2)


    population = population + children
    while(len(population) < size):
        population.append(create_new_member())


    return population

def Genetic_Algorithm(num_iterations, pop_size, XY_index, mutation_probability):
    population = create_starting_population(pop_size)

    for i in range(0, num_iterations):
        population = create_next_population(population, mutation_probability)
        #for a in population:
            #print(a.calculate())

    #testing
    population[1].print()
    print(population[1].calculate())
    print(len(population[1].list_of_nodes))
    return True

print("--------------------------------------------------------")
#test = create_new_member()
#print(test.calculate())
num_iterations = 500
pop_size = 30
iteracija = 0
XY_index = iteracija
mutation_probability = 0.1

Genetic_Algorithm(num_iterations, pop_size, XY_index, mutation_probability)



"""
diary:
    naredil prvo "working" verzijo algoritma, testiran na preprostih enačbah tipa x^2
    začetno populacija vsebuje osebke z eno operacijo
    novo populacijo naredi po principu: odstranimo polovico najslabših, ostali se razmnožijo

    pri dataset primerih pride do overflowa, zmanjšal velikosti iz 100 na 10.
    v fitness funkcijo dodal razliko med velikostjo naše in originalne enačbe
    probaval različne parametre, uspešnost algoritma slaba, nastane preveč
    enakih osebkov.

    spremenil nastanek nove populacije: odvrže več osebkov populacije da se generira več novih.
    rezultati so boljši ampak je nastanek populacij preveč predvidljiv saj je naša
    funkcija za začetno populacijo in generiranje novega osebka preveč preprosta.

    dodal funkcije create_new_member da lahko generira drevesa velikosti 5, zdaj je generiranje
    novega člena bolj naključno.
"""