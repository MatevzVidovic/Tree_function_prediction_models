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

x = np.array(ds['Xs'])
y = np.array(ds['Ys'])
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
iteracija = 0

#print(X[0])
#neki custom input za testing
x = np.array([0,1,2,3,4,5])
y = np.array([0,1,4,9,16,25])


simple_elem_func_tuples = [(addition, 2), (subtraction, 2), (division, 2), (multiplication, 2), (exponentiation, 2)]
additional_elem_func_tuples = [(cosine, 1)]


















#fitness function
#lahko probava še kakšne druge načine računanja napak
def fitness(tree):
    st_operacij = len(tree.list_of_nodes)

    dep_vector = Y[XY_index]
    
    calc = tree.calculate()
    if np.isfinite(calc).all():
        #return -np.sum((y-calc)**2)
        L1_errors = np.abs(dep_vector - calc)
        average_L1_error = np.sum(L1_errors / len(L1_errors))

        # This seems to not work. The values mostly just stay the same for some reason.:
        # multiplied_with_num_of_ops_function = average_L1_error * (st_operacij ** 3)
        
        return -(average_L1_error)
    else:
        return float('-inf')




def give_simple_operation():
    operations = [(addition, 2), (subtraction, 2), (division, 2), (multiplication, 2), (exponentiation, 2)]
    return operations[random.randint(0, len(operations)-1)]

def give_non_simple_operation():
    operations = [(cosine, 1)]
    return operations[random.randint(0, len(operations)-1)]


def give_small_constant(length):
    c = random.randint(1, 10)
    return c * np.ones(length)


#naredi random člana populacije
def create_new_member(indep_probability=0.2):
    
    c = random.randint(0,1)
    if (c == 0):
        return create_new_member_3(indep_probability)
    else:
        return create_new_member_5(indep_probability)


#drevo velikosti 3
def create_new_member_3(indep_probability=0.2):
    """
    example use: create_new_member1(X[iteracija], 0.3)

    indep_probability: how likely a leaf node is to have the indep variable, as opposed to a constant
    """
    
    indep_vector = X[XY_index]

    new_tree = TreeContainer()



    operation = give_simple_operation()
    # Alternativa:
    # operation = None
    # simple_operation_prob mora biti parameter
    # if(random.random() < simple_operation_prob):
    #     operation = give_simple_operation()
    # else:
    #     operation = give_non_simple_operation()



    const1 = None
    if (random.random() < indep_probability):
        const1 = np.copy(indep_vector)
    else: 
        const1 = give_small_constant(indep_vector.size)
    
    const2 = None
    if (random.random() < indep_probability):
        const2 = np.copy(indep_vector)
    else: 
        const2 = give_small_constant(indep_vector.size)

    
    new_tree.ultimate_parent.set_left_child(TreeNode(False, None, operation))
    new_tree.ultimate_parent.children[0].set_left_child(TreeNode(True, const1))
    if operation[1] == 2:
        new_tree.ultimate_parent.children[0].set_right_child(TreeNode(True, const2))

    #!!! treba je tole, da se update-a list_of_nodes
    new_tree.update_list_of_nodes()

    #print(const1)
    #print(const2)
    return new_tree




#drevo velikosti 5
def create_new_member_5(indep_probability=0.2):
    
    indep_vector = X[XY_index]

    new_tree = TreeContainer()
    op1 = give_simple_operation()
    op2 = give_simple_operation()

    const1 = None
    if (random.random() < indep_probability):
        const1 = np.copy(indep_vector)
    else: 
        const1 = give_small_constant(indep_vector.size)
    
    const2 = None
    if (random.random() < indep_probability):
        const2 = np.copy(indep_vector)
    else: 
        const2 = give_small_constant(indep_vector.size)

    const3 = None
    if (random.random() < indep_probability):
        const3 = np.copy(indep_vector)
    else: 
        const3 = give_small_constant(indep_vector.size)



    c = random.randint(0, 1)
    if (c == 0):
        new_tree.ultimate_parent.set_left_child(TreeNode(False, None, op1))
        new_tree.ultimate_parent.children[0].set_left_child(TreeNode(False, None, op2))
        new_tree.ultimate_parent.children[0].set_right_child(TreeNode(True, const1))
        new_tree.ultimate_parent.children[0].children[0].set_left_child(TreeNode(True, const2))
        new_tree.ultimate_parent.children[0].children[0].set_right_child(TreeNode(True, const3))
    else:
        new_tree.ultimate_parent.set_left_child(TreeNode(False, None, op1))
        new_tree.ultimate_parent.children[0].set_right_child(TreeNode(False, None, op2))
        new_tree.ultimate_parent.children[0].set_left_child(TreeNode(True, const1))
        new_tree.ultimate_parent.children[0].children[1].set_left_child(TreeNode(True, const2))
        new_tree.ultimate_parent.children[0].children[1].set_right_child(TreeNode(True, const3))
    
    #!!! treba je tole, da se update-a list_of_nodes
    new_tree.update_list_of_nodes()

    #print(const1)
    #print(const2)
    return new_tree



# naredi začetno populacijo
def create_starting_population(n, indep_prob):
    population = []

    for i in range(0, n):
        population.append(create_new_member(indep_prob))
    return population

# izračuna fitness vsem v populaciji, uporabljal za testing
def evaluate_population(population):
    values = []

    for i in range(0, len(population)):
        values.append(fitness(population[i]))

    return values


def tournement_get_2_children(population, num_of_participants_in_tournament):
    tournament_participants = random.sample(population, num_of_participants_in_tournament)
    tournament_participants.sort(reverse=True, key=fitness)

    child1, child2 = crossover(tournament_participants[0], tournament_participants[1])
    return (child1, child2)



def create_next_population(population, elitism, new_chromosomes_proportion, mutation_prob, tournament_num_of_participants, indep_prob_when_creating_new=0.2):
    """
    elitism: what percentage of the best gets kept
    new_chromosomes_proportion: percentage of the next pop that will be made up by new random individuals
    mutation_prob: chance that a child will also incur a mutation
    """

    # print(evaluation)
    size = len(population)
    population.sort(reverse=True, key=fitness)

    evaluation = evaluate_population(population)
    print(evaluation)
    print("____________________________")


    # ohranimo najboljse (elitism)
    next_population = population[0:int(len(population) * elitism)].copy()


    for i in range(int(len(population) * new_chromosomes_proportion)):
        next_population.append(create_new_member(indep_prob_when_creating_new))





    children = []

    num_of_children_to_be_created = len(population) - len(next_population)

    for i in range(int(num_of_children_to_be_created / 2 + 1)):

        child1, child2 = tournement_get_2_children(population, tournament_num_of_participants)

        if (random.random() < mutation_prob):
            child1.simple_mutation(simple_elem_func_tuples, X[iteracija], [give_small_constant(X[iteracija].size)], 0.3)

        if (random.random() < mutation_prob):
            child2.simple_mutation(simple_elem_func_tuples, X[iteracija], [give_small_constant(X[iteracija].size)], 0.3)

        children.append(child1)
        children.append(child2)


    next_population = next_population + children[0:num_of_children_to_be_created]


    return next_population




def Genetic_Algorithm(num_iterations, pop_size,       elitism, new_chromosomes_proportion, mutation_prob, tournament_num_of_participants, indep_prob_when_creating_new=0.2):
    
    population = create_starting_population(pop_size, indep_prob_when_creating_new)

    for i in range(0, num_iterations):
        population = create_next_population(population, elitism, new_chromosomes_proportion, mutation_prob, tournament_num_of_participants, indep_prob_when_creating_new=0.2)
        #for a in population:
            #print(a.calculate())

    #testing
    population[0].print()
    print()
    print("Calculation:")
    print(population[0].calculate())
    print("Actual Y values:")
    print(Y[XY_index])
    print(len(population[0].list_of_nodes))
    return True





print("--------------------------------------------------------")
#test = create_new_member()
#print(test.calculate())
num_iterations = 500
pop_size = 15
mutation_probability = 0.8

# !!!!! To je sedaj nujna globalna spremenljivka, ki se povsod uporablja.
# Razlog: imava že parameter hell. Da se vsaj malo olajsa. Itak ni velik projekt, in je ok malo slabe prakse.
XY_index = 0

Genetic_Algorithm(num_iterations, pop_size, elitism=0.1, new_chromosomes_proportion=0.1, mutation_prob=0.1, tournament_num_of_participants=4, indep_prob_when_creating_new=0.2)



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