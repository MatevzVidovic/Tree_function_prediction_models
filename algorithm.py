import pygad
import matplotlib.pyplot as plt
import pandas as pd
from trees import *
import random



class Parameters:

    def __init__(self):
        # These are only the default values. They should be changed before use.
        self.num_iterations = 500
        self.pop_size = 15
        self.elitism = 0.1
        self.new_chromosomes_proportion = 0.1
        self.mutation_prob = 0.1
        self.tournament_num_of_participants = 4
        self.indep_prob = 0.2    # prob of choosing the indep vector when creating a new leaf (instead of a constant)
        self.fitness_func_exponent = 1/3   # The exponent that is used on the num_of_operations



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
def fitness(tree, params: Parameters):
    st_operacij = len(tree.list_of_nodes)

    dep_vector = Y[XY_index]
    
    calc = tree.calculate()
    if np.isfinite(calc).all():
        #return -np.sum((y-calc)**2)
        L1_errors = np.abs(dep_vector - calc)
        average_L1_error = np.sum(L1_errors / len(L1_errors))

        # These two solutions are equally good according to this function:
        # Solution 1:   average_L1_error == 2x, st_operacij == n
        # Solution 2:   average_L1_error == x,  st_operacij == (2**(1/exponent))*n == (2**(1/(1/3)))*n == (2**3)*n in the current case
        multiplied_with_num_of_ops_function = average_L1_error * (st_operacij ** (params.fitness_func_exponent))

        if multiplied_with_num_of_ops_function <= 0:
            # This way, when we do hit the correct equation, it doesn't get ruined by next iterations with longer equations.
            # For example, case 0:    we found the correct equation soon. Then the evaluation vector was all zeros. In the end, the winning function was: (x**4)+3+3
            # We chose 100 - st_operacij, because if st_operacij is larger than 100 it might as well go into the negatives, since it is so bad.
            # The minus is added because of the negation of the value when returning. The brackets aren't simplifed to retain comprehensibility.
            multiplied_with_num_of_ops_function = -(100 - st_operacij)
        
        return -(multiplied_with_num_of_ops_function)
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
def create_new_member(params: Parameters):
    
    c = random.randint(0,1)
    if (c == 0):
        return create_new_member_3(params)
    else:
        return create_new_member_5(params)


#drevo velikosti 3
def create_new_member_3(params: Parameters):
        
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
    if (random.random() < params.indep_prob):
        const1 = np.copy(indep_vector)
    else: 
        const1 = give_small_constant(indep_vector.size)
    
    const2 = None
    if (random.random() < params.indep_prob):
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
def create_new_member_5(params: Parameters):
    
    indep_vector = X[XY_index]

    new_tree = TreeContainer()
    op1 = give_simple_operation()
    op2 = give_simple_operation()

    const1 = None
    if (random.random() < params.indep_prob):
        const1 = np.copy(indep_vector)
    else: 
        const1 = give_small_constant(indep_vector.size)
    
    const2 = None
    if (random.random() < params.indep_prob):
        const2 = np.copy(indep_vector)
    else: 
        const2 = give_small_constant(indep_vector.size)

    const3 = None
    if (random.random() < params.indep_prob):
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
def create_starting_population(params: Parameters):
    population = []

    print(params.pop_size)
    for i in range(0, params.pop_size):
        population.append(create_new_member(params))
    return population

# izračuna fitness vsem v populaciji, uporabljal za testing
def evaluate_population(population, params):
    values = []

    for i in range(0, len(population)):
        values.append(fitness(population[i], params))

    return values


def tournement_get_2_children(population, params: Parameters):
    tournament_participants = random.sample(population, params.tournament_num_of_participants)
    tournament_participants.sort(reverse=True, key=lambda tree: fitness(tree, params))

    child1, child2 = crossover(tournament_participants[0], tournament_participants[1])
    return (child1, child2)



def create_next_population(population, params: Parameters):
    """
    elitism: what percentage of the best gets kept
    new_chromosomes_proportion: percentage of the next pop that will be made up by new random individuals
    mutation_prob: chance that a child will also incur a mutation
    """

    # print(evaluation)
    size = len(population)
    population.sort(reverse=True, key=lambda tree: fitness(tree, params))

    evaluation = evaluate_population(population, params)
    print(evaluation)
    print("____________________________")


    # ohranimo najboljse (elitism)
    next_population = population[0:int(len(population) * params.elitism)].copy()


    for i in range(int(len(population) * params.new_chromosomes_proportion)):
        next_population.append(create_new_member(params))





    children = []

    num_of_children_to_be_created = len(population) - len(next_population)

    for i in range(int(num_of_children_to_be_created / 2 + 1)):

        child1, child2 = tournement_get_2_children(population, params)

        if (random.random() < params.mutation_prob):
            child1.simple_mutation(simple_elem_func_tuples, X[iteracija], [give_small_constant(X[iteracija].size)], 0.3)

        if (random.random() < params.mutation_prob):
            child2.simple_mutation(simple_elem_func_tuples, X[iteracija], [give_small_constant(X[iteracija].size)], 0.3)

        children.append(child1)
        children.append(child2)


    next_population = next_population + children[0:num_of_children_to_be_created]


    return next_population




def Genetic_Algorithm(params: Parameters):
    
    population = create_starting_population(params)

    for i in range(0, params.num_iterations):
        population = create_next_population(population, params)
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













parameters = Parameters()
parameters.num_iterations=500
parameters.pop_size=15
parameters.elitism=0.1
parameters.new_chromosomes_proportion=0.1
parameters.mutation_prob=0.1
parameters.tournament_num_of_participants=4
parameters.indep_prob=0.2 # prob of choosing the indep vector when creating a new leaf (instead of a constant)
parameters.fitness_func_exponent= 1/3    # The exponent that is used on the num_of_operations



print("--------------------------------------------------------")
#test = create_new_member()
#print(test.calculate())


# !!!!! To je sedaj nujna globalna spremenljivka, ki se povsod uporablja.
# Razlog: itak ne bova izvajala nekega multithreadanja. Vseeno če prej nastaviva, pa ni treba potem podajat.
XY_index = 0

Genetic_Algorithm(parameters)



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

    XY_index je sedaj globalna spremenljivka. Nikjer se je ne podaja.
    Preprosto olajša zadeve. Ker saj itak X in Y tudi globalno podajava.
    Pac ne delava huge projekta. Naj bodo dobre prakse za kdaj drugic.

    Vsi parametri se zdaj nastavljajo v objekt parameters, ki je tipa Parameters.
    Tako nimava takega function parameter hell-a, kot sva ga imela prej.
    Ta objekt si potem funkcije stalno podajajo, in vsaka uporabi tiste atribute, ki jih potrebuje.
    Saj bi lahko tudi to imela globalno, kot imava XY-index, ampak ade, vsaj malo dobre prakse.
    Pa tudi lazje je programirat, ce ves, na kaj se navezujes.

    Fitness function je dozivela spremembe. Se mi zdi, da kar okej deluje trenutno.
"""