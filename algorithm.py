
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.indep_prob = 0.2    # prob of choosing the indep vector when creating a new leaf (instead of a constant) and during mutation
        self.fitness_func_exponent = 1/3   # The exponent that is used on the num_of_operations
        self.iterations_since_correct_equation = 20    # The number of iterations after we already have a positive fitness function. 
                                                # This means we get the equation perfectly, we just want to search for a shorter solution.
        self.varied_new_member_max_height = 7    # create_new_member_varied_size() creates trees recursively. This is the max size. of the tree that is allowed.
        self.recursive_chance_is_leaf = 0.4    # As the tree is built recrsively, this is the chance that the new recursive node will be a leaf.
        self.inserting_mutation_prob = 0.5   # When we perform a mutation, how likely is it, that the mutation will be an inserting one rather than a simple one.
        self.elite_mutated_twins = True    # Do we perform the cloning of the elite, mutating it, and adding it to the population.

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


# only used for display of the results, not in the algorithm
def average_L1_error(tree):
    dep_vector = Y[XY_index]
    calc = tree.calculate()

    L1_errors = np.abs(dep_vector - calc)
    average_L1_error = np.sum(L1_errors / len(L1_errors))

    return average_L1_error


def give_simple_operation():
    operations = [(addition, 2), (subtraction, 2), (division, 2), (multiplication, 2), (exponentiation, 2)]
    return operations[random.randint(0, len(operations)-1)]

def give_non_simple_operation():
    operations = [(cosine, 1)]
    return operations[random.randint(0, len(operations)-1)]


def give_small_constant(length):
    c = random.randint(1, 10)
    return c * np.ones(length)


#naredi random clana populacije
def create_new_member(params: Parameters):
    
    return create_new_member_varied_size(params)

    # c = random.randint(0,1)
    # if (c == 0):
    #     return create_new_member_3(params)
    # else:
    #     return create_new_member_5(params)


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





def create_new_member_varied_size(params: Parameters):
    
    indep_vector = X[XY_index]

    new_tree = TreeContainer()
    op1 = give_simple_operation()
    
    new_tree.ultimate_parent.set_left_child(TreeNode(False, None, op1))
    
    left_subtree = create_new_subtree_recursively(params.varied_new_member_max_height-1, params)
    right_subtree = create_new_subtree_recursively(params.varied_new_member_max_height-1, params)

    new_tree.ultimate_parent.children[0].set_left_child(left_subtree)
    new_tree.ultimate_parent.children[0].set_right_child(right_subtree)
    
    #!!! treba je tole, da se update-a list_of_nodes
    new_tree.update_list_of_nodes()

    #print(const1)
    #print(const2)
    return new_tree



def give_leaf(params):
    
    indep_vector = X[XY_index]

    const1 = None
    if (random.random() < params.indep_prob):
        const1 = np.copy(indep_vector)
    else: 
        const1 = give_small_constant(indep_vector.size)
    
    return TreeNode(True, const1)


def create_new_subtree_recursively(max_height_to_go, params: Parameters):
    
    # The default condition - prevents infinite trees.
    if max_height_to_go <= 1:
        return give_leaf(params)

    if random.random() < params.recursive_chance_is_leaf:
        return give_leaf(params)


    # Now we are certain that this isn't a leaf.

    operation = give_simple_operation()
    current_node = TreeNode(False, None, operation)

    if operation[1] == 2:
        current_node.set_left_child(create_new_subtree_recursively(max_height_to_go-1, params))
        current_node.set_right_child(create_new_subtree_recursively(max_height_to_go-1, params))
    elif operation[1] == 1:
        current_node.set_left_child(create_new_subtree_recursively(max_height_to_go-1, params))

    return current_node
    
    

# naredi začetno populacijo
def create_starting_population(params: Parameters):
    population = []

    # print(params.pop_size)
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

def mutate(tree, params):
    if params.inserting_mutation_prob < random.random():
        tree.inserting_mutation(simple_elem_func_tuples, [give_small_constant(X[iteracija].size)])
    else:
        tree.simple_mutation(simple_elem_func_tuples, X[iteracija], [give_small_constant(X[iteracija].size)], params.indep_prob)
    return



def create_next_population(population, params: Parameters):
    
    # print(evaluation)
    # size = len(population)
    population.sort(reverse=True, key=lambda tree: fitness(tree, params))

    # evaluation = evaluate_population(population, params)
    # print(evaluation[0:5])
    # print("____________________________")


    # ohranimo najboljse (elitism)
    elite = population[0:int(len(population) * params.elitism)].copy()
    next_population = elite

    if params.elite_mutated_twins:
        mutated_elite = []
        for i in elite:
            # Here the .copy() is necessary. Otherwise we are mutating the elite itself
            # since above only the pointers get copied, not the actual trees.
            current_tree = i.copy()
            mutate(current_tree, params)
            mutated_elite.append(current_tree)
            
        next_population += mutated_elite


    for i in range(int(len(population) * params.new_chromosomes_proportion)):
        next_population.append(create_new_member(params))





    children = []

    num_of_children_to_be_created = len(population) - len(next_population)

    for i in range(int(num_of_children_to_be_created / 2 + 1)):

        child1, child2 = tournement_get_2_children(population, params)

        if (random.random() < params.mutation_prob):
            mutate(child1, params)
        if (random.random() < params.mutation_prob):
            mutate(child2, params)

        children.append(child1)
        children.append(child2)


    next_population = next_population + children[0:num_of_children_to_be_created]


    return next_population




def Genetic_Algorithm(params: Parameters):
    
    population = create_starting_population(params)

    # This is only for accounting reasons.
    # It changes if we end prematurely.
    end_num_of_iterations = params.num_iterations

    iters_since_positive_fitness = 0
    for i in range(0, params.num_iterations):
        population = create_next_population(population, params)
        
        # Checking if we already have the right equation:
        if fitness(population[0], params) >= 0:
            iters_since_positive_fitness += 1
        if iters_since_positive_fitness >= params.iterations_since_correct_equation:
            end_num_of_iterations = i
            break
        
        #for a in population:
            #print(a.calculate())

    #testing
    # print("-------------------------------------------------")
    # print("Our formula:")
    # population[0].print()
    # print("Given formula:")
    # print(eq[XY_index])
    # print("Num of necessary iterations:")
    # print(end_num_of_iterations)
    # print("Fitness function:")
    # print(fitness(population[0], params))
    # print("Calculation:")
    # print(population[0].calculate())
    # print("Actual Y values:")
    # print(Y[XY_index])
    # print("-------------------------------------------------")
    # print("\n\n")

    our_formulas.append(population[0].to_string())
    their_formulas.append(eq[XY_index])
    lengths_of_our_formulas.append(len(population[0].list_of_nodes))
    average_L1_errors.append(average_L1_error(population[0]))
    needed_nums_of_iterations.append(end_num_of_iterations)
    fitness_functions_of_best.append(fitness(population[0], params))
    our_calculations.append(population[0].calculate())
    their_calculations.append(Y[XY_index])


    return population[0]











    


parameters = Parameters()
parameters.num_iterations=1000
parameters.pop_size=30
parameters.elitism=0.1     # what percentage of the best gets kept.
                            # Mind that this will take up 2*elitism of our new population, because of mutated twins of the elite.
parameters.new_chromosomes_proportion=0.3    # percentage of the next pop that will be made up by new random individuals
parameters.mutation_prob=0.1    # chance that a new child will incur a mutation
parameters.tournament_num_of_participants=3    # 2 makes it just random, and 4 seemed to make it too hard for the bad to ever get a chance, so new ideas don't get created.
parameters.indep_prob=0.2 # prob of choosing the indep vector when creating a new leaf (instead of a constant)
parameters.fitness_func_exponent= 1/3    # The exponent that is used on the num_of_operations
parameters.iterations_since_correct_equation = 100    # The number of iterations after we already have a positive fitness function. 
                                                # This means we get the equation perfectly, we just want to search for a shorter solution.

parameters.varied_new_member_max_height = 7    # create_new_member_varied_size() creates trees recursively. This is the max size. of the tree that is allowed.
parameters.recursive_chance_is_leaf = 0.5    # As the tree is built recrsively, this is the chance that the new recursive node will be a leaf.
                                            # 0.5 seems like a good value. 0.3 too little, 0.6 too much.

parameters.inserting_mutation_prob = 0.5   # When we perform a mutation, how likely is it, that the mutation will be an inserting one rather than a simple one.
parameters.elite_mutated_twins = True    # Do we perform the cloning of the elite, mutating it, and adding it to the population.

print("--------------------------------------------------------")
#test = create_new_member()
#print(test.calculate())



# accounting info to store results in:
our_formulas = []
their_formulas = []
lengths_of_our_formulas = []
average_L1_errors = []
needed_nums_of_iterations = []
fitness_functions_of_best = []
our_calculations = []
their_calculations = []



# !!!!! To je sedaj nujna globalna spremenljivka, ki se povsod uporablja.
# Razlog: itak ne bova izvajala nekega multithreadanja. Vseeno če prej nastaviva, pa ni treba potem podajat.
XY_index = 0

num_of_equations_taken = len(eq)
for i in range(num_of_equations_taken):
    XY_index = i
    Genetic_Algorithm(parameters)


# print("Checking how well the varied_size member creation works:")
# for i in range(30):
#     create_new_member_varied_size(parameters).print()


is_correct_solution = np.array(needed_nums_of_iterations) < parameters.num_iterations
print(is_correct_solution)
is_wrong_solution = is_correct_solution == False
print(is_wrong_solution)


# For correct solutions histogram/piechart:
num_of_correct_solutions = np.sum(is_correct_solution)
num_of_wrong_solutions = len(needed_nums_of_iterations) - num_of_correct_solutions
print("num_of_correct_colutions:")
print(num_of_correct_solutions)
print("num_of_correct_colutions:")
print(num_of_wrong_solutions)

# For line graph of length differences for the correct solutions:
num_operations = num_operations[0:num_of_equations_taken]
correct_solutions_length_difference = np.array(lengths_of_our_formulas) - np.array(num_operations)
correct_solutions_length_difference = correct_solutions_length_difference[is_correct_solution]
correct_solutions_length_difference.sort()
print(correct_solutions_length_difference)

# For scatter plot of needed_nums_of_iterations for the correct solutions:
needed_nums_of_iterations = np.array(needed_nums_of_iterations)
correct_solutions_needed_nums_of_iterations = needed_nums_of_iterations[is_correct_solution]
correct_solutions_needed_nums_of_iterations.sort()
print("correct_solutions_needed_nums_of_iterations:")
print(correct_solutions_needed_nums_of_iterations)


# For skatla z brki graph of errors for the wrong solutions:
average_L1_errors = np.array(average_L1_errors)
wrong_solutions_errors = average_L1_errors[is_wrong_solution]
wrong_solutions_errors.sort()
print("wrong_solutions_errors:")
print(wrong_solutions_errors)




# Piechart of correct solutions:
plt.figure(figsize=(8, 8))
plt.pie([num_of_correct_solutions, num_of_wrong_solutions], labels=["correct", "wrong"], autopct='%1.1f%%')
plt.title("Correct solutions")
plt.ylabel("")
plt.show()

# plt.plot(correct_solutions_length_difference)
plt.scatter(range(len(correct_solutions_length_difference)), correct_solutions_length_difference, marker='o', s=25)
plt.title("correct_solutions_length_difference")
plt.ylabel("length difference")
plt.show()

plt.scatter(range(len(correct_solutions_needed_nums_of_iterations)), correct_solutions_needed_nums_of_iterations, marker='o', s=25)
plt.title("Needed number of iterations:")
plt.xlabel("")
plt.ylabel("Number of iterations")
plt.show()

sns.boxplot(wrong_solutions_errors)
plt.xlabel("")
plt.ylabel("L1 errors")
plt.title("L1 error of wrong solutions")
plt.show()



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

    create_new_member_varied_size(params) rekurzivno zgradi novega memberja.
    Vzema parametra:
    - varied_new_member_max_height. Ta pove, kaj je maksimalna visina novega drevesa.
    - recursive_chance_is_leaf.     Ko se drevo rekurzivno gradi, je za vsak nov node taksna moznost, da bo leaf, in ne se en operation node.
    Vse nase operacije imajo 2 sinova, zato je moznost, da se drevo zakljuci na tem koraku, enaka: recursive_chance_is_leaf**2
    Imamo pa (1-recursive_chance_is_leaf)**2 moznosti, da bosta oba sinova operaciji.
    Če bomo torej recursive_chance_is_leaf prestavili na premajhno vrednost, bomo skoraj zagotoo dobili drevo z visino varied_new_member_max_height,
    ker bo preprosto raslo tako globoko.

    Dodal sem inserting_mutation. Ta ne samo spremeni ene od funkcij ali vrednosti, ampak vstavi nov node.

    Problem je, da se samo otroci mutirajo. Nase najboljse resitve pa ostajajo enake.
    Zato sem uvedel podvajanje vseh, ki ostanejo zaradi elitizma in na njih izvedel mutacijo.
"""