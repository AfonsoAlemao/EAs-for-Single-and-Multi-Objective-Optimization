#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random

from deap import base
from deap import creator
from deap import tools
from datetime import timedelta
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


AAL = pd.read_csv('ACI_Project2_2324_Data/AAL.csv', encoding='utf-8', sep=';') 
AAPL = pd.read_csv('ACI_Project2_2324_Data/AAPL.csv', encoding='utf-8', sep=';') 
AMZN = pd.read_csv('ACI_Project2_2324_Data/AMZN.csv', encoding='utf-8', sep=';') 
BAC = pd.read_csv('ACI_Project2_2324_Data/BAC.csv', encoding='utf-8', sep=';') 
F = pd.read_csv('ACI_Project2_2324_Data/F.csv', encoding='utf-8', sep=';') 
GOOG = pd.read_csv('ACI_Project2_2324_Data/GOOG.csv', encoding='utf-8', sep=';') 
IBM = pd.read_csv('ACI_Project2_2324_Data/IBM.csv', encoding='utf-8', sep=';') 
INTC = pd.read_csv('ACI_Project2_2324_Data/INTC.csv', encoding='utf-8', sep=';') 
NVDA = pd.read_csv('ACI_Project2_2324_Data/NVDA.csv', encoding='utf-8', sep=';') 
XOM = pd.read_csv('ACI_Project2_2324_Data/XOM.csv', encoding='utf-8', sep=';')


AAL['Date'] = pd.to_datetime(AAL['Date'], format='%d/%m/%Y')  
AAPL['Date'] = pd.to_datetime(AAPL['Date'], format='%d/%m/%Y')  
AMZN['Date'] = pd.to_datetime(AMZN['Date'], format='%d/%m/%Y')  
BAC['Date'] = pd.to_datetime(BAC['Date'], format='%d/%m/%Y')  
F['Date'] = pd.to_datetime(F['Date'], format='%d/%m/%Y')  
GOOG['Date'] = pd.to_datetime(GOOG['Date'], format='%d/%m/%Y')  
IBM['Date'] = pd.to_datetime(IBM['Date'], format='%d/%m/%Y')  
INTC['Date'] = pd.to_datetime(INTC['Date'], format='%d/%m/%Y')  
NVDA['Date'] = pd.to_datetime(NVDA['Date'], format='%d/%m/%Y')  
XOM['Date'] = pd.to_datetime(XOM['Date'], format='%d/%m/%Y')  

csvs = [AAL, AAPL, AMZN, BAC, F, GOOG, IBM, INTC, NVDA, XOM]
csvs_names = ['AAL', 'AAPL', 'AMZN', 'BAC', 'F', 'GOOG', 'IBM', 'INTC', 'NVDA', 'XOM']

GENERATIONS = 100
INITIAL_POPULATION = 100
N_RUNS = 1
INFINITY = np.inf
GAP_ANALYZED = 20
PERF_THRESHOLD = 1
#TODO
 
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)



toolbox = base.Toolbox()


def get_n_days():
    r = random.randint(1, 3)
    return r * 7

def get_multiple_five(num, low_high):
    
    if low_high == 'low':
        # get r < num
        r = random.randint(0, num / 5 - 1)
    else:
        # get r >= num
        r = random.randint(num / 5, 20)
    
    return r * 5

def get_multiple_five_and_get_n_days():
    # RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    
    r_days_short = get_n_days()
    r_days_long = get_n_days()
    r_multiple5_high_short = get_multiple_five(5, 'high')
    r_multiple5_low_short = get_multiple_five(r_multiple5_high_short, 'low')
    
    r_multiple5_high_long = get_multiple_five(5, 'high')
    r_multiple5_low_long = get_multiple_five(r_multiple5_high_long, 'low')
    
    return [r_days_long, r_days_short, r_multiple5_low_long, r_multiple5_high_long, r_multiple5_low_short, r_multiple5_high_short]

# Attribute generator 
toolbox.register("n_days", get_n_days)
toolbox.register("multiple_five", get_multiple_five)
toolbox.register("get_multiple_five_and_get_n_days", get_multiple_five_and_get_n_days)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
# toolbox.register("individual", tools.initRepeat, creator.Individual, 
#     toolbox.get_multiple_five_and_get_n_days, 1)  # beginLP tem de ser sempre menor que o endLP
def generate():
    part = creator.Individual() 
    ind = get_multiple_five_and_get_n_days()

    for i in ind:
        part.append(i)
    
    return part

toolbox.register("individual", generate)


# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalROI(individual, csv_name):
    RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    # print(individual)
    # Para cada csv calcular ROI (2020-2022)
    # cortar df para periodo 20-22
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    
    csv = csvs[csvs_names.index(csv_name)]
    
    ROI_short = 0
    ROI_long = 0
    
    
    csv_2020_22 = csv[(csv['Date'] >= start_date) & (csv['Date'] <= end_date)].reset_index(drop=True)

    begin_LP = -1 # valor de compra de ações
    end_LP = -1 # valor de venda de ações
    begin_SP = -1
    end_SP = -1
    
    
    
    if RSI_long == 7:
        col_RSI_long = 'RSI_7days'
    elif RSI_long == 14:
        col_RSI_long = 'RSI_14days'
    else:
        col_RSI_long = 'RSI_21days'
    
    if RSI_short == 7:
        col_RSI_short = 'RSI_7days'
    elif RSI_short == 14:
        col_RSI_short = 'RSI_14days'
    else:
        col_RSI_short = 'RSI_21days'

    for index, row in csv_2020_22.iterrows():

        if(row[col_RSI_long] <= LB_LP and begin_LP == -1):
            begin_LP = row['Close']
        if(row[col_RSI_short] >= UP_SP and begin_SP == -1):
            begin_SP = row['Close']
        
        if(row[col_RSI_long] >= UP_LP and begin_LP != -1):
            end_LP = row['Close']
            ROI_long += ((end_LP - begin_LP)/begin_LP) * 100
            begin_LP = -1
            end_LP = -1
        
        if(row[col_RSI_short] <= LB_SP and begin_SP != -1):
            end_SP = row['Close']
            ROI_short += ((begin_SP - end_SP)/begin_SP) * 100
            begin_SP = -1
            end_SP = -1
        
    # Retornar media dos ROIs de cada individuo
    return (ROI_short + ROI_long)/2,

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalROI)

# register the crossover operator
#tested
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", tools.	cxOnePoint)
# toolbox.register("mate", tools.	cxPartialyMatched)

# not tested
# toolbox.register("mate", tools.cxUniform)
# toolbox.register("mate", tools.	cxOrdered)
# toolbox.register("mate", tools.cxBlend)
# toolbox.register("mate", tools.cxESBlend)
# toolbox.register("mate", tools.cxESTwoPoint)
# toolbox.register("mate", tools.	cxSimulatedBinary)
# toolbox.register("mate", tools.cxSimulatedBinaryBounded)
# toolbox.register("mate", tools.	cxMessyOnePoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
# tested
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# toolbox.register("mutate", tools.mutGaussian,mu=0,sigma=0.05, indpb=0.05)
# toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
def mutCustom(individual, indpb):
    # RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual

    if random.random() <= indpb:
        random_feature = random.randint(0, 5)
        if random_feature <= 1:
            individual[random_feature] = get_n_days()
        elif random_feature == 2 or random_feature == 4:
            individual[random_feature] = get_multiple_five(individual[random_feature + 1], 'low')
        elif random_feature == 3 or random_feature == 5:
            individual[random_feature] = get_multiple_five(individual[random_feature - 1] + 5, 'high')
    return individual   

toolbox.register('mutate', mutCustom, indpb = 0.05) 

# not tested
# toolbox.register("mutate", tools.mutPolynomialBounded, indpb=0.05)
# toolbox.register("mutate", tools.mutUniformInt, indpb=0.05)
# toolbox.register("mutate", tools.mutESLogNormal, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

#----------

def oa_csv(csv_name):

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=INITIAL_POPULATION) #menor que 144

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = []
    for individual in pop:
        fitnesses.append(toolbox.evaluate(individual, csv_name))
        
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Initialize hall of fame     
    hof = tools.HallOfFame(1)
    
    improve_perf = INFINITY
    max_by_generations = []
    
    # Begin the evolution
    while g < GENERATIONS and improve_perf > PERF_THRESHOLD: #TODO TESTAR THRESHOLD
        # A new generation
        g = g + 1
        
        if (g % 50 == 0):
            print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = []
        for i in invalid_ind:
            fitnesses.append(toolbox.evaluate(i, csv_name))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        hof.update(pop)
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        
        max_by_generations.append(max(fits))
        
        # print("  Min %s" % min(fits))
        # print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)
        
        if g > GAP_ANALYZED:
            improve_perf = max_by_generations[g - 1] - max_by_generations[g - GAP_ANALYZED - 1]  
        
        best_ind_gen = tools.selBest(pop, 1)[0]
        # print("Best individual in generation %d: %s, %s" % (g, best_ind_gen, best_ind_gen.fitness.values))
        # print("Hall of fame: {} {}".format(hof[0], hof[0].fitness.values[0]))
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print('Obtained at generation ', g)
    print('Succesive generations maximums ', max_by_generations)
    
    return min(fits), max(fits), mean, std, best_ind, best_ind.fitness.values


def generate_histograms(best_individuals):
    
    data = pd.DataFrame(best_individuals, columns=['RSI_long', 'RSI_short', 'LB_LP', 'UP_LP', 'LB_SP', 'UP_SP'])

    # RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    array_days = [7, 14, 21]
    array_multiples_five = [5*i for i in range(21)]
    
    # print('HIST, RSI_long', np.array(best_individuals)[:, 0])
    plt.figure(0)
    categories = data['RSI_long'].value_counts().index
    counts = data['RSI_long'].value_counts().values
    plt.bar(categories, counts, width=7)
    plt.xlabel('RSI period to apply for long positions')
    plt.ylabel('Frequency')
    plt.title('Histogram of RSI_long')
    plt.xlim([3.5, 24.5])
    plt.xticks(array_days) 
    plt.savefig('RSI_long.png')
    
    # print('HIST, RSI_short', np.array(best_individuals)[:, 1])
    plt.figure(1)
    categories = data['RSI_short'].value_counts().index
    counts = data['RSI_short'].value_counts().values
    plt.bar(categories, counts, width=7)
    plt.xlabel('RSI period to apply for short positions')
    plt.ylabel('Frequency')
    plt.title('Histogram of RSI_short')
    plt.xlim([3.5, 24.5])
    plt.xticks(array_days)
    plt.savefig('RSI_short.png')
    
    # print('HIST, LB_LP', np.array(best_individuals)[:, 2])
    plt.figure(2)
    categories = data['LB_LP'].value_counts().index
    counts = data['LB_LP'].value_counts().values
    plt.bar(categories, counts, width=5)
    plt.xlabel('Lower band value to open a long position')
    plt.ylabel('Frequency')
    plt.title('Histogram of LB_LP')
    plt.xlim([-2.5, 102.5])
    plt.xticks(array_multiples_five) 
    plt.savefig('LB_LP.png')
    
    # print('HIST, UP_LP', np.array(best_individuals)[:, 3])
    plt.figure(3)
    categories = data['UP_LP'].value_counts().index
    counts = data['UP_LP'].value_counts().values
    plt.bar(categories, counts, width=5)
    plt.xlabel('Upper band value to close a long position')
    plt.ylabel('Frequency')
    plt.title('Histogram of UP_LP')
    plt.xlim([-2.5, 102.5])
    plt.xticks(array_multiples_five) 
    plt.savefig('UP_LP.png')
    
    # print('HIST, LB_SP', np.array(best_individuals)[:, 4])
    plt.figure(4)
    categories = data['LB_SP'].value_counts().index
    counts = data['LB_SP'].value_counts().values
    plt.bar(categories, counts, width=5)
    plt.xlabel('Lower band value to close a short position')
    plt.ylabel('Frequency')
    plt.title('Histogram of LB_SP')
    plt.xlim([-2.5, 102.5])
    plt.xticks(array_multiples_five) 
    plt.savefig('LB_SP.png')
    
    # print('HIST, UP_SP', np.array(best_individuals)[:, 5])
    plt.figure(5)
    categories = data['UP_SP'].value_counts().index
    counts = data['UP_SP'].value_counts().values
    plt.bar(categories, counts, width=5)  
    plt.xlabel('Upper band value to open a short position')
    plt.ylabel('Frequency')
    plt.title('Histogram of UP_SP')
    plt.xlim([-2.5, 102.5])
    plt.xticks(array_multiples_five) 
    plt.savefig('UP_SP.png')
    
def generate_boxplots(fitness_csvs):
    #TODO    
    return

def main():
    
    result = pd.DataFrame()
    result['Stocks'] = csvs_names
    
    max_final = []
    min_final = []
    avg_final = []
    std_final = []
    best_individuals = []
    
    fitness_csvs = []
    
    for name in csvs_names:
        list_max = []
        list_min = []
        list_avg = []
        list_std = []
        fitness_final = []
         
        for i in range(N_RUNS):
            random.seed(i)
            max, min, avg, std, best_individual, fitness = oa_csv(name)
            best_individuals.append(best_individual)
            list_max.append(max)
            list_min.append(min)
            list_avg.append(avg)
            list_std.append(std)
            fitness_final.append(fitness)
        max_final.append(np.mean(list_max))
        min_final.append(np.mean(list_min))
        avg_final.append(np.mean(list_avg))
        std_final.append(np.mean(list_std))
        fitness_csvs.append(fitness_final)        
    
    result['Max'] = max_final 
    result['Min'] = min_final
    result['Mean'] = avg_final 
    result['STD'] = std_final 
    result.to_csv('ACI_Project2_2324_Data/' + 'results' + '.csv', index = None, header=True, encoding='utf-8')
    
    generate_histograms(best_individuals)
    generate_boxplots(fitness_csvs)
    

import time

if __name__ == "__main__":
    start_time = time.time()
    
    main()
    
    time_program = time.time() - start_time
    print("--- %s seconds ---" % (time_program))
    