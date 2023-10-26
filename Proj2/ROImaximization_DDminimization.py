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
import math

from deap import base
from deap import creator
from deap import tools
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

GENERATIONS = 160
INITIAL_POPULATION = 64
N_RUNS = 30
INFINITY = np.inf
GAP_ANALYZED = 10
PERF_THRESHOLD = 1
 
creator.create("FitnessMaxMin", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMaxMin)


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
def evalROI_DD(individual, csv_name, start_date, end_date):
    RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    
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
        
    minLong = INFINITY
    maxShort = -INFINITY
    
    DD = 0 # Drawdown

    for index, row in csv_2020_22.iterrows():

        if(row[col_RSI_long] <= LB_LP and begin_LP == -1):
            begin_LP = row['Close']
            minLong = begin_LP
        if(row[col_RSI_short] >= UP_SP and begin_SP == -1):
            begin_SP = row['Close']
            maxShort = begin_SP
        
        if(begin_LP !=-1 and row['Close'] < minLong):
            minLong = row['Close']
            
        if(begin_SP !=-1 and row['Close'] > maxShort):
            maxShort = row['Close']
            
        
        if(row[col_RSI_long] >= UP_LP and begin_LP != -1):
            end_LP = row['Close']
            ROI_long += ((end_LP - begin_LP)/begin_LP) * 100
            DD += (begin_LP - minLong) / begin_LP
            minLong = INFINITY
            begin_LP = -1
            end_LP = -1
        
        if(row[col_RSI_short] <= LB_SP and begin_SP != -1):
            end_SP = row['Close']
            ROI_short += ((begin_SP - end_SP)/begin_SP) * 100
            DD += (maxShort - begin_SP) / begin_SP
            maxShort = -INFINITY
            begin_SP = -1
            end_SP = -1
        
    # Retornar media dos ROIs de cada individuo
    return (ROI_short + ROI_long)/2, DD

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalROI_DD)

# register the crossover operator
toolbox.register("mate", tools.cxOnePoint)
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mate", tools.cxUniform, indpb = 0.5)
def blending(individual1, individual2):
    # RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    new_individual1 = individual1.copy()
    new_individual2 = individual2.copy()
    new_individual = individual1.copy() 
    for i in range(len(new_individual)):
        if i >= 2:
            factor = 5
            new_individual[i] = (new_individual1[i] + new_individual2[i])/(2*factor)
            random_num = random.randint(0,1)
            if random_num == 0:
                new_individual[i] = math.floor(new_individual[i]) * factor
            else:
                new_individual[i] = math.ceil(new_individual[i]) * factor

    return new_individual   

# toolbox.register("mate", blending)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.5
def mutCustom(individual, indpb):
    # RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    new_individual = individual.copy()
    for i in range(len(new_individual)):
        if random.random() <= indpb:
            random_feature = i
            if random_feature <= 1:
                new_individual[random_feature] = get_n_days()
            elif random_feature == 2 or random_feature == 4:
                new_individual[random_feature] = get_multiple_five(new_individual[random_feature + 1], 'low')
            elif random_feature == 3 or random_feature == 5:
                new_individual[random_feature] = get_multiple_five(new_individual[random_feature - 1] + 5, 'high')

    return new_individual   

def mutCustom_2(individual, indpb):
    # RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    new_individual = individual.copy()
    for i in range(2,len(new_individual)):
        if random.random() <= indpb:
            random_feature = i
            if random_feature == 2 or random_feature == 4:
                candidate = new_individual[random_feature] + random.choice([-5,5]) 
                if(candidate >= 0 and candidate <= 100  and candidate < new_individual[random_feature + 1]):
                    new_individual[random_feature] = candidate
            elif random_feature == 3 or random_feature == 5:
                candidate = new_individual[random_feature] + random.choice([-5,5]) 
                if(candidate >= 0 and candidate <= 100  and candidate > new_individual[random_feature - 1]):
                    new_individual[random_feature] = candidate

    return new_individual   

# toolbox.register('mutate', mutCustom, indpb = 0.5)
toolbox.register('mutate', mutCustom_2, indpb = 0.5) 

# operator for selecting individuals for breeding the next 
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation
toolbox.register("select", tools.selNSGA2)

#----------

def oa_csv(csv_name, start_date_training, end_date_training):
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    
    pareto = tools.ParetoFront()
    
    pop = toolbox.population(n=INITIAL_POPULATION) 
    
    pop = toolbox.select(pop, len(pop))
    
    
    
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.7, 0.1
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = []
    for individual in pop:
        fitnesses.append(toolbox.evaluate(individual, csv_name, start_date_training, end_date_training))
        
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    
    pareto.update(pop)
    # Variable keeping track of the number of generations
    g = 0
    
    # Initialize hall of fame     
    hof = tools.HallOfFame(1)
    
    improve_perf = INFINITY
    max_by_generationsROI = []
    min_by_generationsDD = []
    
    # Begin the evolution
    while g < GENERATIONS and improve_perf > PERF_THRESHOLD: #TODO TESTAR THRESHOLD
        # print(len(pop))
        # A new generation
        g = g + 1
        
        if (g % 10 == 0):
            print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        # Select the next generation individuals
        offspring = tools.selTournamentDCD(pop, len(pop))
        # offspring = tools.selTournament(pop, len(pop), tournsize=2)
        # offspring = tools.selTournament(pop, len(pop), tournsize=3)
        # offspring = tools.selTournament(pop, len(pop), tournsize=4)
        # offspring = tools.selTournament(pop, len(pop), tournsize=8)
        # offspring = tools.selRoulette(pop, len(pop))
        # offspring = tools.selRandom(pop, len(pop))
        # offspring = tools.selBest(pop, len(pop))
        # offspring = tools.selStochasticUniversalSampling(pop, len(pop))
        # offspring = tools.selLexicase(pop, len(pop))
        
        offspring = [toolbox.clone(ind) for ind in offspring]
        
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
            fitnesses.append(toolbox.evaluate(i, csv_name, start_date_training, end_date_training))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        
        pop = toolbox.select(pop + offspring, INITIAL_POPULATION)
        
        
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values for ind in pop]
        
        pareto.update(pop)
        hof.update(pop)
        
        max_by_generationsROI.append(max(fits, key=lambda x: x[0])[0])
        min_by_generationsDD.append(min(fits, key=lambda x: x[1])[1])
        
        if g > GAP_ANALYZED:
            improve_perf_ROI = max_by_generationsROI[g - 1] - max_by_generationsROI[g - GAP_ANALYZED - 1]  
            improve_perf_DD = - (min_by_generationsDD[g - 1] - min_by_generationsDD[g - GAP_ANALYZED - 1])  
            improve_perf = improve_perf_ROI + improve_perf_DD
    
    print("-- End of (successful) evolution --")
    
    index_ROI, fit_maxROI = max(enumerate(fits), key=lambda x: (x[1][0], -x[1][1]))
    index_DD, fit_minDD = min(enumerate(fits), key=lambda x: (x[1][1], -x[1][0]))

    best_ind = [pop[index_ROI], pop[index_DD]]
    
    print("Best individual are %s, %s and %s, %s" % (best_ind[0], best_ind[0].fitness.values, best_ind[1], best_ind[1].fitness.values))
    print('Obtained at generation ', g)
    gen_results = [(max_by_generationsROI[i], min_by_generationsDD[i]) for i in range(len(max_by_generationsROI))]
    print('Succesive generations', gen_results)
    
    return fit_maxROI, fit_minDD, best_ind, pareto

def generate_paretos(pareto_csvs):
    # Plot current population and Pareto front
    for i, pareto_csv in enumerate(pareto_csvs):
        plt.figure(i)
        plt.xlabel("ROI")
        plt.ylabel("DD")
        plt.title('Pareto front ' + csvs_names[i])
        for j in range(N_RUNS):
            front = np.array([ind.fitness.values for ind in pareto_csv[j]])
            plt.scatter(front[:, 0], front[:, 1], c="r", label="Pareto Front")
            
        plt.grid()
        plt.savefig('3_4_1_pareto/3_4_1_pareto_' + csvs_names[i] + '.png')
    
    
def main3_4_1(start_date_training, end_date_training):
    
    result = pd.DataFrame()
    result['Stocks'] = csvs_names
    
    final_maxROI_ROI = []
    final_maxROI_DD = []
    final_minDD_ROI = []
    final_minDD_DD = []
    pareto_csvs = []
    
    for name in csvs_names:
        list_maxROI = []
        list_minDD = []
        pareto_csv = []
        print(name)
         
        for i in range(N_RUNS):
            random.seed(i)
            maxROI, minDD, _, pareto = oa_csv(name, start_date_training, end_date_training)
            pareto_csv.append(pareto)
            list_maxROI.append(maxROI)
            list_minDD.append(minDD)
            
        pareto_csvs.append(pareto_csv)     
        
        fit_maxROI = max(list_maxROI, key=lambda x: x[0])
        fit_minDD = min(list_minDD, key=lambda x: x[1])
      
        final_maxROI_ROI.append(fit_maxROI[0])
        final_maxROI_DD.append(fit_maxROI[1])
        final_minDD_ROI.append(fit_minDD[0])
        final_minDD_DD.append(fit_minDD[1]) 
    
    result['MaxROI_ROI'] = final_maxROI_ROI
    result['maxROI_DD'] = final_maxROI_DD
    result['minDD_ROI'] = final_minDD_ROI
    result['minDD_DD'] = final_minDD_DD
    
    result.to_csv('ACI_Project2_2324_Data/results' + 'results_3_4_1' + '.csv', index = None, header=True, encoding='utf-8')
    
    generate_paretos(pareto_csvs)

def main3_4_2(start_date_training, end_date_training):
    
    train_result = pd.DataFrame()
    train_result['Stocks'] = csvs_names
    
    test_result = pd.DataFrame()
    test_result['Stocks'] = csvs_names
    
    test_final_maxROI_ROI = []
    test_final_maxROI_DD = []
    test_final_minDD_ROI = []
    test_final_minDD_DD = []
    
    train_final_maxROI_ROI = []
    train_final_maxROI_DD = []
    train_final_minDD_ROI = []
    train_final_minDD_DD = []
        
    for name in csvs_names:
        list_maxROI = []
        list_minDD = []
        eval_csv = []
        print(name)
        for i in range(N_RUNS):
            random.seed(i)
            maxROI, minDD, _, pareto = oa_csv(name, start_date_training, end_date_training)
            list_maxROI.append(maxROI)
            list_minDD.append(minDD)
            
            for ind in pareto.items:
                eval_csv.append(evalROI_DD(ind, name, '2020-01-01', '2022-12-31')) 
                
        train_fit_maxROI = max(list_maxROI, key=lambda x: x[0])
        train_fit_minDD = min(list_minDD, key=lambda x: x[1])
        
        train_final_maxROI_ROI.append(train_fit_maxROI[0])
        train_final_maxROI_DD.append(train_fit_maxROI[1])
        train_final_minDD_ROI.append(train_fit_minDD[0])
        train_final_minDD_DD.append(train_fit_minDD[1])
        
        test_fit_maxROI = max(eval_csv, key=lambda x: (x[0], -x[1]))
        test_fit_minDD = min(eval_csv, key=lambda x: (x[1], -x[0]))
        
        test_final_maxROI_ROI.append(test_fit_maxROI[0])
        test_final_maxROI_DD.append(test_fit_maxROI[1])
        test_final_minDD_ROI.append(test_fit_minDD[0])
        test_final_minDD_DD.append(test_fit_minDD[1])    
    
    train_result['MaxROI_ROI'] = train_final_maxROI_ROI
    train_result['maxROI_DD'] = train_final_maxROI_DD
    train_result['minDD_ROI'] = train_final_minDD_ROI
    train_result['minDD_DD'] = train_final_minDD_DD
    train_result.to_csv('ACI_Project2_2324_Data/results' + 'train_results_3_4_2' + '.csv', index = None, header=True, encoding='utf-8')
    
    test_result['MaxROI_ROI'] = test_final_maxROI_ROI
    test_result['maxROI_DD'] = test_final_maxROI_DD
    test_result['minDD_ROI'] = test_final_minDD_ROI
    test_result['minDD_DD'] = test_final_minDD_DD
    test_result.to_csv('ACI_Project2_2324_Data/results' + 'test_results_3_4_2' + '.csv', index = None, header=True, encoding='utf-8')

import time

if __name__ == "__main__":
    start_time = time.time()
    
    # main3_4_1('2020-01-01', '2022-12-31')
    main3_4_2('2011-01-01', '2019-12-31')
    
    time_program = time.time() - start_time
    print("--- %s seconds ---" % (time_program))
    