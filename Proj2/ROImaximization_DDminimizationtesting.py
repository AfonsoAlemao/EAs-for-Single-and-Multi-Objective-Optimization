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

GENERATIONS = 20
INITIAL_POPULATION = 64
N_RUNS = 5
INFINITY = np.inf
GAP_ANALYZED = 8
PERF_THRESHOLD = 1
 
MUTPB = 0.7
indpbMut = 0.5
CXPB = 0.9
CrossType = 'OnePoint'
SelType = 'NSGA2_' + 'TornDCD'
 
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
    # print(individual)
    # Para cada csv calcular ROI (2020-2022)
    # cortar df para periodo 20-22
    # start_date = '2020-01-01'
    # end_date = '2022-12-31'
    
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

toolbox.register('mutate', mutCustom, indpb = indpbMut) 

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
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
    # CXPB, MUTPB = 0.5, 0.5
    
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
    max_by_generations = []
    
    # Begin the evolution
    while g < GENERATIONS and improve_perf > PERF_THRESHOLD: #TODO TESTAR THRESHOLD
        # print(len(pop))
        # A new generation
        g = g + 1
        
        if (g % 10 == 0):
            print("-- Generation %i --" % g)
        
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
        fits = [ind.fitness.values[0] for ind in pop]
        
        pareto.update(pop)
        hof.update(pop)
        
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
    
    return max(fits), min(fits), mean, std, best_ind, best_ind.fitness.values, pareto
        

def main3_4_1(start_date_training, end_date_training):
    
    result = pd.DataFrame()
    csvs_names_to_df = csvs_names.copy()
    csvs_names_to_df.append(None)
    result['Stocks'] = csvs_names_to_df
    
    max_final = []
    min_final = []
    avg_final = []
    std_final = []
    best_individuals_csvs = []
    fitness_csvs = []
    pareto_csvs = []
    
    for name in csvs_names:
        list_max = []
        list_min = []
        list_avg = []
        list_std = []
        fitness_final = []
        best_individuals = []
        pareto_csv = []
        print(name)
         
        for i in range(N_RUNS):
            random.seed(i)
            max, min, avg, std, best_individual, fitness, pareto = oa_csv(name, start_date_training, end_date_training)
            pareto_csv.append(pareto)
            best_individuals.append(best_individual)
            list_max.append(max)
            list_min.append(min)
            list_avg.append(avg)
            list_std.append(std)
            fitness_final.append(fitness[0])
        best_individuals_csvs.append(best_individuals)
        max_final.append(np.max(list_max))
        min_final.append(np.min(list_min))
        avg_final.append(np.mean(list_avg))
        std_final.append(np.std(fitness_final))
        fitness_csvs.append(fitness_final)
        pareto_csvs.append(pareto_csv)        
    
    max_final.append(np.mean(max_final))
    min_final.append(None)
    avg_final.append(None)
    std_final.append(None)
    
    result['Max'] = max_final 
    result['Min'] = min_final
    result['Mean'] = avg_final 
    result['STD'] = std_final 
    
    nameee = '_PMut' + str(MUTPB) + '_indpb' + str(indpbMut) + '_Cross' + CrossType + '_PCross' + str(CXPB) + '_Sel' + SelType
    result.to_csv('ACI_Project2_2324_Data/tuning_3_4/' + 'results' + nameee + '.csv', index = None, header=True, encoding='utf-8')
    
        
def main3_4_2(start_date_training, end_date_training):
    
    train_result = pd.DataFrame()
    train_result['Stocks'] = csvs_names
    
    test_result = pd.DataFrame()
    test_result['Stocks'] = csvs_names
    
    test_max_final = []
    test_min_final = []
    test_avg_final = []
    test_std_final = []
    
    train_max_final = []
    train_min_final = []
    train_avg_final = []
    train_std_final = []
    
    best_individuals_csvs = []
    fitness_csvs = []
    eval_csvs = []
    
    for name in csvs_names:
        
        list_max = []
        list_min = []
        list_avg = []
        list_std = []
        fitness_final = []
        best_individuals = []
        eval_csv = []
        for i in range(N_RUNS):
            random.seed(i)
            max, min, avg, std, best_individual, fitness, _ = oa_csv(name, start_date_training, end_date_training)
            best_individuals.append(best_individual)
            list_max.append(max)
            list_min.append(min)
            list_avg.append(avg)
            list_std.append(std)
            fitness_final.append(fitness[0])
            eval_csv.append(evalROI_DD(best_individual, name, '2020-01-01', '2022-12-31')) #training
        best_individuals_csvs.append(best_individuals)
        
        eval_csvs.append(eval_csv)
        
        train_max_final.append(np.max(list_max))
        train_min_final.append(np.min(list_min))
        train_avg_final.append(np.mean(list_avg))
        train_std_final.append(np.std(fitness_final))
        fitness_csvs.append(fitness_final)  
        
        test_max_final.append(np.max(eval_csv))
        test_min_final.append(np.min(eval_csv))
        test_avg_final.append(np.mean(eval_csv))
        test_std_final.append(np.std(eval_csv))      
    
    train_result['Max'] = train_max_final 
    train_result['Min'] = train_min_final
    train_result['Mean'] = train_avg_final 
    train_result['STD'] = train_std_final 
    train_result.to_csv('ACI_Project2_2324_Data/' + 'train_results_3_4_2' + '.csv', index = None, header=True, encoding='utf-8')
    
    test_result['Max'] = test_max_final 
    test_result['Min'] = test_min_final
    test_result['Mean'] = test_avg_final 
    test_result['STD'] = test_std_final 
    test_result.to_csv('ACI_Project2_2324_Data/' + 'test_results_3_4_2' + '.csv', index = None, header=True, encoding='utf-8')

import time

if __name__ == "__main__":
    start_time = time.time()
    
    main3_4_1('2020-01-01', '2022-12-31')
    # main3_4_2('2011-01-01', '2019-12-31')
    
    time_program = time.time() - start_time
    print("--- %s seconds ---" % (time_program))
    