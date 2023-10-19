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

import pandas as pd


AAL = pd.read_csv('ACI_Project2_2324_Data/AAL.csv', encoding='utf-8') 
AAPL = pd.read_csv('ACI_Project2_2324_Data/AAPL.csv', encoding='utf-8') 
AMZN = pd.read_csv('ACI_Project2_2324_Data/AMZN.csv', encoding='utf-8') 
BAC = pd.read_csv('ACI_Project2_2324_Data/BAC.csv', encoding='utf-8') 
F = pd.read_csv('ACI_Project2_2324_Data/F.csv', encoding='utf-8') 
GOOG = pd.read_csv('ACI_Project2_2324_Data/GOOG.csv', encoding='utf-8') 
IBM = pd.read_csv('ACI_Project2_2324_Data/IBM.csv', encoding='utf-8') 
INTC = pd.read_csv('ACI_Project2_2324_Data/INTC.csv', encoding='utf-8') 
NVDA = pd.read_csv('ACI_Project2_2324_Data/NVDA.csv', encoding='utf-8') 
XOM = pd.read_csv('ACI_Project2_2324_Data/XOM.csv', encoding='utf-8')


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


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def get_n_days():
    r = random.randint(1, 3)
    return r * 7

def get_multiple_five():
    r = random.randint(1, 20)
    return r * 5

# Attribute generator 
toolbox.register("n_days", get_n_days)
toolbox.register("multiple_five", get_multiple_five)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.n_days, 2, toolbox.multiple_five, 4)  # beginLP tem de ser sempre menor que o endLP

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalROI(individual):
    RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    
    # Para cada csv calcular ROI (2020-2022)
    # cortar df para periodo 20-22
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    
    AAL_2020_22 = AAL[(AAL['Date'] >= start_date) & (AAL['Date'] <= end_date)].reset_index(drop=True)
        
    previous_date_short = None
    previous_date_long = None
    begin_LP = -1 # valor de compra de ações
    end_LP = -1 # valor de venda de ações
    begin_SP = -1
    end_SP = -1
    
    ROI_short = 0
    ROI_long = 0

    for index, row in AAL_2020_22.iterrows():
        current_date = row['Date']

        if(row['RSI'] <= LB_LP and begin_LP == -1):
            begin_LP = row['Open']
            previous_date_long = current_date
        if(row['RSI'] >= UP_SP and begin_SP == -1):
            begin_SP = row['Open']
            previous_date_short = current_date
        
        if(row['RSI'] >= UP_LP):
            end_LP = row['Open']
            if(not(begin_LP == -1 and end_LP == -1)):
                ROI_long += ((end_LP - begin_LP)/begin_LP) * 100
        
        if(row['RSI'] <= LB_SP):
            end_SP = row['Open']
            ROI_short += ((begin_SP - end_SP)/begin_SP) * 100
        
        if current_date - previous_date_long >= timedelta(RSI_long):
            if (begin_LP > 0) and (end_LP == -1):
                end_LP = row['Close']
            if(not(begin_LP == -1 and end_LP == -1)):
                ROI_long += ((end_LP - begin_LP)/begin_LP) * 100
            
            begin_LP = -1
            end_LP = -1
                
        if current_date - previous_date_short >= timedelta(RSI_short):        
            if (begin_SP > 0) and (end_SP == -1):
                end_SP = row['Close']
            if(not(begin_SP == -1 and end_SP == -1)):    
                ROI_short += ((begin_SP - end_SP)/begin_SP) * 100 
            
            begin_SP = -1
            end_SP = -1

    #chegámos ao fim dos dados: se estivermos a meio de um investimento, temos de o terminar
    
    if (begin_SP > 0) and (end_SP == -1):
        end_SP = row['Close']   
        ROI_short += ((begin_SP - end_SP)/begin_SP) * 100   
    
    if (begin_LP > 0) and (end_LP == -1):
        end_LP = row['Close']
        ROI_long += ((end_LP - begin_LP)/begin_LP) * 100  
    
    
    # Retornar media dos ROIs de cada csv
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
toolbox.register("mutate", tools.mutGaussian,mu=0,sigma=0.05, indpb=0.05)
# toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

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

def main():
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=100) #menor que 144

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Initialize hall of fame     
    hof = tools.HallOfFame(1)
    
    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        # A new generation
        g = g + 1
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
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        hof.update(pop)
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        
        best_ind_gen = tools.selBest(pop, 1)[0]
        print("Best individual in generation %d: %s, %s" % (g, best_ind_gen, best_ind_gen.fitness.values))
        print("Hall of fame: {} {}".format(hof[0], hof[0].fitness.values[0]))
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


import time
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))

N_RUNS = 30
if __name__ == "__main__":
    start_time = time.time()
    for i in range(N_RUNS):
        random.seed(i)
        main()
    
    av_time = (time.time() - start_time) / N_RUNS
    print("--- %s seconds ---" % (av_time))
    
    

# Results

# (cxTwoPoint,mutFlipBit): 0.813s
# (cxTwoPoint,mutGaussian): 0.960s
# (cxTwoPoint,mutShuffleIndexes): 0.626s

# (cxOnePoint,mutFlipBit): 0.951s
# (cxOnePoint,mutGaussian): 1.278s
# (cxOnePoint,mutShuffleIndexes): 0.609s

# (cxPartialyMatched,mutFlipBit): 1.94s
# (cxPartialyMatched,mutGaussian): NAN
# (cxPartialyMatched,mutShuffleIndexes): 21.2s
