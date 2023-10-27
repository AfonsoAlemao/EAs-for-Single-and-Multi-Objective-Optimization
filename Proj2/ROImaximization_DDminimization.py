import random
import math
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

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

# Global Variables
GENERATIONS = 160
INITIAL_POPULATION = 64
N_RUNS = 30
INFINITY = np.inf
GAP_ANALYZED = 10 # Early stopping patience between generations
PERF_THRESHOLD = 1 # Threshold for Early stopping analysis

# The goal is to maximize the ROI and minimize the DD
creator.create("FitnessMaxMin", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMaxMin)

toolbox = base.Toolbox()

# Generates 7, 14 or 21 days
def get_n_days():
    r = random.randint(1, 3)
    return r * 7

# Generates an integer between 0 and 100, considering num as lower/higher constraint 
def get_multiple_five(num, low_high):
    if low_high == 'low':
        # Get 5r < num
        r = random.randint(0, num / 5 - 1)
    else:
        # Get 5r >= num
        r = random.randint(num / 5, 20)
    return r * 5

# Generates individual = [RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP]
def get_multiple_five_and_get_n_days():    
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
def generate():
    part = creator.Individual() 
    ind = get_multiple_five_and_get_n_days()

    for i in ind:
        part.append(i)
    
    return part

toolbox.register("individual", generate)

# Define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# The goal ('fitness') function to be maximized is ROI
# and the goal function to be minimized is DD
def evalROI_DD(individual, csv_name, start_date, end_date):
    RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    
    csv = csvs[csvs_names.index(csv_name)]
    
    ROI_short = 0
    ROI_long = 0
    
    csv_2020_22 = csv[(csv['Date'] >= start_date) & (csv['Date'] <= end_date)].reset_index(drop=True)

    begin_LP = -1 # Share purchase value for LP
    end_LP = -1 # Share sale value for LP
    begin_SP = -1 # Share sale value for LP
    end_SP = -1 # Share purchase value for SP
    
    # Select RSI to use
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

    for _, row in csv_2020_22.iterrows():
        # Checks the feasibility of starting a LP/SP strategy
        if(row[col_RSI_long] <= LB_LP and begin_LP == -1):
            begin_LP = row['Close']
            minLong = begin_LP
        if(row[col_RSI_short] >= UP_SP and begin_SP == -1):
            begin_SP = row['Close']
            maxShort = begin_SP
        
        # Check the minimum/maximum share value during current LP/SP strategy 
        # in order to obtain the corresponding DD
        if(begin_LP !=-1 and row['Close'] < minLong):
            minLong = row['Close']
        if(begin_SP !=-1 and row['Close'] > maxShort):
            maxShort = row['Close']
            
        # Checks the feasibility of finalizing a LP/SP strategy
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
        
    # Returns global considered ROI and Drawdown
    return (ROI_short + ROI_long)/2, DD

# Operator registration: register the goal / fitness function
toolbox.register("evaluate", evalROI_DD)

# Register the crossover operator
toolbox.register("mate", tools.cxOnePoint)

# Other crossover operators tested
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mate", tools.cxUniform, indpb = 0.5)

# Create blending in the features relative to bounds
# assuring that the new individuals have feasible values
def blending(individual1, individual2):
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

# Register mutation operator:

# Mutates LB_LP, UB_LP, LB_SP and UB_SP features with a probability of indpb, 
# by introducing a small variation of ±5, if possible.
def mutCustom_2(individual, indpb):
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

# Other mutation operators tested
toolbox.register('mutate', mutCustom_2, indpb = 0.5) 

# Mutates each feature with a probability of indpb and assigns a new 
# random value within the entire range of the feature.  
def mutCustom(individual, indpb):
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

# toolbox.register('mutate', mutCustom, indpb = 0.5)

# Operator for selecting individuals for breeding the next
# generation
toolbox.register("select", tools.selNSGA2)

# Execute the optimization algorithm for a .csv file, in a certain data range
def oa_csv(csv_name, start_date_training, end_date_training):
    # Initialize the pareto front
    pareto = tools.ParetoFront()
    
    # Create an initial population
    pop = toolbox.population(n=INITIAL_POPULATION) 
    
    # Apply selection
    pop = toolbox.select(pop, len(pop))
    
    # CXPB: tuned probability with which two individuals are crossed
    # MUTPB: tuned probability for mutating an individual
    CXPB, MUTPB = 0.7, 0.1
    
    # print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = []
    for individual in pop:
        fitnesses.append(toolbox.evaluate(individual, csv_name, start_date_training, end_date_training))
        
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses 
    fits = [ind.fitness.values[0] for ind in pop]
    
    # Update pareto front
    pareto.update(pop)
    
    # Variable keeping track of the number of generations
    g = 0
    
    # Initialize hall of fame     
    hof = tools.HallOfFame(1)
    
    # Early stopping monitoring initializations
    improve_perf = INFINITY
    max_by_generationsROI = []
    min_by_generationsDD = []
    
    # Evolution with a Early stopping strategy with patience = GAP_ANALYSED, and monitor = improve_perf
    while g < GENERATIONS and improve_perf > PERF_THRESHOLD: 
        # A new generation
        g = g + 1
        
        # if (g % 10 == 0):
        #     print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = tools.selTournamentDCD(pop, len(pop))
        
        # Other selection methods tested
        # offspring = tools.selTournament(pop, len(pop), tournsize=2)
        # offspring = tools.selTournament(pop, len(pop), tournsize=3)
        # offspring = tools.selTournament(pop, len(pop), tournsize=4)
        # offspring = tools.selTournament(pop, len(pop), tournsize=8)
        # offspring = tools.selRoulette(pop, len(pop))
        # offspring = tools.selRandom(pop, len(pop))
        # offspring = tools.selBest(pop, len(pop))
        # offspring = tools.selStochasticUniversalSampling(pop, len(pop))
        # offspring = tools.selLexicase(pop, len(pop))
        
        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # Fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            # Mutate an individual with probability MUTPB
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
        
        # The population is entirely replaced by selection applied to pop + offspring
        pop = toolbox.select(pop + offspring, INITIAL_POPULATION)
        
        # Gather all the fitnesses in one list
        fits = [ind.fitness.values for ind in pop]
        
        # Update hall of fame and pareto front
        pareto.update(pop)
        hof.update(pop)
        
        # Auxiliar lists for early stopping
        max_by_generationsROI.append(max(fits, key=lambda x: x[0])[0])
        min_by_generationsDD.append(min(fits, key=lambda x: x[1])[1])
        
        # Computes Early stopping monitor
        if g > GAP_ANALYZED:
            improve_perf_ROI = max_by_generationsROI[g - 1] - max_by_generationsROI[g - GAP_ANALYZED - 1]  
            improve_perf_DD = - (min_by_generationsDD[g - 1] - min_by_generationsDD[g - GAP_ANALYZED - 1])  
            improve_perf = improve_perf_ROI + improve_perf_DD
    
    # print("-- End of (successful) evolution --")
   
    # Obtain the elements from pareto front with maximum ROI and with minimum DD
    index_ROI, fit_maxROI = max(enumerate(fits), key=lambda x: (x[1][0], -x[1][1]))
    index_DD, fit_minDD = min(enumerate(fits), key=lambda x: (x[1][1], -x[1][0]))
    best_ind = [pop[index_ROI], pop[index_DD]]
    
    # print("Best individual are %s, %s and %s, %s" % (best_ind[0], best_ind[0].fitness.values, best_ind[1], best_ind[1].fitness.values))
    # print('Obtained at generation ', g)
    # gen_results = [(max_by_generationsROI[i], min_by_generationsDD[i]) for i in range(len(max_by_generationsROI))]
    # print('Succesive generations', gen_results)
    
    return fit_maxROI, fit_minDD, best_ind, pareto

# Generates the Pareto Front graphs for each of the tests and superimpose them in a single plot
def generate_paretos(pareto_csvs):
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

# Exercise 3.4.1: MOO Applied to Maximize Return on Investment (ROI) and Minimize Drawdown using Technical
# Indicators.
# Apply MO OA to the complete data series values of each csv from Jan 2020 until Dec 2022
def main3_4_1(start_date_training, end_date_training):
    result = pd.DataFrame()
    result['Stocks'] = csvs_names
    
    # Initializations
    final_maxROI_ROI = []
    final_maxROI_DD = []
    final_minDD_ROI = []
    final_minDD_DD = []
    pareto_csvs = []
    
    # For each csv, execute N_RUNS obtaining the best individuals from each one (pareto front
    # and the elements from the pareto front with maximum ROI and with minimum DD)
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
        
        # Get from the obtained individuals the ones with the maximum ROI and with the minimum DD
        fit_maxROI = max(list_maxROI, key=lambda x: (x[0], -x[1]))
        fit_minDD = min(list_minDD, key=lambda x: (x[1], -x[0]))
      
        final_maxROI_ROI.append(fit_maxROI[0])
        final_maxROI_DD.append(fit_maxROI[1])
        final_minDD_ROI.append(fit_minDD[0])
        final_minDD_DD.append(fit_minDD[1]) 
    
    result['MaxROI_ROI'] = final_maxROI_ROI
    result['maxROI_DD'] = final_maxROI_DD
    result['minDD_ROI'] = final_minDD_ROI
    result['minDD_DD'] = final_minDD_DD
    
    result.to_csv('ACI_Project2_2324_Data/results/' + 'results_3_4_1' + '.csv', index = None, header=True, encoding='utf-8')
    
    generate_paretos(pareto_csvs)

# Exercise 3.4.2: MOO Train and Test Scheme
# Train period from Jan 2011 until Dec 2019 and test period from Jan 2020 until Dec 2022
def main3_4_2(start_date_training, end_date_training):
    train_result = pd.DataFrame()
    train_result['Stocks'] = csvs_names
    
    test_result = pd.DataFrame()
    test_result['Stocks'] = csvs_names
    
    # Initializations
    test_final_maxROI_ROI = []
    test_final_maxROI_DD = []
    test_final_minDD_ROI = []
    test_final_minDD_DD = []
    train_final_maxROI_ROI = []
    train_final_maxROI_DD = []
    train_final_minDD_ROI = []
    train_final_minDD_DD = []
    
    # For each csv, execute N_RUNS obtaining the best individuals from each one (the whole pareto front) and test them
    for name in csvs_names:
        list_maxROI = []
        list_minDD = []
        eval_csv = []
        print(name)
        for i in range(N_RUNS):
            random.seed(i)
            # Training
            maxROI, minDD, _, pareto = oa_csv(name, start_date_training, end_date_training)
            list_maxROI.append(maxROI)
            list_minDD.append(minDD)
            # Testing
            for ind in pareto.items:
                eval_csv.append(evalROI_DD(ind, name, '2020-01-01', '2022-12-31')) 
                
        train_fit_maxROI = max(list_maxROI, key=lambda x: (x[0], -x[1]))
        train_fit_minDD = min(list_minDD, key=lambda x: (x[1], -x[0]))
        
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
    train_result.to_csv('ACI_Project2_2324_Data/results/' + 'train_results_3_4_2' + '.csv', index = None, header=True, encoding='utf-8')
    
    test_result['MaxROI_ROI'] = test_final_maxROI_ROI
    test_result['maxROI_DD'] = test_final_maxROI_DD
    test_result['minDD_ROI'] = test_final_minDD_ROI
    test_result['minDD_DD'] = test_final_minDD_DD
    test_result.to_csv('ACI_Project2_2324_Data/results/' + 'test_results_3_4_2' + '.csv', index = None, header=True, encoding='utf-8')

if __name__ == "__main__":
    start_time = time.time()
    
    main3_4_1('2020-01-01', '2022-12-31')
    main3_4_2('2011-01-01', '2019-12-31')
    
    time_program = time.time() - start_time
    print("--- %s seconds ---" % (time_program))
    