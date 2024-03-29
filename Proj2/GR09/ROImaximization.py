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
GENERATIONS = 100
INITIAL_POPULATION = 100
N_RUNS = 30
INFINITY = np.inf
GAP_ANALYZED = 10 # Early stopping patience between generations
PERF_THRESHOLD = 1 # Threshold for Early stopping analysis

# The goal is to maximize the ROI 
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

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
def evalROI(individual, csv_name, start_date, end_date):
    RSI_long, RSI_short, LB_LP, UP_LP, LB_SP, UP_SP = individual
    
    csv = csvs[csvs_names.index(csv_name)]
    
    ROI_short = 0
    ROI_long = 0 
    
    csv_2020_22 = csv[(csv['Date'] >= start_date) & (csv['Date'] <= end_date)].reset_index(drop=True)

    begin_LP = -1 # Share purchase value for LP
    end_LP = -1 # Share sale value for LP
    begin_SP = -1 # Share sale value for SP
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

    for _, row in csv_2020_22.iterrows():
        # Checks the feasibility of starting a LP/SP strategy
        if(row[col_RSI_long] <= LB_LP and begin_LP == -1):
            begin_LP = row['Close']
        if(row[col_RSI_short] >= UP_SP and begin_SP == -1):
            begin_SP = row['Close']
        
        # Checks the feasibility of finalizing a LP/SP strategy
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
        
    # Returns global considered ROI
    return (ROI_short + ROI_long)/2,

# Operator registration: register the goal / fitness function
toolbox.register("evaluate", evalROI)

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
            new_individual[i] = (new_individual1[i] + new_individual2[i]) / (2 * factor)
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
toolbox.register("select", tools.selTournament, tournsize=2)

# Other tested operators:
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("select", tools.selTournament, tournsize=4)
# toolbox.register("select", tools.selTournament, tournsize=8)
# toolbox.register("select", tools.selRoulette)
# toolbox.register("select", tools.selRandom)
# toolbox.register("select", selBest)
# toolbox.register("select", selStochasticUniversalSampling)
# toolbox.register("select", selLexicase)

# Execute the optimization algorithm for a .csv file, in a certain data range
def oa_csv(csv_name, start_date_training, end_date_training):
    # Create an initial population
    pop = toolbox.population(n=INITIAL_POPULATION) 

    # CXPB: tuned probability with which two individuals are crossed
    # MUTPB: tuned probability for mutating an individual
    CXPB, MUTPB = 0.9, 0.7
    
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

    # Variable keeping track of the number of generations
    g = 0
    
    # Initialize hall of fame     
    hof = tools.HallOfFame(1)
    
    # Early stopping monitoring
    improve_perf = INFINITY
    max_by_generations = []
    
    # Evolution with an Early stopping strategy with patience = GAP_ANALYSED, and monitor = improve_perf
    while g < GENERATIONS and improve_perf > PERF_THRESHOLD: 
        # A new generation
        g = g + 1
        
        # if (g % 10 == 0):
            # print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
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
        
        # Update hall of fame
        hof.update(pop)
        
        # Gather all the fitnesses in one list
        fits = [ind.fitness.values[0] for ind in pop]
        
        # Auxiliar list for Early stopping
        max_by_generations.append(max(fits))
        
        # Computes Early stopping monitor
        if g > GAP_ANALYZED:
            improve_perf = max_by_generations[g - 1] - max_by_generations[g - GAP_ANALYZED - 1]  
    
    # print("-- End of (successful) evolution --")
    
    # Obtain best individual
    best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    # print('Obtained at generation ', g)
    # print('Succesive generations maximums ', max_by_generations)
    
    return max(fits), best_ind

# Creates 6 histograms, one for each of the optimization variables with all the achieved results on the
# experiments. Each histogram have 300 values (30 (runs) x 10 (stocks))
def generate_histograms(best_individuals):
    best_individuals_copy = [sublist for outer_list in best_individuals for sublist in outer_list]
    data = pd.DataFrame(best_individuals_copy, columns=['RSI_long', 'RSI_short', 'LB_LP', 'UP_LP', 'LB_SP', 'UP_SP'])

    # We normalize the histograms using percentages

    # Possible values
    array_days = [7, 14, 21]
    array_multiples_five = [5*i for i in range(21)]
    
    # Plot RSI_long histogram
    plt.figure(0)
    categories = data['RSI_long'].value_counts().index
    counts = data['RSI_long'].value_counts().values
    counts = [count / sum(counts) for count in counts]
    plt.bar(categories, counts, width=7)
    plt.xlabel('RSI period to apply for long positions')
    plt.ylabel('Relative Frequency')
    plt.title('Histogram of RSI_long')
    plt.xlim([3.5, 24.5])
    plt.xticks(array_days) 
    plt.grid()
    plt.savefig('3_2_hist_boxplot/RSI_long.png')
    
    # Plot RSI_short histogram
    plt.figure(1)
    categories = data['RSI_short'].value_counts().index
    counts = data['RSI_short'].value_counts().values
    counts = [count / sum(counts) for count in counts]
    plt.bar(categories, counts, width=7)
    plt.xlabel('RSI period to apply for short positions')
    plt.ylabel('Relative Frequency')
    plt.title('Histogram of RSI_short')
    plt.xlim([3.5, 24.5])
    plt.xticks(array_days)
    plt.grid()
    plt.savefig('3_2_hist_boxplot/RSI_short.png')
    
    # Plot LB_LP histogram
    plt.figure(2)
    categories = data['LB_LP'].value_counts().index
    counts = data['LB_LP'].value_counts().values
    counts = [count / sum(counts) for count in counts]
    plt.bar(categories, counts, width=5)
    plt.xlabel('Lower band value to open a long position')
    plt.ylabel('Relative Frequency')
    plt.title('Histogram of LB_LP')
    plt.xlim([-2.5, 102.5])
    plt.xticks(array_multiples_five) 
    plt.grid()
    plt.savefig('3_2_hist_boxplot/LB_LP.png')
    
    # Plot UP_LP histogram
    plt.figure(3)
    categories = data['UP_LP'].value_counts().index
    counts = data['UP_LP'].value_counts().values
    counts = [count / sum(counts) for count in counts]
    plt.bar(categories, counts, width=5)
    plt.xlabel('Upper band value to close a long position')
    plt.ylabel('Relative Frequency')
    plt.title('Histogram of UP_LP')
    plt.xlim([-2.5, 102.5])
    plt.xticks(array_multiples_five) 
    plt.grid()
    plt.savefig('3_2_hist_boxplot/UP_LP.png')
    
    # Plot LB_SP histogram
    plt.figure(4)
    categories = data['LB_SP'].value_counts().index
    counts = data['LB_SP'].value_counts().values
    counts = [count / sum(counts) for count in counts]
    plt.bar(categories, counts, width=5)
    plt.xlabel('Lower band value to close a short position')
    plt.ylabel('Relative Frequency')
    plt.title('Histogram of LB_SP')
    plt.xlim([-2.5, 102.5])
    plt.xticks(array_multiples_five) 
    plt.grid()
    plt.savefig('3_2_hist_boxplot/LB_SP.png')
    
    # Plot UP_SP histogram
    plt.figure(5)
    categories = data['UP_SP'].value_counts().index
    counts = data['UP_SP'].value_counts().values
    counts = [count / sum(counts) for count in counts]
    plt.bar(categories, counts, width=5)  
    plt.xlabel('Upper band value to open a short position')
    plt.ylabel('Relative Frequency')
    plt.title('Histogram of UP_SP')
    plt.xlim([-2.5, 102.5])
    plt.xticks(array_multiples_five) 
    plt.grid()
    plt.savefig('3_2_hist_boxplot/UP_SP.png')

# Generates a boxplot graph with the results obtained each of above runs
def generate_boxplots(fitness_csvs):
    #  Uses a min-max normalization so that the boxplot for different stocks can be compared
    max = np.amax(fitness_csvs)
    min = np.amin(fitness_csvs)  
    normalized_fitness_csvs = (np.array(fitness_csvs) - min) / (max - min)            
    
    plt.figure(6)
    plt.boxplot(normalized_fitness_csvs.T, labels=csvs_names)
    plt.xlabel('Stocks')
    plt.ylabel('Normalized ROI')
    plt.title('Normalized ROI Boxplot')
    plt.grid()
    plt.savefig('3_2_hist_boxplot/normalized_boxplot.png')
    return

# Exercise 3.2: SOO Applied to Maximize ROI using Technical Indicators
# Apply OA to the complete data series values of each csv from Jan 2020 until Dec 2022
def main3_2(start_date_training, end_date_training):
    result = pd.DataFrame()
    result['Stocks'] = csvs_names
    
    # Initializations
    max_final = []
    min_final = []
    avg_final = []
    std_final = []
    best_individuals_csvs = []
    fitness_csvs = []
    
    # For each csv, execute N_RUNS obtaining the best individuals from each one
    for name in csvs_names:
        list_max = []
        best_individuals = []
        print(name)
         
        for i in range(N_RUNS):
            random.seed(i)
            max, best_individual = oa_csv(name, start_date_training, end_date_training)
            best_individuals.append(best_individual)
            list_max.append(max)
            
        best_individuals_csvs.append(best_individuals)
        
        # Get from the list of best individuals the max, min, avg and std of the obtained ROI's
        max_final.append(np.max(list_max))
        min_final.append(np.min(list_max))
        avg_final.append(np.mean(list_max))
        std_final.append(np.std(list_max))
        fitness_csvs.append(list_max)        
    
    result['Max'] = max_final 
    result['Min'] = min_final
    result['Mean'] = avg_final 
    result['STD'] = std_final 
    result.to_csv('ACI_Project2_2324_Data/results/' + 'results_3_2' + '.csv', index = None, header=True, encoding='utf-8')
    
    generate_histograms(best_individuals_csvs)
    generate_boxplots(fitness_csvs)

# Exercise 3.3: SOO Train and Test Scheme
# Train period from Jan 2011 until Dec 2019 and test period from Jan 2020 until Dec 2022
def main3_3(start_date_training, end_date_training):
    train_result = pd.DataFrame()
    train_result['Stocks'] = csvs_names
    
    test_result = pd.DataFrame()
    test_result['Stocks'] = csvs_names
    
    # Initializations
    test_max_final = []
    test_min_final = []
    test_avg_final = []
    test_std_final = []
    train_max_final = []
    train_min_final = []
    train_avg_final = []
    train_std_final = []
    
    # For each csv, execute N_RUNS obtaining the best individuals from each one and test them
    for name in csvs_names:
        print(name)
        list_max = []
        eval_csv = []
        for i in range(N_RUNS):
            random.seed(i)
            # Training
            max, best_individual = oa_csv(name, start_date_training, end_date_training)
            list_max.append(max)
            # Testing
            eval_csv.append(evalROI(best_individual, name, '2020-01-01', '2022-12-31')) 
                    
        train_max_final.append(np.max(list_max))
        train_min_final.append(np.min(list_max))
        train_avg_final.append(np.mean(list_max))
        train_std_final.append(np.std(list_max))
        
        test_max_final.append(np.max(eval_csv))
        test_min_final.append(np.min(eval_csv))
        test_avg_final.append(np.mean(eval_csv))
        test_std_final.append(np.std(eval_csv))      
    
    # From training and testing, get from the list of best individuals the max, min, avg and std of the obtained ROI's
    train_result['Max'] = train_max_final 
    train_result['Min'] = train_min_final
    train_result['Mean'] = train_avg_final 
    train_result['STD'] = train_std_final 
    train_result.to_csv('ACI_Project2_2324_Data/results/' + 'train_results_3_3' + '.csv', index = None, header=True, encoding='utf-8')
    test_result['Max'] = test_max_final 
    test_result['Min'] = test_min_final
    test_result['Mean'] = test_avg_final 
    test_result['STD'] = test_std_final 
    test_result.to_csv('ACI_Project2_2324_Data/results/' + 'test_results_3_3' + '.csv', index = None, header=True, encoding='utf-8')

if __name__ == "__main__":
    start_time = time.time()
    
    main3_2('2020-01-01', '2022-12-31')
    main3_3('2011-01-01', '2019-12-31')
    
    time_program = time.time() - start_time
    print("--- %s seconds ---" % (time_program))
    
