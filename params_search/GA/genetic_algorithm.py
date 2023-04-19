import pygad


def get_best_params(function_inputs, fitness_function):
    num_generations = 100 # Number of generations.
    num_parents_mating = 4 # Number of solutions to be selected as parents in the mating pool.

    # To prepare the initial population, there are 2 ways:
    # 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
    # 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
    sol_per_pop = 50 # Number of solutions in the population.
    num_genes = len(function_inputs)

    init_range_low = -1
    init_range_high = 1

    last_fitness = 0

    def callback_generation(ga_function):
        print("---- GA Log ----")
        print("Generation = {generation}".format(generation=ga_function.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_function.best_solution()[1]))
        print("Change     = {change}".format(change=ga_function.best_solution()[1]))
        print("-----------")

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        on_generation=callback_generation
    )
    ga_instance.run()
    ga_instance.plot_fitness()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    return solution, solution_fitness
