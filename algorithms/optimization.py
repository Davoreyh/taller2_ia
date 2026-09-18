import math
import random

from optimization.problem import SmartGridOptimizationProblem
from optimization.result import Configuration, OptimizationResult


def configuration_score(
    problem: SmartGridOptimizationProblem, configuration: Configuration
) -> float:
    """
    Combina cobertura, redundancia y exposición en un puntaje a maximizar.

    Tips:
    - Use problem.score_components(configuration); ya retorna cobertura,
      redundancia y exposición en ese orden.
    """
    cobertura, redundancia, exposicion = problem.score_components(configuration)
    return cobertura - redundancia - exposicion


def hill_climbing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    max_iterations: int = 500,
) -> OptimizationResult:
    """
    Ejecuta ascenso de colina con mejora estricta.

    Debe examinar todos los vecinos, seleccionar el de mayor puntaje y
    conservar el orden entregado por el problema para desempatar. La búsqueda
    termina cuando no existe una mejora estricta o se alcanza el límite.

    Tips:
    - problem.neighbors(current) retorna vecinos válidos en el orden que debe
      usarse para desempatar.
    - Cada llamada a configuration_score(...) cuenta como una evaluación.
    - Inicialice los historiales con la configuración inicial y agregue solo las
      mejoras aceptadas antes de retornar el OptimizationResult.
    """
    
    #estado y puntaje actual
    estado_actual = initial_configuration
    puntaje_actual = configuration_score(problem, initial_configuration)
    
    evaluaciones = 1
    
    #historial de estados y puntajes 
    historial = [estado_actual]
    historial_puntajes= [puntaje_actual]
    
    #hill climbing 
    optimo = False
    iteraciones = 0
    while not optimo and iteraciones < max_iterations:
        mejor_estado = estado_actual
        mejor_puntaje = puntaje_actual
        
        for vecino in problem.neighbors(estado_actual):
            puntaje = configuration_score(problem,vecino )
            evaluaciones += 1
            
            if puntaje > mejor_puntaje:
                mejor_estado = vecino
                mejor_puntaje = puntaje
                
        if mejor_puntaje > puntaje_actual:   
            estado_actual = mejor_estado
            puntaje_actual = mejor_puntaje
            historial.append(estado_actual)
            historial_puntajes.append(puntaje_actual)
                
        else:
            optimo = True
        
        iteraciones+=1
        
    return OptimizationResult(estado_actual, puntaje_actual, evaluaciones, iteraciones, historial, historial_puntajes)


def cooling_schedule(initial_temperature: float, cooling_rate: float, iteration: int) -> float:
    """
    Retorna el programa geométrico T(t) = T0 * alpha**t.

    Esta función se invoca desde simulated_annealing en cada iteración.
    """
    return initial_temperature * (cooling_rate**iteration)



def simulated_annealing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    initial_temperature: float = 20.0,
    cooling_rate: float = 0.97,
    max_iterations: int = 500,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta recocido simulado para un problema de maximización.

    Debe proponer un vecino aleatorio por iteración, aceptar siempre las
    mejoras y aplicar exp(delta / temperature) en los demás casos. El estado
    actual y el mejor estado encontrado deben conservarse por separado.

    Tips:
    - Seleccione el candidato con rng.choice(problem.neighbors(current)) y use
      exclusivamente rng para conservar la reproducibilidad.
    - Obtenga la temperatura con cooling_schedule(...) y calcule la aceptación
      con delta = puntaje_candidato - puntaje_actual y math.exp(...).
    - Mantenga separados el estado actual y el mejor encontrado; registre el
      estado actual después de cada intento, incluso si se rechaza.
    - Detenga la ejecución cuando la temperatura alcance minimum_temperature.
    """
    rng = rng or random.Random()
    minimum_temperature = 1e-9
    
    #estado incial
    evaluaciones = 1
    estado_actual = initial_configuration
    puntaje_actual = configuration_score(problem, initial_configuration)
    
    #historiales
    historial_estados = [estado_actual]
    historial_puntajes = [puntaje_actual]
    
    #ciclo principal
    mejor_estado = estado_actual
    mejor_puntaje = puntaje_actual
    i = 0 
    frio = False
    while i < max_iterations and not frio:
        temp = cooling_schedule(initial_temperature, cooling_rate, i)
        
        if temp > minimum_temperature:
            estado_candidato = rng.choice(problem.neighbors(estado_actual))
            puntaje_candidato = configuration_score(problem, estado_candidato)
            evaluaciones +=1
            delta = puntaje_candidato - puntaje_actual
            
            if (delta > 0):
                 estado_actual = estado_candidato
                 puntaje_actual = puntaje_candidato
                                
                 if puntaje_candidato > mejor_puntaje:
                     mejor_estado = estado_candidato
                     mejor_puntaje = puntaje_candidato
                 
            elif(delta <= 0 and rng.random() < math.exp(delta/temp)):
                estado_actual = estado_candidato
                puntaje_actual = puntaje_candidato
  
            historial_estados.append(estado_actual)
            historial_puntajes.append(puntaje_actual)
            
            i +=1
        
        else:
            frio = True 
            

        
    return OptimizationResult(mejor_estado, mejor_puntaje, evaluaciones, i, historial_estados, historial_puntajes)
        

def one_point_crossover(
    parent1: Configuration, parent2: Configuration, rng: random.Random
) -> tuple[Configuration, Configuration]:
    """
    Realiza un cruce de un punto y retorna dos descendientes.

    La reparación de la cantidad de módulos se realiza posteriormente.

    Tips:
    - Seleccione con rng un corte interior, entre las posiciones 1 y len-1.
    - Cada descendiente combina el prefijo de un padre con el sufijo del otro.
    - Retorne tuplas y no repare aquí los descendientes.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud")
    if len(parent1) < 2:
        return parent1, parent2

    # TODO: Add your code here
    raise NotImplementedError("Punto 3: implemente one_point_crossover")


def swap_mutation(
    individual: Configuration, mutation_probability: float, rng: random.Random
) -> Configuration:
    """
    Aplica mutación por intercambio con la probabilidad indicada.

    Cuando ocurre una mutación, intercambia un bit activo y uno inactivo para
    conservar la cantidad de módulos instalados.

    Tips:
    - Use rng.random() para decidir si se aplica la mutación.
    - Identifique por separado los índices activos e inactivos y seleccione uno
      de cada grupo con rng.choice(...).
    - Si alguno de los dos grupos está vacío, no hay un intercambio posible.
    - Retorne una tupla nueva; no modifique el individuo recibido.
    """
    # TODO: Add your code here
    raise NotImplementedError("Punto 3: implemente swap_mutation")


def genetic_algorithm(
    problem: SmartGridOptimizationProblem,
    population_size: int = 40,
    generations: int = 100,
    mutation_probability: float = 0.05,
    elite_size: int = 2,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta un algoritmo genético generacional.

    Debe integrar la población inicial, la selección por torneo, el cruce, la
    reparación, la mutación y el elitismo entregados por el proyecto. Retorna
    el mejor individuo encontrado durante toda la ejecución.

    Tips:
    - Use problem.initial_population(...), problem.tournament_select(...) y
      problem.repair_configuration(...) para las operaciones ya entregadas.
    - Aplique one_point_crossover(...) antes de reparar y swap_mutation(...)
      después de la reparación.
    - Conserve los mejores individuos por elitismo y registre en los historiales
      el mejor global de cada generación.
    """
    rng = rng or random.Random()
    if population_size < 2:
        raise ValueError("La población debe tener al menos dos individuos")
    if generations < 0:
        raise ValueError("El número de generaciones no puede ser negativo")
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
    if not 0 <= elite_size <= population_size:
        raise ValueError("elite_size debe estar entre 0 y population_size")

    # TODO: Add your code here
    raise NotImplementedError("Punto 3: implemente genetic_algorithm")
