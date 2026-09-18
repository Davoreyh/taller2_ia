from abc import ABC, abstractmethod

from algorithms.evaluation import evaluation_function
from world.game_state import GameState



class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""
    MAX = 0
    MIN = 1
    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1) y cada
        acción consume un ply. Debe respetar el orden de las acciones legales,
        usar evaluation_function en terminales y cortes, y contar cada estado
        procesado una vez en self.nodes_evaluated, incluida la raíz.

        Tips:
        - Use state.get_legal_actions(agent_index) y
          state.generate_successor(agent_index, action) para expandir el árbol.
        - Compruebe state.is_win(), state.is_lose() y el corte de profundidad;
          evalúe esos estados con evaluation_function(state).
        - El siguiente agente es (agent_index + 1) % state.get_num_agents().
          depth=1 incluye una acción de MAX y depth=2 una de MAX y una de MIN.
        - Reinicie las métricas y cuente una vez cada estado procesado, incluida
          la raíz. Retorne la acción de MAX y conserve la primera en los empates.
        """
        # TODO: Add your code here
        self.nodes_evaluated = 0
        action, util = self.evaluate_max(state, self.depth)
        return action
      
    def evaluate_max(self, state: GameState, max_depth: int) -> tuple[str, float]:
      """Obtiene una tupla con la información de la acción recomendada para MAX en un estado dado

      Args:
          state (GameState): Estado de la partida
          max_depth (int): La máxima profundidad que puede tener el árbol

      Returns:
          tuple[str, float]: Tupla de la forma (accion, utilidad), con la accion recomendada
          a realizar según el estado inicial y el valor de su utilidad.
      """
      self.nodes_evaluated += 1
      if state.is_lose() or state.is_win or max_depth == 0:
        return (None, evaluation_function(state), self.nodes_evaluated)
      best_util = float("-inf")
      best_action = None
      actions = state.get_legal_actions(self.MAX)
      for action in actions:
        next_action, next_util = self.evaluate_min(state.generate_successor(self.MAX, action), max_depth-1)
        if next_util > best_util:
          best_util = next_util
          best_action = action
      return (best_action, best_util)
      
      
    def evaluate_min(self, state: GameState, max_depth: int)->tuple[str, float]:
      """Obtiene una tupla con la información de la acción recomendada para MAX en un estado dado

      Args:
          state (GameState): Estado de la partida
          max_depth (int): La máxima profundidad que puede tener el árbol

      Returns:
          tuple[str, float]: Tupla de la forma (accion, utilidad), con la accion recomendada
          a realizar según el estado inicial y el valor de su utilidad.
      """
      self.nodes_evaluated += 1
      if state.is_lose() or state.is_win or max_depth == 0:
        return (None, evaluation_function(state), self.nodes_evaluated)
      best_util = float("inf")
      best_action = None
      actions = state.get_legal_actions(self.MIN)
      for action in actions:
        next_action, next_util = self.evaluate_max(state.generate_successor(self.MIN, action), max_depth-1)
        if next_util < best_util:
          best_util = next_util
          best_action = action
      return (best_action, best_util)
      

class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        self.nodes_evaluated = 0
        alpha = float("-inf")
        beta = float("inf")
        action, util = self.evaluate_max(state, self.depth, alpha, beta)
        return action
      
    def evaluate_max(self, state: GameState, max_depth: int, alpha:float, beta: float) -> tuple[str, float]:
      """Obtiene la mejor accion para MAX, optimizando la búsqueda con poda alpha-beta

      Args:
          state (GameState): Estado de la partida
          max_depth (int): La máxima profundidad que puede tener el árbol
          alpha (float): Utilidad de la mejor acción encontrada para MAX
          beta (float): Utilidad de la mejor acción encontrada para MIN

      Returns:
          tuple[str, float]: Tupla estilo (accion, utilidad) de la mejor acción para MAX
      """
      self.nodes_evaluated += 1
      if state.is_lose() or state.is_win or max_depth == 0:
        return (None, evaluation_function(state), self.nodes_evaluated)
      best_action = None
      actions = state.get_legal_actions(self.MAX)
      for action in actions:
        next_action, next_util = self.evaluate_min(state.generate_successor(self.MAX, action), max_depth-1, alpha, beta)
        if beta >= next_util:
          break
        if next_util > alpha:
          alpha = next_util
          best_action = action
      return (best_action, alpha)
      
      
    def evaluate_min(self, state: GameState, max_depth: int, alpha:float, beta: float) -> tuple[str, float]:
      """Obtiene la mejor accion para MAX, optimizando la búsqueda con poda alpha-beta

      Args:
          state (GameState): Estado de la partida
          max_depth (int): La máxima profundidad que puede tener el árbol
          alpha (float): Utilidad de la mejor acción encontrada para MAX
          beta (float): Utilidad de la mejor acción encontrada para MIN

      Returns:
          tuple[str, float]: Tupla estilo (accion, utilidad) de la mejor acción para MAX
      """
      self.nodes_evaluated += 1
      if state.is_lose() or state.is_win or max_depth == 0:
        return (None, evaluation_function(state), self.nodes_evaluated)
      best_action = None
      actions = state.get_legal_actions(self.MIN)
      for action in actions:
        next_action, next_util = self.evaluate_max(state.generate_successor(self.MIN, action), max_depth-1, alpha, beta)
        if next_util <= alpha:
          break
        if next_util < beta:
          beta = next_util
          best_action = action
      return (best_action, beta)
