# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
import math

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        pacPos = newPos

        # Calculate manhattan distance of food from Pacman
        foodDist = math.inf
        for food in currentGameState.getFood().asList():
            foodDist = min(foodDist, manhattanDistance(pacPos, food))
            if Directions.STOP in action:
                return -math.inf

        # Check if Ghost and Pacman collide
        for ghostState in newGhostStates:
            # assign position
            ghostPos = ghostState.getPosition()
            # if collision
            if ghostPos == pacPos:
                return -math.inf

        return 1.0/(1.0 + foodDist)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def helper(s, p, d):
            n = s.getNumAgents()
            if p == n:
                p = 0
                d += 1
            b = None
            if d == self.depth or s.isWin() or s.isLose() or s.getLegalActions(p) == 0:
                return b, self.evaluationFunction(s)

            if p == 0:
                v = -math.inf
            if p >= 1:
                v = math.inf
            for m in s.getLegalActions(p):
                np = s.generateSuccessor(p, m)
                nm, nv = helper(np, p + 1, d)
                if p == 0 and nv > v:
                    v, b = nv, m
                if p >= 1 and nv < v:
                    v, b = nv, m
            return b, v

        return helper(gameState, 0, 0)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def helper(s, p, d, a, b):

            n = s.getNumAgents()
            if p == n:
                p = 0
                d += 1
            bm = None
            if d == self.depth or s.isWin() or s.isLose() or s.getLegalActions(p) == 0:
                return bm, self.evaluationFunction(s)

            if p == 0:
                v = -math.inf
            if p >= 1:
                v = math.inf
            for m in s.getLegalActions(p):
                np = s.generateSuccessor(p, m)
                nm, nv = helper(np, p + 1, d, a, b)
                if p == 0:
                    if nv > v:
                        v, bm = nv, m
                    if v >= b:
                        return bm, v
                    a = max(a, v)
                if p >= 1:
                    if nv < v:
                        v, bm = nv, m
                    if v <= a:
                        return bm, v
                    b = min(b, v)
            return bm, v
        return helper(gameState, 0, 0, -math.inf, math.inf)[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def helper(s, p, d):
            n = s.getNumAgents()
            if p == n:
                p = 0
                d += 1
                bm = None
            if d == self.depth or s.isWin() or s.isLose() or s.getLegalActions(p) == 0:
                return bm, self.evaluationFunction(s)
    
            if p == 0:
                v = -math.inf
            if p >= 1:
                v = 0
            a = s.getLegalActions(p)
            for m in a:
                np = s.generateSuccessor(p, m)
                nm, nv = helper(np, p + 1, d)
                if p == 0 and nv > v:
                    v, bm = nv, m
                if p >= 1:
                    v += nv / len(a)
            return bm, v
    
        return helper(gameState, 0, 0)[0]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Favourable states are :
      1. States with less food pellets
      2. States where pacman is close to the capsules (ghosts are scared so he can eat them)
      Unfavourable states are:
      3. State where Pacman is close to ghost
      4. State where Pacman gets killed by a ghost
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    curFoodList = currentGameState.getFood().asList()
    curFoodCount = currentGameState.getNumFood()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
    curCap = currentGameState.getCapsules()
    curScore = currentGameState.getScore()

    ghostDist = math.inf
    scaredGhosts = 0
    for ghostState in curGhostStates:
        ghostPos = ghostState.getPosition()
        if pacPos == ghostPos:
            return -math.inf
        else:
            ghostDist = min(ghostDist, manhattanDistance(pacPos, ghostPos))
        if ghostState.scaredTimer != 0:
            scaredGhosts += 1

    capDist = math.inf
    for capsuleState in curCap:
        capDist = min(capDist, manhattanDistance(pacPos, capsuleState))

    ghostDist = 1.0 / (1.0 + (ghostDist / (len(curGhostStates))))
    scaredGhosts = 1.0 / (1.0 + scaredGhosts)
    foodLeft = 1.0 / (curFoodCount + 1.0)
    capDist = 1.0 / (1.0 + len(curCap))

    return curScore + (foodLeft + ghostDist + capDist + scaredGhosts)


# Abbreviation
better = betterEvaluationFunction

# if __name__ == "__main__":
#     ReflexAgent.evaluationFunction()
