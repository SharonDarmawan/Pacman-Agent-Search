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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        score = successorGameState.getScore()

        # Location of closest ghost
        ghostPosList = successorGameState.getGhostPositions()
        distanceToGhost = float('inf')
        for ghostPos in ghostPosList:
            distanceToGhost = min(distanceToGhost, manhattanDistance(newPos, ghostPos)) + 1

        # Location of the closest food pellet
        newFoodList = newFood.asList()
        distanceToFood = float('inf')
        for food in newFoodList:
            distanceToFood = min(distanceToFood, manhattanDistance(newPos, food)) + 1

        # Number of food left
        numFood = successorGameState.getNumFood()

        # If ghost gets too near, increase weight of ghost location distance
        if distanceToGhost <= 2:
            return 10 * score + 5 * (1 / distanceToFood) - 100 * (1 / distanceToGhost) - 10 * numFood

        return 100 * score + 5 * (1 / distanceToFood) - (1 / distanceToGhost) - 100 * numFood

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def miniMax(agentIndex, gameState, depth):
            if depth == self.depth and agentIndex % gameState.getNumAgents() == 0:
                return [self.evaluationFunction(gameState), Directions.STOP]

            if agentIndex % gameState.getNumAgents() == 0:
                return maxValue(agentIndex % gameState.getNumAgents(), gameState, depth)
            else:
                return minValue(agentIndex % gameState.getNumAgents(), gameState, depth)


        def maxValue(agentIndex, gameState, depth):
            value = float('-inf')
            action = Directions.STOP

            #stop the game if pacman is winning or losing
            if gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState), Directions.STOP]

            #iterate through every action to find the next state and value
            for legalAction in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, legalAction)
                nextValue = miniMax(agentIndex + 1, nextState, depth + 1)[0]
                if nextValue > value:
                    value, action = nextValue, legalAction

            return [value, action]

        def minValue(agentIndex, gameState, depth):
            value = float('inf')
            action = Directions.STOP

            #stop the game if pacman is winning or losing
            if gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState), Directions.STOP]

            #iterate through every action to find the next state and value
            for legalAction in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, legalAction)
                nextValue = miniMax(agentIndex + 1, nextState, depth)[0]
                if nextValue < value:
                    value, action = nextValue, legalAction

            return [value, action]

        return miniMax(self.index, gameState, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def miniMax(agentIndex, gameState, depth, alpha, beta):

            if depth == self.depth and agentIndex % gameState.getNumAgents() == 0:
                return [self.evaluationFunction(gameState), Directions.STOP]

            if agentIndex % gameState.getNumAgents() == 0:
                return maxValue(agentIndex % gameState.getNumAgents(), gameState, depth, alpha, beta)
            else:
                return minValue(agentIndex % gameState.getNumAgents(), gameState, depth, alpha, beta)


        def maxValue(agentIndex, gameState, depth, alpha, beta):
            value = float('-inf')
            action = Directions.STOP

            #game ends if pacman is winning or losing
            if gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState), Directions.STOP]

            #find the next state and value for every action
            for legalAction in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, legalAction)
                nextValue = miniMax(agentIndex + 1, nextState, depth + 1, alpha, beta)[0]
                if nextValue > value:
                    value, action = nextValue, legalAction

                if value > beta:
                    return [value, action]
                alpha = max(alpha, value)

            return [value, action]

        def minValue(agentIndex, gameState, depth, alpha, beta):
            value = float('inf')
            action = Directions.STOP

            #game ends if pacman is winning or losing
            if gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState), Directions.STOP]

            #iterate through every action then getting the next value and state
            for legalAction in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, legalAction)
                nextValue = miniMax(agentIndex + 1, nextState, depth, alpha, beta)[0]

                if nextValue < value:
                    value, action = nextValue, legalAction

                if value < alpha :
                    return [value, action]
                beta = min(beta, value)

            return [value, action]

        alpha, beta = float('-inf'), float('inf')
        return miniMax(self.index, gameState, 0, alpha, beta)[1]

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
        def expectiMax(agentIndex, gameState, depth):
            if depth == self.depth and agentIndex % gameState.getNumAgents() == 0:
                return [self.evaluationFunction(gameState), Directions.STOP]

            if agentIndex % gameState.getNumAgents() == 0:
                return maxValue(agentIndex % gameState.getNumAgents(), gameState, depth)

            else:
                return expValue(agentIndex % gameState.getNumAgents(), gameState, depth)


        def maxValue(agentIndex, gameState, depth):
            value = float('-inf')
            action = Directions.STOP

            #game ends if pacman is winning or losing
            if gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState), Directions.STOP]

            #iterate through every action then getting the next value and state
            for legalAction in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, legalAction)
                nextValue = expectiMax(agentIndex + 1, nextState, depth + 1)[0]
                if nextValue > value:
                    value, action = nextValue, legalAction

            return [value, action]

        def expValue(agentIndex, gameState, depth):
            value = 0
            action = Directions.STOP
            legalActions = gameState.getLegalActions(agentIndex)

            #game ends if pacman is winning or losing
            if gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState), Directions.STOP]

            #iterate through every actions and taking the weighted average
            for legalAction in legalActions:
                nextState = gameState.generateSuccessor(agentIndex, legalAction)
                nextValue = (1 / len(legalActions)) * expectiMax(agentIndex + 1, nextState, depth)[0]
                value += nextValue
                action = legalAction

            return [value, action]

        return expectiMax(self.index, gameState, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    1. Add a higher score for food that has a lesser distance.
    2. Subtract the score for close ghosts, subtract more score for ghost distance <= 2.
    3. If scared ghost is closer than ghost, then increase the score.
    4. Subtract score if there is a lot of food pellet left.
    5. Subtract score if there is a lot of capsule left.
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsule = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    score = currentGameState.getScore()

    # Location of closest ghost
    distanceToGhost = float('inf')
    distanceToScaredGhost = float('inf')
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        distanceToGhost = min(distanceToGhost, manhattanDistance(pos, ghostPos)) + 1
        if ghost.scaredTimer > 0:
            distanceToScaredGhost = min(distanceToGhost, distanceToScaredGhost)

    # Location of the closest food pellet
    foodList = food.asList()
    distanceToFood = float('inf')
    for food in foodList:
        distanceToFood = min(distanceToFood, manhattanDistance(pos, food)) + 1

    # Number of food left
    numFood = currentGameState.getNumFood()

    # Number of capsule (power pellet)
    numCapsule = len(capsule)

    # Eat ghost if possible
    if distanceToScaredGhost < distanceToGhost:
        return 10 * score + (1 / distanceToFood) + 200 * (1 / distanceToScaredGhost) - 5 * numFood

    # If ghost gets too near, increase weight of ghost location distance
    if distanceToGhost <= 2:
        return score + (1 / distanceToFood) - 100 * (1 / distanceToGhost) - numFood - numCapsule

    return 100 * score - (1 / distanceToGhost) + 100 * (1 / distanceToFood) - 100 * numFood - 100 * numCapsule

# Abbreviation
better = betterEvaluationFunction
