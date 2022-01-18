# TicketToRideAI
Attempting to create a human-level AI which learns to complete routes and destinations in the board game called [Ticket to Ride](https://www.daysofwonder.com/tickettoride/en/usa/) using the newest reinforcement methods.
## Currently Implemented in this repo
* Single Actor Deep Q-Learning Algorithm (Check the DeepMonteCarlo branch if not in master)
  - Use two neural networks, one target net and one working net, initialized to the same weights
  - Step through the environment and record the initial state of the environment, action taken, reward, and new state in a replay buffer
  - Train the working net using a batch of memories from the replay buffer
  - Do back propogation with Adam optimizer and Mean Squared Error loss function
  - Periodically update the target net with the weights of the working net
  
  #### Analysis
  This algorithm is great at solving OpenAI's CartPole environment. Unfortunately, it's not quite as good at solving Ticket To Ride.
  Deep Q Learning has been shown to diverge on DouDizhu, as described in the paper [DouZero 2021](https://arxiv.org/abs/2106.06135)

## Working on it
* Deep Monte Carlo Algorithm
  - Used to solve the game DouDizhu 
  - Use a stack of 5 previous moves in Long Short Term Memory (LSTM) which is a form of Recurrent Neural Network (RNN)
  - Output from the RNN feeds into 6 dense layers connected by rectified linear transformations
  - The pair of networks is used to estimate the value of a game state
  - Using the number of valid moves as a batch dimension, for each valid move, estimate the expected value of the game state
  - Train the network with the Adam optimizer and Mean Squared Error loss function
