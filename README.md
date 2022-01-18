# TicketToRideAI
Attempting to create a human-level AI which learns to complete routes and destinations in the board game called [Ticket to Ride](https://www.daysofwonder.com/tickettoride/en/usa/) using the newest reinforcement methods.
# What's Interesting about Ticket To Ride?
- There is hidden information (the cards and goals of each player)
- There is the potential for collaboration between agents
- Agents can try to sabotage their opponents instead of working toward their own goals
- Human theory suggests multiple viable winning strategies
- A general algorithm could be used to solve other board games faster than previous methods (i.e. [NEAT algorithm MIT 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf))


# Currently Implemented in this repo
### Single Actor Deep Q-Learning Algorithm 
#### (Check the DeepMonteCarlo branch if not in master)
  1. Use two neural networks, one target net and one working net, initialized to the same weights
  2. Step through the environment and record the initial state of the environment, action taken, reward, and new state in a replay buffer
  3. Train the working net using a batch of memories from the replay buffer
  4. Do back propogation with Adam optimizer and Mean Squared Error loss function
  5. Periodically update the target net with the weights of the working net
  
  #### Analysis
  Through testing, this algorithm is great at solving OpenAI's CartPole environment, but does not show promising results on the game Ticket to Ride.
  Deep Q Learning has also been shown to diverge on DouDizhu, as described in the paper [DouZero 2021](https://arxiv.org/abs/2106.06135). 
  Because Ticket to Ride is a multiplayer game, it is possible that actions from the other player creates issues converging to equilibrium.
  
---

### Monte Carlo CounterFactual Regret Minimization (MCCFR) 
#### (in the Real-MCCFR branch)
  1. Create an abstracted version of the game
  2. Play through the game tree, training one player at a time. If there is a random event, like drawing cards, only sample one of the possibilities
  3. If there are regrets for a particular game node, select actions with the highest regret, otherwisew play a random valid move
  4. Play through every action that the agent could have made for every point in the tree for a game
  5. Regret = maximum hypothetical reward - actual reward per action
  6. Update regrets for each game node after checking all the branches
  7. After training the abtracted version of the game, fit this strategy into the real game and compute real-time sub-game solving

  #### Analysis
  This approach has worked for researchers at Carnegie Mellon + Facebook when training their agent [Pluribus](https://www.cmu.edu/news/stories/archives/2019/july/cmu-facebook-ai-beats-poker-pros.html). It might work for Ticket to Ride, but it might not. Before even getting to the real-time sub-game solving, this algorithm takes a lot of work to compute. Personally, I spent multiple weeks of training time without any meaningful results.
  It uses a lot of hard drive space and trains very slowly for games with high action spaces and a large number of turns. 
  Even abstracting the action space to a few actions still lead to around 2*10^10 game nodes (each game node ~= 2kB)

## Working on it
### Deep Monte Carlo Algorithm
  - Used to solve the game DouDizhu (see [DouZero 2021](https://arxiv.org/abs/2106.06135))
  1. Use a stack of 5 previous moves in Long Short Term Memory (LSTM) which is a form of Recurrent Neural Network (RNN)
  2. Output from the RNN feeds into 6 dense layers connected by rectified linear transformations
  3. The pair of networks is used to estimate the value of a game state
  4. Using the number of valid moves as a batch dimension, for each valid move, estimate the expected value of the game state
  5. Train the network with the Adam optimizer and Mean Squared Error loss function
