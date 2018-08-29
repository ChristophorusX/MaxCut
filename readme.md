# MaxCut: an implementation of belief propagation and extremal optimization on regular graphs

Belief propagation is a message-passing algorithm for performing inference on graphical models, such as Bayesian networks and Markov random fields. It calculates the marginal distribution for each unobserved node, conditional on any observed nodes. has demonstrated empirical success in numerous applications including free energy approximation. We tried to utilize BP to tackle the task of clustering in stochastic block model with random initialization of the partial density functions on each vertex, but with no luck.

Extremal optimization is a method for approximating solutions to hard optimization problems, e.g., determination of ground-state configurations in disordered materials. Many of them concern the problems where the relation between individual components of the system is frustrated, and thereby the corresponding cost functions often exhibit some complex “landscape” in configuration space. Further, for growing system size the cost function may have an exponentially increasing number of local extrema separated by sizable barriers, which make the search for the exact optimal solution very costly. However, the method called *extremal optimization* (EO), which is based on the dynamics of non-equilibrium processes that exhibit self-organized criticality, where better solutions emerge dynamically, often has surprisingly good performance on some of the hardest optimization problems.

For further details, see [this link](https://cims.nyu.edu/~rx262/projects.html).
