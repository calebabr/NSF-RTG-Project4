## NSF-RTG-Project4

# Goals to Complete
- Numerically verify the bound holds across many targets and many values of m
- Try to identify what breaks if you have k+1 clusters. Show numerically that such configurations are unstable, meaning small perturbations cause one cluster to dissolve
- Run the simulation to convergence, record final bias and amplitude values, plug them back into the two ODE expressions, verify they are near zero, then check that the integrated residual from each final bias location to 1 is approximately zero for every active neuron — doing this across many targets to build evidence that the number of locations satisfying this condition never exceeds k
- Build intuition for why extra clusters are unstable that could guide a future proof

# Why These Goals Matter
- Numerically verifying the bound across many targets and values of m builds a strong empirical case for the conjecture before any formal proof exists — if the bound fails even once, that is a counterexample that disproves the conjecture entirely and redirects the research
- Identifying instability in k+1 cluster configurations is important because instability is often easier to prove mathematically than convergence. Showing that overcomplete configurations always dissolve gives a concrete mechanism that a future proof could formalize
- Checking the orthogonality conditions at fixed points connects the numerical simulation directly to the mathematical structure of the problem — if the conditions hold empirically across many targets, it suggests the fixed point characterization from the ODE analysis is the right framework for an eventual proof
- Building intuition for why extra clusters are unstable is the bridge between numerical evidence and rigorous mathematics. Concrete intuition about the mechanism is what guides which proof strategy to pursue, whether that is a Lyapunov argument, a fixed point analysis, or borrowing from approximation theory


