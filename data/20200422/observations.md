# Tweaking the learning parameters

In one figure a summary of all 20200422 runs:

![](Delta_all.png)

Quite a mess, so let me break it down. All our previous simulations had learning parameters α = 0.9, γ = 0.9 and ε = 0.5. They are included in the above plot in Grey (the old [gradient runs](../20200420/observations.md), with the longest one excluded). We then ran a lot of other simulations where we tweaked the learning parameters one at a time. Each color in this graph corresponds to one of the three parameters. We will quickly define these parameters and then go through each set of results separately.

(NB: the two black lines are two new )

## Defining the parameters

For reference, here is the Q-learning formula from Wikipedia where α and γ appear and are named:

![Q-learning formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686)

From this formula we can see that α, the learning rate, basically is a measure for how much the birds will update their Q-tables: when α = 0 they will stay constant, while when α = 1 the previous values will not play any role. (Thus it can be seen as some sort of 'memory', giving a weight to past values as opposed to the direct reward and future estimates).

As can be seen, γ is a measure for the relative weight of the estimate of the future value. Inserting this term will thus not only update the Q-values based on the direct reward r, but also the future reward. Since this again involves the Q-values, it recursively includes a term of the expected reward at further timesteps ahead, each with an additional factor γ. This is reflects the theoretical goal of reinforcement learning, which is maximizing the future reward signal <img src="https://render.githubusercontent.com/render/math?math=G_t = \sum_{n = 0}^\infty \gamma^{n} r_{t%2Bn}">. Thus γ can be interpreted as a term that weighs in long-term behaviour, and the higher gamma is, the more terms in this sum become significant.

In short we thus might say: α weighs in the past, and γ weighs in the future.

Finally, ε is the parameter controlling the exploration of the birds using an ε-greedy policy. While Q-learning aims towards a deterministic policy (such that in any given state the agent it is in, it chooses the action that yields the maximum value of its Q-table), but it can only do so by exploring different possibilities. ε defines the balance between exploration & following the optimal policy as follows: during the learning process, there is a chance of (1 - ε) that the agent will choose the action with the maximal Q-value, and a chance of ε that the agent will choose an action at random, in order to encourage the agent to discover new actions and adjust their Q-tables accordingly.

## Results for each parameter

### The learning rate α

![Alpha](Delta_alpha.png)

As the graph above shows, changing α does not seem to have a significant effect on the learning curve. Variations occur, but there is not a correlation to be seen between this variance and α (note for example the two different runs with α = 0.5 on both ends of this variance). This variance is probably some random fluctuation as a consequence of the randomized initialization of the birds.

It does seem to be common in literature to keep α low, which makes sense, since it will make the Q-values more stable. This is contrary to my previous (uneducated) guess of α = 0.9, so I will shift to α = 0.1 from now on.

NB: That there is not a clear effect of α might be a consequence of the maximum reward signal being relatively large compared to the initial Q-values. As the above equation shows, this will still cause the reward term to dominate compared to the initial Q-value, even when α is low. Since I suspect there are not any complicated long term strategy that has to be learned (as the results for γ below might also indicate), these first updates of the Q-values might already be 'the right ones', meaning that any future updates do not have an impact on the Q-values, and the specific value of α is again irrelevant.

I don't know whether this means that I should lower the maximum reward signal (or increase the initial Q-values). Might not be the case, since the birds do learn as I want them to. But I think I still should investigate that.

### The discount factor γ

![Gamma](Delta_gamma.png)

Here are the results for γ. Again, no significant correlation between the learning curve and the value of γ. What does strike me, is that they all start quite a bit steeper than the reference runs. To check what is going on here I redid some reference runs (e.g. simulations with γ = 0.9) and these also turned out to be steeper than the old references (see below). I'm still not sure why this happens, all other parameters are equal. Perhaps something I changed to the code along the way is causing this. But ultimately this is good news, and it might not really be worthwile investigating this difference, since after all the difference is not very significant, since we're only graphing a little snippet of the whole y-axis, zooming out this difference is not so big anymore. Furthermore, after a while the runs do cross each other, so the effect is too tiny to care about in my opinion.

![Gamma with new references](Delta_gamma_new_refs.png)

The fact that no significant correlation between γ and the learning curve is observed, might be a confirmation of my intuition that no complicated long-term behaviour strategies might exist in this problem (that is, strategies that take a relatively large amount of timesteps to anticipate upon). For this is ultimately what γ represents: the higher gamma is, the more terms representing future timesteps become significant.

But since I decided above to change α to 0.1, it might be a good idea to do another iteration with tweaking γ, where I now take α = 0.1 and γ = 0.9 as a reference, instead of α = γ = 0.9, which was the starting point of the current exploration in the parameter space.

### The exploration paramater ε

![Delta](Delta_epsilon.png)

In the last parameter ε we can finally see a definite correlation between its value and the steepness of the learning curve: higher ε means better learning. This makes a lot of sense, since a higher ε means more exploration to different states, so more Q-values in the table are visited and adjusted. This is also another confirmation of my belief that Q-learning in this multi-agent setting will converge to the desired policy. Note also the big jump in the learning curve that happens somewhere between ε = 0.15 and ε = 0.20.
