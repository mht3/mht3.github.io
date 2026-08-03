---
layout:                 post
title:                  "Energy Based Policies"
subtitle:               >
  Using energy-based models to represent policies in control and the benefit compared to other commonly used models.
noindex:                false
---

<p class="post-subtitle">{{ page.subtitle }}</p>

**Under Construction**. Don't peek!

I've started to incorporate energy-based models into my own research for learning robot control policies and wanted to share what I've learned so far. This blog is intended for other researchers or anyone interested in learning about learning-based policies for robots. I won't go into depth on model architectures, or exact ways of training. Feel free to check my code for that. I do hope you gain a high level understanding of the kinds of robot policies out there, and build a solid intuition for how energy based policies work and why they're useful.

Recently, there has been some excitement in industry about energy-based models, backed by researchers like Yann LeCun and companies such as Logical Intelligence with their newest [KONA reasoning model](https://logicalintelligence.com/kona-ebms-energy-based-models). Part of what's appealing is that energy-based models learn a scalar function over candidate solutions, allowing inference to be framed as an optimization problem rather than a purely autoregressive generation process like with large-language models. If we assign low energy to "good" solutions and high energy to "bad" solutions, then finding good solutions boils down to a function minimization problem. 

A robot control policy is simply a framework for taking actions given states/observations. We usually denote this policy as \\(\pi(a \mid s)\\). A robot can (and probably should) do different things depending on what it observes. Here we will focus on a purely offline imitation learning setting where we assume that optimal demonstrations are given in the form of a dataset. The goal is to "imitate" the actions we see in the data and generalize well to unseen states. 


<div class="post-figure" markdown="1">
![Figure 1: Visualization of different types of policies.](/assets/posts/ebp_blog_assets/ebp_teaser.png)

**Figure 1**: *Visualization of different types of policies [(source)](https://arxiv.org/abs/2303.04137). (a) explicit policies that learn to predict actions directly from observations. (b) Implicit/energy-based policies where optimal actions lie on areas with lowest energy. (c) Iterative generative models refining data from noise.*
</div>


As of this post, diffusion [[3]](#ref3) and score/flow matching [[5]](#ref5) models have taken the stage in the robotics community as robot control policies. These models learn noise from data and iteratively denoise to produce outputs that match the data distribution. Our focus today is not these models, but instead energy based models, or EBMs (middle of Figure 1). EBMs in continuous, high-dimensional spaces have a reputation for being impractical to train, while Diffusion and flow matching policies have a reputation for being easier to learn high dimensional data such as images. The go-to objective for EBM-based behavior cloning, implicit behavior cloning (IBC) [[1]](#ref1), only reinforced this despite extremely impressive results in lower dimensions [[3]](#ref3). 


Recently, I stumbled upon a paper while reading related work for my current research project that explained how the IBC objective is ill-posed [[2]](#ref2). The authors proposed a solution and showed that energy-based models can work just as well, if not better than diffusion models in high dimensions. This is my attempt at recreating the work by [[2]](#ref2) and understanding the bare-bones implementation of their ranking noise contrastive estimation loss and learnable proposal distribution. 

Why bother with EBMs as policies?! This is a question you may have if you are a flow/score matching enthusiast. My main interest in these energy-based models is **composability**. Say we train an energy function that pulls a robot arm toward a target. If an obstacle shows up at test time that wasn't in the training data, we don't need to retrain anything! We can just add a repulsive energy term that grows large near the obstacle, and minimize the summation of both energies instead. In the real world, this could have profound impacts on safety and interpretability compared to traditional AI controllers being used. I am not here to say that flow and score matching models should not be used. We've seen real world robotics problems be solved that we never thought possible with these models! As a researcher, I think it's always good to take a step back and ask why? Why are these models so good? What knowledge can we take from them? 

## Methods

Here we will focus on behavior cloning settings, i.e. learning a policy from a dataset \\(\mathcal{D}\\) of optimal demonstrations. Along the way I will attempt to answer questions such as: How good are EBMs at modeling multimodal data and how easy is it to train EBMs in higher dimensions? 

Let's start with our simplest baseline: **Vanilla behavior cloning**. Vanilla behavior cloning seeks to minimize the mean square error between an offline dataset of optimal actions, and predicted actions from a learned model. We can define a mean-square error objective for the simple vanilla behavior cloning objective. This is extremely simple to implement and with sufficient data can learn reasonable policies.

$$\mathcal{L}_{MSE} = \frac{1}{|\mathcal{D}|}\sum_{i \in \mathcal{D}} \|\hat{a}_i - a^*_i\|_2^2 \tag{1}$$

#### Implicit Behavior Cloning

#### Ranking-Noise Contrastive Estimation

Ranking-Noise Contrastive Estimation

## Results!

#### Moons Toy Dataset

<div class="post-figure" markdown="1">
![Figure 2: Test set results for the moons toy example task.](/assets/posts/ebp_blog_assets/make_moons.png)

**Figure 2**: *Test set action predictions for the moons toy example task. MSE averages both action modes and gives incorrect predictions while energy-based models correctly find multimodal structure in the data.*
</div>

<div class="post-figure" markdown="1">
<img src="/assets/posts/ebp_blog_assets/make_moons_energy_slice.gif" width="600" height="245"/>

**Figure 3**: *Standard energy based model inference on the moons toy dataset for a held out state of s=0.4. Samples start uniformly over the action space and are iteratively updated with Langevin dynamics.*
</div>

<div class="post-figure" markdown="1">
<img src="/assets/posts/ebp_blog_assets/make_moons_rnce_energy_slice.gif" width="600" height="245"/>

**Figure 4**: *Ranking noise contrastive estimation EBM inference on the moons toy dataset for a for a held out state of s=0.4.  A Gaussian proposal distribution warm-starts the samples and actions are iteratively updated with Langevin dynamics.*
</div>

#### Coordinate Regression

<div style="display: flex; justify-content: center;" markdown="1">

|       | MSE | IBC | R-NCE |
|-------|-----|-----|-------|
| N=10 | $$0.368$$ | $$0.994$$ | $$\mathbf{1.0}$$ |
| N=30 | $$0.858$$ | $$0.986$$ | $$\mathbf{0.994}$$ |

</div>

<div class="post-figure" markdown="1">
![Figure 5: Test set results for the coordinate regression task.](/assets/posts/ebp_blog_assets/coordinate_regression.png)

**Figure 5**: *Test set results for the coordinate regression task. Top: Explicit and implicit models trained on 10 images. Bottom: models trained on 30 images. MSE overfits easily with little training data.*
</div>

#### Push-T

The final task in this blog is a true sequential control problem where an end-effector must push a T-shaped block into a target (green) position [[3]](#ref3). The goal state stays fixed and the end-effector and T-block have random starting positions. The dataset itself is directly from [[3]](#ref3), and consists of 20 state dimensions: 9 fixed points on the T-block, and the pusher's (x, y) position. Note that the original Diffusion policy paper also had a Push-T dataset with image observations. I chose to only use the keypoint-states for simplicity with an MLP, so no CNN or transformer backbones are used. The action is the 2D coordinate for where to move the end-effector to. Internally, a PD controller moves from the current position to the next position.

Score is the max target area coverage averaged over 20 random seeds and 32 rollouts, i.e. `s = min(coverage / 0.95, 1)`. On the left hand side of the table below you will see symbols $T_o$, $T_a$, and $T_p$. These are the observation history, action executions, and action predictions. The first row shows scores where we don't action chunk and predict a single action from the current and previous observation. The second, higher performing row, shows action chunking where 8 actions are executed at once, more similar to diffusion policy execution and model-predictive control settings.
<div style="display: flex; justify-content: center;" markdown="1">

|       | MSE | IBC | R-NCE |
|-------|-----|-----|-------|
| Score ($$T_o=2, $T_a=1, T_p=1$$)| $$0.306 \pm 0.026$$ | $$0.459 \pm 0.027$$ | $$\mathbf{0.787 \pm 0.019}$$ |
| Score (($$T_o=2, $T_a=8, T_p=8$$))| $$0.306 \pm 0.026$$ | $$0.459 \pm 0.027$$ | $$\mathbf{0.787 \pm 0.019}$$ |
</div>



<div class="post-figure" markdown="1">
![Figure 6: MSE, IBC, and R-NCE rollouts for a fixed initial state. Interestingly, energy-based models don't use the full multimodal action landscape.](/assets/posts/ebp_blog_assets/push_t_multimodal.png)

**Figure 6**: *MSE, IBC, and R-NCE rollouts for a fixed initial state. Interestingly, energy-based models don't use the full multimodal action landscape.*
</div>

<div class="post-figure" markdown="1">
![Figure 7: MSE, IBC, and R-NCE rollouts from four test set initial states.](/assets/posts/ebp_blog_assets/push_t.png)

**Figure 7**: *MSE, IBC, and R-NCE rollouts from four test set initial states.*
</div>


## Code

Code for this blog is available here: [github.com/mht3/ebp](https://github.com/mht3/ebp/).

If you have any questions or comments, feel free to email me at mat028 [at] ucsd [dot] edu.


## References

1. <a id="ref1"></a>Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. "Implicit Behavioral Cloning." arXiv:2109.00137, 2021. [[arXiv]](https://arxiv.org/abs/2109.00137)
2. <a id="ref2"></a>Sumeet Singh, Stephen Tu, and Vikas Sindhwani. "Revisiting Energy Based Models as Policies: Ranking Noise Contrastive Estimation and Interpolating Energy Models." arXiv:2309.05803, 2023. [[arXiv]](https://arxiv.org/abs/2309.05803)
3. <a id="ref3"></a>Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." arXiv:2303.04137, 2024. [[arXiv]](https://arxiv.org/abs/2303.04137)
4. <a id="ref4"></a>Kevin Zakka. "A PyTorch Implementation of Implicit Behavioral Cloning." Version 0.0.1, 2021. [[GitHub]](https://github.com/kevinzakka/ibc)
5. <a id="ref5"></a>Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. "Flow Matching for Generative Modeling." *arXiv:2210.02747*, 2023. [[arXiv]](https://arxiv.org/abs/2210.02747)