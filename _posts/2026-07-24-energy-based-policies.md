---
layout:                 post
title:                  "Energy Based Policies"
subtitle:               >
  Using energy-based models to represent policies in control and the benefit compared to other commonly used models.
noindex:                false
---

<p class="post-subtitle">{{ page.subtitle }}</p>

<div class="post-figure" markdown="1">
![Figure 1: Visualization of different types of policies.](/assets/posts/ebp_blog_assets/ebp_teaser.png)

**Figure 1**: *Visualization of different types of policies. [(source)](https://arxiv.org/abs/2303.04137)*
</div>

## Tasks

### Moons Toy Dataset

<div class="post-figure" markdown="1">
![Figure 2: Test set results for the moons toy example task.](/assets/posts/ebp_blog_assets/make_moons.png)

**Figure 2**: *Test set action predictions for the moons toy example task. MSE averages both action modes and gives incorrect predictions while energy-based models correctly find multimodal structure in the data.*
</div>

<div class="post-figure" markdown="1">
<img src="/assets/posts/ebp_blog_assets/make_moons_energy_slice.gif" width="600" height="245"/>

**Figure 3**: *Test set action predictions for the moons toy example task. MSE averages both action modes and gives incorrect predictions while energy-based models correctly find multimodal structure in the data.*
</div>

<div class="post-figure" markdown="1">
<img src="/assets/posts/ebp_blog_assets/make_moons_rnce_energy_slice.gif" width="600" height="245"/>

**Figure 4**: *Test set action predictions for the moons toy example task. MSE averages both action modes and gives incorrect predictions while energy-based models correctly find multimodal structure in the data.*
</div>

### Coordinate Regression

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

### Push T


Score is the mean episode score over 20 random initial conditions x 32 rollouts, where each episode's score is the maximum over time of `s = min(coverage / 0.95, 1)` (coverage = block-goal intersection area / block area).

<div style="display: flex; justify-content: center;" markdown="1">

|       | MSE | IBC | R-NCE |
|-------|-----|-----|-------|
| Score | $$0.306 \pm 0.340$$ | $$0.459 \pm 0.351$$ | $$\mathbf{0.787 \pm 0.246}$$ |

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

Code for this blog is available here: [github.com/mht3/ebp](https://github.com/mht3/ebp/)


## References

1. Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. "Implicit Behavioral Cloning." arXiv:2109.00137, 2021. [[arXiv]](https://arxiv.org/abs/2109.00137)
2. Sumeet Singh, Stephen Tu, and Vikas Sindhwani. "Revisiting Energy Based Models as Policies: Ranking Noise Contrastive Estimation and Interpolating Energy Models." arXiv:2309.05803, 2023. [[arXiv]](https://arxiv.org/abs/2309.05803)
3. Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." arXiv:2303.04137, 2024. [[arXiv]](https://arxiv.org/abs/2303.04137)
4. Kevin Zakka. "A PyTorch Implementation of Implicit Behavioral Cloning." Version 0.0.1, 2021. [[GitHub]](https://github.com/kevinzakka/ibc)
