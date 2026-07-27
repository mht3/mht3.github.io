---
layout:                 post
title:                  "Energy Based Policies"
subtitle:               >
  Using energy-based models to represent policies in control and the benefit compared to other commonly used models.
noindex:                false
---

<p class="post-subtitle">{{ page.subtitle }}</p>

<div class="post-figure" markdown="1">
![Figure 1: Visualization of different types of policies.](/assets/posts/ebp_teaser.png)

**Figure 1**: *Visualization of different types of policies. [(source)](https://arxiv.org/abs/2303.04137)*
</div>

## Tasks

### Moons Toy Dataset

| MSE | IBC | R-NCE |
|-----|-----|-----|
|<img src="/assets/posts/ebc_blog_assets/mse_make_moons_predictions.png" width="250" height="250"/>|<img src="/assets/posts/ebc_blog_assets/ibc_make_moons_predictions.png" width="250" height="250"/>|<img src="/assets/posts/ebc_blog_assets/rnce_make_moons_predictions.png" width="250" height="250"/>|

<div class="post-figure" markdown="1">
<img src="/assets/posts/ebc_blog_assets/make_moons_energy_slice.gif" width="600" height="245"/>
</div>

<div class="post-figure" markdown="1">
<img src="/assets/posts/ebc_blog_assets/make_moons_rnce_energy_slice.gif" width="600" height="245"/>
</div>

### Coordinate Regression

### Push T

$$
\pi(a \mid s) \propto \exp(-E_\theta(s, a))
$$


## Code

Code for this blog is available here: [github.com/mht3/ebp](https://github.com/mht3/ebp/)


## References

1. Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. "Implicit Behavioral Cloning." arXiv:2109.00137, 2021. [[arXiv]](https://arxiv.org/abs/2109.00137)
2. Sumeet Singh, Stephen Tu, and Vikas Sindhwani. "Revisiting Energy Based Models as Policies: Ranking Noise Contrastive Estimation and Interpolating Energy Models." arXiv:2309.05803, 2023. [[arXiv]](https://arxiv.org/abs/2309.05803)
3. Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." arXiv:2303.04137, 2024. [[arXiv]](https://arxiv.org/abs/2303.04137)
4. Kevin Zakka. "A PyTorch Implementation of Implicit Behavioral Cloning." Version 0.0.1, 2021. [[GitHub]](https://github.com/kevinzakka/ibc)
