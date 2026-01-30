# Recipe: Entropy Mechanism

Last updated: 06/27/2025.


<div align="center">

  The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning.



<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#🎉news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#✨getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#📖introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a>
  </p>
  <p>
    <a href="#🎈citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a> •
    <a href="#🌻acknowledgement" style="text-decoration: none; font-weight: bold;">🌻 Acknowledgement</a> •
    <a href="#📬Contact" style="text-decoration: none; font-weight: bold;">📬 Contact</a> •
    <a href="#📈star-history" style="text-decoration: none; font-weight: bold;">📈 Star History</a>
  </p>
</div>

</div>


## 🎉News




## ✨Getting started


```
cd verl
conda activate your_env
bash recipe/dapo/7b_kl_cov.sh
```


```
cd verl
conda activate your_env
bash recipe/dapo/32b_kl_cov.sh
```

## 📖Introduction

<div align="left">
</div>

This paper addresses the entropy collapse issue in scaling reinforcement learning (RL) for large language models (LLMs), where policy entropy drops sharply during training, leading to overconfidence and performance saturation. We empirically establish a relationship between entropy ($H$) and performance ($R$): $R=−aexp(H)+b$, showing performance is bottlenecked by entropy exhaustion. 

<div align="left">
</div>


## 📃Evaluation

<div align="left">
</div>


| ----------------- | ---------: | ---------: | -------: | -----------: | ------------: | ----------------: | ----------: | -------: |
| GRPO              |       21.2 |        9.6 |     58.7 |         78.8 |          27.9 |              40.7 |        36.7 |     38.6 |
| w. Clip-higher    |       18.1 |       11.5 |     56.6 |         79.2 |          29.8 |              43.3 |        40.4 |     38.8 |
| w. **`CLIP-Cov`** |       22.1 |   **15.8** |     58.2 |         80.4 |      **30.5** |          **44.1** |    **41.1** |     40.4 |
| w. **`KL-Cov`**   |   **22.6** |       12.9 | **61.4** |     **80.8** |          29.1 |              42.6 |        38.2 | **40.6** |
| GRPO              |       21.8 |       16.2 |     69.7 |         84.2 |          35.2 |              43.6 |        45.5 |     45.8 |
| w. Clip-higher    |       35.6 |       22.3 |     69.5 |         77.2 |          35.1 |              42.5 |        43.0 |     47.2 |
| w. **`CLIP-Cov`** |       32.3 |       22.7 |     67.2 |     **87.0** |      **42.0** |          **57.2** |        46.0 |     50.3 |
| w. **`KL-Cov`**   |   **36.8** |   **30.8** | **74.5** |         84.6 |          39.1 |              49.0 |    **46.3** | **52.2** |



## 🎈Citation
If you find this paper or repo helpful, please cite us.

```bibtex
  title={The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models},
  author={Cui, Ganqu and Zhang, Yuchen and Chen, Jiacheng and Yuan, Lifan and Wang, Zhi and Zuo, Yuxin and Li, Haozhan and Fan, Yuchen and Chen, Huayu and Chen, Weize and others},
  year={2025}
}
```
## 🌻Acknowledgement

## 📬 Contact

For questions, discussion, or collaboration opportunities, feel free to contact:

