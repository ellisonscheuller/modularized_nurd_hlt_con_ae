## Learning Representation

The goal is to create representations of physics events which are more sensitive to New Physics than those currently used.  

Traditional supervised learning requires labelled data and is limited in generalisation to unseen physics. Our objective is to learn a **semantically meaningful representation** of each event without explicit labels, so that:

- Signal/background separation becomes linearly separable.
- Anomalies (e.g., exotic decays) can be detected via unsupervised scoring.

---

## Method: Masked Particle Modelling (JEPA-like)

We use a strategy based on **masked-particle prediction**, where:

- A subset of PUPPI candidates is masked.
- The model is trained to predict the representation of the masked candidates based on the context.

This forces the model to learn event-level structure and particle correlations in the latent space.

However, because particle features are drawn from complex distributions that are difficult to model, the prediction takes place in latent space.

### Architecture Diagram

![JEPA Framework Diagram](https://images.prismic.io/encord/64b9cc18-dcb9-4fd1-95c4-c5f34f4f0877_image8.png?auto=compress,format)

*Just instead of pixels, we have particles.*

### Losses Used

1. **Latent Prediction Loss** (JEPA style):  
   Predict masked latent vectors directly rather than original features.  
   See: [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)

2. **DeepCluster-style Clustering Loss** ([Caron et al., 2018](https://arxiv.org/abs/1807.05520)):  
   Encourages compact, separable cluster structures in latent space. Target clusters are formed from particle feature clusters to provide weak supervision.

---

## Reinforcement Learning Fine-Tuning (WIP)

During fine-tuning, I experiment with **reinforcement learning (RL)**, where an agent learns:

- Which particles to mask to maximise separation.
- Reward is based on linear probe classification accuracy in latent space.

This leads to **adaptive masking policies** that exploit physics-specific cues to optimise downstream performance.

---
To run training on a Kubernetes cluster, use the manifest:
```kubeflow/jepa_rl_pipeline.yaml```

this Docker container can be used: https://gitlab.cern.ch/groups/cms-phase2-repr-learning/-/container_registries
