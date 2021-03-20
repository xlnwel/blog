---
title: "Deep Reinforcement Learning and its Neuroscientific Implications"
excerpt: "Notes from Deep Reinforcement Learning and Its Neuroscientific Implications"
categories:
  - Reinforcement Learning
tags:
  - Overviews
---

## Introduction

We cover a few notes from [Botvinick et al. 2020](#ref1).

## Vanguard Studies

[Wang et al. 2018](#ref2) describe a meta-reinforcement learning effect: When trained on a series of interrelated tasks—for example, a series of decision tasks with the same overall structure but different reward probabilities—recurrent deep RL develop the ability to adapt to new tasks of the same kind without weight changes. Furthermore, structured representations emerge in the RNN activations during the adaptation phase, showing that RL can be supported, in some cases, by activity-based working memory. 

Early work shows dopamine transmit a signal corresponding to reward-prediction error(RPE) in RL. [Dabney et al. 2020](#ref3) studied electrophysiological data from mice to test whether the dopamine system might employ the kind of vector code involved in distributional RL. They find this dopaminergic signal is distributional, conveying a spectrum of RPE from pessimistic to optimistic.

## Topics for Future Research

### Representation Learning

In deep RL, reward-based learning shape the internal representation, and these representations in turn support reward-based decision making. However, there are two potential drawbacks of representations shaped by RL alone. One problem is that rewards are generally sparse, which only provides weak signals for learning and makes credit assignment problem hard to deal with. The second problem is overfitting—internal representations shaped exclusively by task-specific reward may end up being useful only for tasks that the learner performed, but completely wrong for new task. Current DRL often turns to auxiliary tasks, which shape the representation in a way that is not exclusively to the specific tasks confronted by the learner.

One further issue foregrounded in deep RL involves the role of inductive biases in shaping representation learning. Most RL systems that takes visual input employ a CNN that biases them towards representations that take into account the translation invariance of images. More recently developed architectures build in a bias to visual inputs as comprising sets of discrete objects with recurring pairwise relationship([Watters et al. 2019](#ref5))

#### Future reading

[Richards et al. 2019](#ref4) shows that focusing on the three components of artificial neural networks—the objective functions, the learning rules and the architectures—also benefits neuroscience.

### Model-Based RL

Despite the success of model-based RL in many challenging tasks, such as board games and Atari, a key open question is whether systems can learn to capture transition dynamics at a high-level of abstraction ("if I throw a rock at that window, it will shatter") rather than being tied to detailed prediction about perceptual observations (predicting whether each shard would fall) ([Konidaris 2019](#ref6], [Behrens et al. 2018](#ref7)).

One particularly intriguing finding from deep RL is that there are circumstances under which processes resembling model-based RL may emerge spontaneously within systems trained using model-free RL algorithms. Intriguingly, model-based behavior is also seen in RL systems that employ a particular form of predictive code, referred to as the 'successor representation'([Momennejad 2020](#ref8)), suggesting one possible mechanism through which model-free planning might arise

An interesting question that has arisen in neuroscientiﬁc work is how the balance between model-free and model-based RL is arbitrated, that is, what are the mechanisms that decide, moment to moment, whether behavior is controlled by model-free or model-based mechanisms.

### Memory

In deep RL, three kinds of mechanisms are used to maintain and retrieve memory for decision making. First, episodic memory systems read and write storage-slots. Second,  recurrent networks store working memory in activations. Last, attention and relational systems combine and coordinate working and episodic memory using attention.

## References

<a name="ref1"></a>Botvinick, Matthew, Jane X. Wang, Will Dabney, Kevin J. Miller, and Zeb Kurth-Nelson. 2020. “Deep Reinforcement Learning and Its Neuroscientific Implications,” 603–16. https://doi.org/10.1016/j.neuron.2020.06.014.

<a name="ref2"></a>Wang, Jane X., Zeb Kurth-Nelson, Dharshan Kumaran, Dhruva Tirumala, Hubert Soyer, Joel Z. Leibo, Demis Hassabis, and Matthew Botvinick. 2018. “Prefrontal Cortex as a Meta-Reinforcement Learning System.” *Nature Neuroscience* 21 (6): 860–68. https://doi.org/10.1038/s41593-018-0147-8.

<a name="ref3"></a>Dabney, Will, Zeb Kurth-Nelson, Naoshige Uchida, Clara Kwon Starkweather, Demis Hassabis, Rémi Munos, and Matthew Botvinick. 2020. “A Distributional Code for Value in Dopamine-Based Reinforcement Learning.” *Nature* 577 (7792): 671–75. https://doi.org/10.1038/s41586-019-1924-6.

<a name="ref4"></a>Richards, Blake A., Timothy P. Lillicrap, Philippe Beaudoin, Yoshua Bengio, Rafal Bogacz, Amelia Christensen, Claudia Clopath, et al. 2019. “A Deep Learning Framework for Neuroscience.” *Nature Neuroscience*. Nature Publishing Group. https://doi.org/10.1038/s41593-019-0520-2.

<a name="ref5"></a>Watters, Nicholas, Loic Matthey, Matko Bošnjak, Christopher P. Burgess, and Alexander Lerchner. 2019. “COBRA: Data-Efficient Model-Based RL through Unsupervised Object Discovery and Curiosity-Driven Exploration.” *ArXiv*. arXiv. https://github.com/deepmind/spriteworld.

<a name="ref6"></a>Konidaris, George. 2019. “On the Necessity of Abstraction.” *Current Opinion in Behavioral Sciences*. Elsevier Ltd. https://doi.org/10.1016/j.cobeha.2018.11.005.

<a name="ref7"></a>Behrens, Timothy E.J., Timothy H. Muller, James C.R. Whittington, Shirley Mark, Alon B. Baram, Kimberly L. Stachenfeld, and Zeb Kurth-Nelson. 2018. “What Is a Cognitive Map? Organizing Knowledge for Flexible Behavior.” *Neuron*. Cell Press. https://doi.org/10.1016/j.neuron.2018.10.002.

<a name="ref8"></a>Momennejad, Ida. 2020. “Learning Structures: Predictive Representations, Replay, and Generalization.” *Current Opinion in Behavioral Sciences*. Elsevier Ltd. https://doi.org/10.1016/j.cobeha.2020.02.017.