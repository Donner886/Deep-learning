
# Softmax


## The origin and Name Meaning (why it is called softmax)
The softmax function is a generalization of the logistic function to multiple dimensions. 
It is widely used in machine learning and deep learing and LLM fields. 
It takes a vector of real numbers and transforms it into a probability distribution over multiple classes. 
The name "softmax" comes from the fact that it "softens" the maximum function, allowing for a smooth transition between the highest score and the others, rather than making a hard decision.



## Where does it come from?

The softmax function is the Boltzmann distribution applied to a finite set of scores.

In statistical mechanics, the Boltzmann distribution （波兹曼分布） gives the probability of a system being in state $i$ with energy $E_i$:
在数学上，波兹曼分布函数（更广泛的形式为Gibbs 吉布斯分布）在统计学和机器学习中有被成为*对数-线性模型*； 在深度学习中波兹曼分布被用于随机神经网络的采样分布。

什么是对数-线性模型？对数-线性模型是指模型的输出是输入特征的线性组合的指数函数。也就是说，模型的输出是输入特征的加权和，然后通过指数函数进行转换。对数-线性模型在统计学和机器学习中被广泛应用于分类、回归和概率建模等任务。


$$
P(i) = \frac{e^{-E_i/(kT)}}{\sum_j e^{-E_j/(kT)}}
$$

where:
- $T$ is the temperature
- $k$ is Boltzmann's constant
- the denominator normalises the probabilities so they sum to $1$

In machine learning, we usually start with scores or logits $z_i$ rather than energies.  

$$
E_i \propto -z_i
$$

Substituting that into the Boltzmann form gives:

$$
P(i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

This is exactly the softmax function with temperature $T$:

$$
\mathrm{softmax}(z_i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

So the connection is simple:

- Boltzmann distribution: probabilities from energies
- Softmax: probabilities from logits
- Relationship: logits act like negative energies

## Why this works

- Exponentials 指数 make larger scores much more likely
    (Exponential: The standard exponential function is a mathematical function written in the form $f(x) = ab^x $, where the variable $x$ is in the exponent position. )   
- Normalisation ensures all outputs are between $0$ and $1$
- The probabilities sum to $1$, so the output is a valid distribution

## Role of temperature
- Small $T$: the distribution becomes sharper, favouring the largest logit
- Large $T$: the distribution becomes flatter, making probabilities more uniform
 
### Diagram: temperature controls sharpness

Using the same logits $[2,1,0]$:

```mermaid
xychart-beta
    title "Softmax probabilities vs temperature (logits [2,1,0])"
    x-axis [Class A, Class B, Class C]
    y-axis "Probability" 0 --> 1
    bar "T = 0.5 (sharp)" [0.867, 0.117, 0.016]
    bar "T = 1.0 (baseline)" [0.665, 0.245, 0.090]
    bar "T = 2.0 (flat)" [0.506, 0.307, 0.186]
```

Interpretation:
- As $T$ decreases, probability mass concentrates on the top logit (more confident/peaky).
- As $T$ increases, probabilities spread more evenly across classes (more uncertain/smoother).
 

## why Boltzmann distribution is generalized to Softmax
In softmax, the scores are just numbers that measure how good each option is or how much you prefer one option over another.  
A score can mean different things in different contexts: 
1. In classification, a score is a model's raw output for each class
2. In recommendation, a score is how relevant an item seems
3. In search ranking, a score is how well a result matches the query
4. In decision making, a score is how attractive an action is
So the word “score” here really means something like preference value, compatibility value, or logit.



## Real-life example
Imagine you are choosing lunch after work. You have three options:

- Restaurant A: closest to you and your favourite food
- Restaurant B: a bit farther away but still good
- Restaurant C: not very attractive today

You do not choose them with absolute certainty. You are more likely to go to the best option, but you may still pick another one if the difference is not huge.

Suppose you assign desirability scores:

- Restaurant A: $2$
- Restaurant B: $1$
- Restaurant C: $0$

If $T = 1$, softmax gives:

$$
\mathrm{softmax}([2,1,0]) =
\frac{[e^2,e^1,e^0]}{e^2+e^1+e^0}
\approx [0.665, 0.245, 0.090]
$$

This means:

- Restaurant A gets about $66.5\%$ chance
- Restaurant B gets about $24.5\%$ chance
- Restaurant C gets about $9\%$ chance

So the best choice is most likely, but the other choices are still possible. That is exactly the Boltzmann idea: better options are exponentially more likely, not absolutely forced.

## Another example: choosing a route home

Imagine you are driving home and you have three routes:

- Route A: fastest, but sometimes crowded
- Route B: slightly slower, but more reliable
- Route C: longest, but scenic

You usually prefer the fastest route, but if traffic looks bad you may still choose another one. That matches the Boltzmann idea well: the best option dominates, but alternatives remain available.

If you score the routes as $[3, 2, 0]$, then with $T = 1$:

$$
\mathrm{softmax}([3,2,0])
= \frac{[e^3,e^2,e^0]}{e^3+e^2+e^0}
\approx [0.705, 0.259, 0.035]
$$

So Route A is most likely, Route B still has a decent chance, and Route C is unlikely but not impossible.

## Another example: picking a video recommendation

Think about a streaming app recommending three videos:

- Video A: very relevant to your taste
- Video B: somewhat relevant
- Video C: not very relevant

The app does not need to always show only Video A. A softmax-like rule turns relevance scores into probabilities, so the best match is shown most often, but some diversity is preserved.

If the scores are $[4, 2, 1]$, then softmax makes the first video much more likely than the others. This is useful because it keeps the system from being too rigid.

In one sentence: softmax is the Boltzmann distribution rewritten for machine learning, where model scores play the role of negative energy.
