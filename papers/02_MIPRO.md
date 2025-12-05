# [Self-Evolving-Agents](https://arxiv.org/pdf/2507.21046)

## [MIPRO](https://dspy.ai/) 
Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs. 
The authors seek to efficiently optimize prompts in arbitrary LM programs, especially those with multiple stages.

### **What is Language Model Program?**

A Language Model Program is software that separates Control Flow (Python code) from Model Parameters (Prompts & Demonstrations). 
It treats LLM calls as modular, trainable functions rather than static text strings.

#### 1. THE PROGRAM (Logic)
```
# defined by YOU. It is deterministic.
class RAG_Program(dspy.Module):
    def __init__(self):
        # "Layers": The specific LLM calls (optimizable parameters)
        self.search = dspy.ChainOfThought("question -> context")
        self.answer = dspy.Predict("context, question -> response")

    def forward(self, question):
        # "Control Flow": Loops, Logic, and Data Handling
        context = self.search(question)
        
        # You can add logic an LLM can't handle (e.g., hard limits)
        if len(context) > 1000:
            context = context[:1000]
            
        return self.answer(context=context, question=question)
```
#### 2. THE OPTIMIZER (MIPRO: Multi-prompt Instruction PRoposal Optimizer)
```
# The "Compiler" that tunes the prompts automatically.
# It treats instructions like weights in a neural network.
optimized_program = MIPRO(
    program=RAG_Program(),
    metric=grading_function
).compile()
```

**Key Difference**

Prompting: You edit the string "You are a helpful assistant..." manually.

LM Program: You write the code above, and MIPRO mathematically finds the best "You are a helpful assistant..." string for you.

### How can we optimize?
Maximize the downstream evaluation metric!

### Problem formalization
![Task](figures/Optimization_Problem.png)
```
Algorithm 1 Optimize Φ with optimizer M
Input: Optimizer M, Initial Program Φ, Metric µ
Input: Max Iterations I, Training Data D
Input: Minibatch size B, Proposer Hyperparameters θ
Output: Optimized version of Φ

M.Initialize(D, θ) ▷ Initialize optimizer using the data
for k ← 1 to I do
    (V → S_k) ←M.Propose(θ) ▷Generate proposal (need to minimize)
    D_k ←{(x_j,x′_j) ∼ D}^B_j=1 ▷ Sample size-B batch
    σ ← 1/B Sum_{(x,x′)∈D_k} µ(Φ_{V→S_k}(x),x′) ▷ Validate updated program (need to minimize)
    M.Update(V → S_k, σ) ▷Update optimizer based on the observed validation score
end for
(V → S_k) ←M.ExtractOptimizedSets()
return Φ_{V→S}
```
Consider an LM program Φ consisting of m modules, each using some LM.
Each module i is defined by a prompt template p_i that contains a set of variables (open slots) v.
Let V be the set of all variables used by prompt templates for Φ, and let V → S be a total assignment of variables V to strings S.
Φ_{V→S} is the specified program.
The high-level goal is to find a total assignment V → S that optimizes Φ with respect to metric µ.

(i) each string s ∈ S can take on any value

(ii) the metric µ provides supervision only at the level of the entire task, so every variable in V is latent

(iii) assume no access to the gradients or embeddings of the LMs involved, which rules out many RL and prompttuning algorithms

(iv) system designers generally have small datasets D

(v) small budgets of LM calls for evaluating Φ

### Designing LM Program Optimizers

#### Bootstrap Random Search
![Task](figures/Bootstrap_Random_Search.png)

In Step 1, demonstrations are bootstrapped by running training inputs through the program Φ and keeping traces that produce sufficiently high scoring outputs, as judged by metric µ. 

In Step 2, these bootstrapped demonstration sets are searched over using random search, and the most performant set is returned.

> **Limitations**
> Inability to Tune Instructions: focuses solely on selecting demonstrations and cannot optimize or tune the free-form instructions for multi-prompt pipelines.

#### The Module-Level [OPRO](https://arxiv.org/abs/2309.03409) optimizer
![Task](figures/Module_Level_OPRO.png)
A history of module-level instructions and program score pairs are given as  input to the proposer LM to generate a new instruction for each module.
These are then evaluated in the program, and the resulting score is added back with each module’s instruction to the module’s history.
The process repeats for I iterations.

> **Limitations**
> Credit Assignment Assumptions: It relies on the strong assumption that the overall program score is a sufficient proxy for an individual instruction's quality, effectively assuming equal credit assignment across modules.
> Independence Assumption: It assumes there is no inter-assignment dependency between different modules, optimizing them as if they are independent.

#### Multi-prompt Instruction PRoposal Optimizer
![Task](figures/MIPRO.png)

In Step 1, demonstrations are bootstrapped using the same process from Step 1 of Bootstrap Random Search. 

In Step 2, instructions are proposed using the grounding strategy described in [The Proposal Problem](). 

In Step 3, Bayesian optimization is used to find the best-performing combination of instruction and demonstration candidates.


**1. The Core Problem: Credit Assignment**

In multi-stage pipelines (e.g., `Retrieve -> Summarize -> Answer`), manual prompting fails because it is difficult to determine which stage caused an error.
* *Did the retrieval fail?*
* *Did the reasoning fail?*
* *Did the formatting fail?*

MIPRO solves this by optimizing the **entire pipeline simultaneously** rather than optimizing modules in isolation.

---

**2. The Algorithm: Two-Stage Process**

MIPRO operates similarly to **AutoML** (Automated Machine Learning), but for prompts.

**Phase 1: Proposal (Exploration)**

MIPRO first generates a "search space" of potential candidates.

1.  **Bootstrapping:** * It runs the program on a small training set.
    * It collects "traces" (input/output pairs) of successful runs to use as potential few-shot examples.
2.  **Instruction Generation (The Proposer):**
    * It looks at the code and the raw data.
    * It uses a powerful "Meta-LLM" to generate $N$ distinct instruction variations for *every* module (e.g., "Be concise", "Think step-by-step", "Act as an expert").
    * **Result:** A generated menu of options (Instructions + Example Sets) for every step in the pipeline.

**Phase 2: Optimization (Bayesian Search)**

MIPRO then searches for the best combination of options.

1.  **Combinatorial Explosion:** * If Module A has 10 options and Module B has 10 options, there are 100 possible programs.
2.  **Bayesian Surrogate Model (TPE):**
    * Instead of running every combination (expensive), it uses a **Tree-structured Parzen Estimator (TPE)**.
    * It samples random combinations initially.
    * It learns which instructions correlate with higher scores.
    * It effectively "backpropagates" the final metric to select the specific instruction/example pair for each module that maximizes the *total* system performance.


Joint Optimization (Instruction and Demonstration)

Bayesian Optimization (Surrogate Model: Tree-structured Parzen Estimator, robustnessto noise)

Grounding (Dataset Summary)

> **Limitations**
> Dependency on Seed Prompts: Like the other optimizers, it has a restricted ability to infer the rules governing complex tasks without a handwritten seed prompt.
> Reward Hacking:
> Black-box Prompts:
> Search Space Explosion: 
