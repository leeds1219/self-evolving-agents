# [Self-Evolving-Agents](https://arxiv.org/pdf/2507.21046)

## [MIPRO](https://dspy.ai/) 
Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs. 
The authors seek to efficiently optimize prompts in arbitrary LM programs, especially those with multiple stages.

### **What is Language Model Program?**

A Language Model Program is software that separates Control Flow (Python code) from Model Parameters (Prompts & Demonstrations). It treats LLM calls as modular, trainable functions rather than static text strings.

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
#### 2. THE OPTIMIZER (MIPRO)
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
```
Algorithm 1 Optimize Φ with optimizer M
Input: Optimizer M, Initial Program Φ, Metric µ
Input: Max Iterations I, Training Data D
Input: Minibatch size B, Proposer Hyperparameters θ
Output: Optimized version of Φ

M.Initialize(D, θ) ▷ Initialize optimizer using the data
for k ← 1toI do
    (V → Sk) ←M.Propose(θ) ▷Generate proposal
    Dk ←{(xj,x′ j) ∼ D}B j=1 ▷ Sample size-B batch
    σ ← 1 B (x,x′)∈Dk µ(ΦV→Sk (x),x′) ▷ Validate updated program
    M.Update(V → Sk, σ) ▷Update optimizer based on the observed validation score
end for
(V → Sk) ←M.ExtractOptimizedSets()
return ΦV→S
```
