# [Self-Evolving-Agents](https://arxiv.org/pdf/2507.21046)

## [MIPRO](https://dspy.ai/) 
Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs. 

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
Maximize the downstream metric! kind of similar to [ReSCORE](https://leeds1219.github.io/ReSCORE/).

