# [Self-Evolving-Agents](https://arxiv.org/pdf/2507.21046)

## [DSPy](https://github.com/stanfordnlp/dspy) 
DSPy (The authors do not provide what the abbreviation is, so my guess is "D"eclarative "S"elf-improving "Py"thon) is a programming model.
DSPy objective is to replace the manual instruction prompts for LLMs.
DSPy replace manual instruction prompts with a short declarative spec.

### Imperative vs declarative spec
```
# 1. Update package list
apt-get update
# 2. Install nginx
apt-get install -y nginx
# 3. Start the service
service nginx start
```
Control Flow (How to do it step-by-step) 

```
# We just declare: "I want Nginx to be present."
- name: Ensure Nginx is installed
  package:
    name: nginx
    state: present  # <--- This is the Declarative Spec
```
Logic / Outcome (What to achieve)

### DSPy signatures
The paper proposes a DSPy signature, which is a tuple of _input_fields_ and _output_fields_ with an optional _instruction_.
```
qa = dspy.Predict("question-> answer")
qa(question="Where is Guaran´ ı spoken?")
# Out: Prediction(answer=’Guaran´ ı is spoken mainly in South America.’)
```
or
```
class GenerateSearchQuery(dspy.Signature):
 """Write a simple search query that will help answer a complex question."""

 context = dspy.InputField(desc="may contain relevant facts")
 question = dspy.InputField()
 query = dspy.OutputField(dtype=dspy.SearchQuery)
 ```

### These signitures can be self-improved, but how?
To use a DSPy signature, one must declare a module with that signature such as the _Predict_ module from the first example. This module declaration returns a _function_ having that signature. 

What does this **mean**?

#### The _Predict_ Module
 ```
class Predict(dspy.Module):
  def __init__(self, signature, **config):
    self.signature = dspy.Signature(signature)
    self.config = config
   
    # Module Parameters.
    self.lm = dspy.ParameterLM(None) # use the default LM
    self.demonstrations = dspy.ParameterDemonstrations([])
 
  def forward(self, **kwargs):
    lm = get_the_right_lm(self.lm, kwargs)
    signature = get_the_right_signature(self.signature, kwargs)
    demonstrations = get_the_right_demonstrations(self.demonstrations, kwargs)
   
    prompt = signature(demos=self.demos, **kwargs)
    completions = lm.generate(prompt, **self.config)
    prediction = Prediction.from_completions(completions, signature=signature)
   
    if dsp.settings.compiling is not None:
      trace = dict(predictor=self, inputs=kwargs, outputs=prediction)
      dspy.settings.traces.append(trace)
   
    return prediction
 ```
 
## [MIPRO](https://dspy.ai/)

## [GEPA](https://github.com/gepa-ai/gepa-artifact)
