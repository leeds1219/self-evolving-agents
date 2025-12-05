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
## [MIPRO](https://dspy.ai/)

## [GEPA](https://github.com/gepa-ai/gepa-artifact)
