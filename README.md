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
Like layers in PyTorch, the instantiated module behaves as a callable function: it takes in keyword arguments corresponding to the signature input fields (e.g., question), formats a prompt to implement the signature and includes the appropriate demonstrations, calls the LM, and parses the output fields.

**Then how do we parameterize these so called signatures and modules?**

We need a specific LM to call (_lm_), prompt instructions (_signature_) and a few-shot examples (_demonstrations_). These are the parameters for optimization.

**Then how can we optimize these parameters?**

### _Teleprompter_
```
class SimplifiedBootstrapFewShot(Teleprompter):
  def __init__(self, metric=None):
  self.metric = metric
 
  def compile(self, student, trainset, teacher=None):
  teacher = teacher if teacher is not None else student
  compiled_program = student.deepcopy()
 
  # Step 1. Prepare mappings between student and teacher Predict modules.
  # Note: other modules will rely on Predict internally.
  assert student_and_teacher_have_compatible_predict_modules(student, teacher)
  name2predictor, predictor2name = map_predictors_recursively(student, teacher)
 
  # Step 2. Bootstrap traces for each Predict module.
  # We’ll loop over the training set. We’ll try each example once for simplicity.
  for example in trainset:
    if we_found_enough_bootstrapped_demos(): break
   
    # turn on compiling mode which will allow us to keep track of the traces
    with dspy.setting.context(compiling=True):
      # run the teacher program on the example, and get its final prediction
      # note that compiling=True may affect the internal behavior here
      prediction = teacher(**example.inputs())
     
      # get the trace of the all interal Predict calls from teacher program
      predicted_traces = dspy.settings.trace
   
    # if the prediction is valid, add the example to the traces
    if self.metric(example, prediction, predicted_traces):
      for predictor, inputs, outputs in predicted_traces:
        d = dspy.Example(automated=True, **inputs, **outputs)
        predictor_name = self.predictor2name[id(predictor)]
        compiled_program[predictor_name].demonstrations.append(d)
  return compiled_program
```
```
class ChainOfThought(dspy.Module):
  def __init__(self, signature):
    # Modify signature from ‘*inputs-> *outputs‘ to ‘*inputs-> rationale, *outputs‘.
    rationale_field = dspy.OutputField(prefix="Reasoning: Let’s think step by step.")
    signature = dspy.Signature(signature).prepend_output_field(rationale_field)
 
    # Declare a sub-module with the modified signature.
    self.predict = dspy.Predict(signature)
 
  def forward(self, **kwargs):
    # Just forward the inputs to the sub-module.
    return self.predict(**kwargs)
```
```
class RAG(dspy.Module):
  def __init__(self, num_passages=3):
    # ‘Retrieve‘ will use the user’s default retrieval settings unless overriden.
    self.retrieve = dspy.Retrieve(k=num_passages)
    # ‘ChainOfThought‘ with signature that generates answers given retrieval & question.
    self.generate_answer = dspy.ChainOfThought("context, question-> answer")
 
  def forward(self, question):
    context = self.retrieve(question).passages
    return self.generate_answer(context=context, question=question)
```
```
def answer_and_context_match(example, pred, trace=None):
  answer_match = dspy.evaluate.answer_exact_match(example, pred)
 
  # Is the prediction a substring of some passage?
  context_match = any((pred.answer.lower() in c) for c in pred.context)
 
  return answer_match and context_match
```
```
# Small training set with only questions and final answers.
qa_trainset = [dspy.Example(question="What is the capital of France?", answer="Paris")]
 
# The teleprompter will bootstrap missing labels: reasoning chains and retrieval contexts.
teleprompter = dspy.BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)
compiled_rag = teleprompter.compile(RAG(), trainset=qa_trainset)
```
Using the EM (exact match based metric of evaluating an answer) we can find the best demonstrations and instructions using the _teleprompter_ as an optimizer.

#### Stage 1 Candidate Generation
The compiler (_teleprompter.compile(RAG(), trainset=qa_trainset)_) recursively find all unique _Predict_ modules. Simple rejection-sampling with an zero-shot baseline can work here. 

#### Stage 2 Parameter Optimization
Hyperparameter tuning algorithms (e.g., random search or Tree structured Parzen Estimators as in [HyperOpt](https://github.com/hyperopt/hyperopt) and [Optuna](https://optuna.org/) can be applied for selection among candidates.

~~Another type of optimization is finetuning with _BootstrapFinetune_, where the demonstrations are  used to update the LM’s weights for each predictor. When this is applied, the LM parameter of each  module is updated to the new LM weights. Typically, we are optimizing average quality using the metric with cross-validation over the training set or a validation set. This is applicable even with no labels for any stages, depending on the nature of metric.~~

#### Stage 3 High-Order Program Optimization
A different type of optimization that the DSPy compiler supports is modifying the control flow of the program. 
One of the simplest forms of these is ensembles, which we use in the case studies in this work. An ensemble will bootstrap multiple copies of the same program, and then replace the program with a new one that runs them all in parallel and reduces their predictions into one with a custom function (e.g., majority voting).

## [MIPRO](https://dspy.ai/)

## [GEPA](https://github.com/gepa-ai/gepa-artifact)
