# Multi-set loss

### Data generation

Generate data with replacement by:

bash generate_data_with_replacement.sh [LENGTH_OF_SEQUENCE]

Sequences ranging from 1 to 10 in length will be generated if LENGTH_OF_SEQUENCE is not specified.

Generate data without replacement by:

bash generate_data_without_replacement.sh [LENGTH_OF_SEQUENCE]

Sequences ranging from 1 to 10 in length will be generated if LENGTH_OF_SEQUENCE is not specified. LENGTH_OF_SEQUENCE cannot exceed 10.

### Criterion evaluation

Evaluate criterions on data generated with replacement by:

evaluate_with_replacement.sh

Evaluate criterions on data generated without replacement by:

bash evaluate_without_replacement.sh

Investigate how the strength of entropy regularizer impacts convergence by:

bash evaluate_entropy_scale.sh [LENGTH_OF_SEQUENCE]

### TODO

- 10-digit sequences generated without replacement
- Experiment on sequences generated with replacement (especially, the performance of RL-based loss, which appears to be unplausible when length of sequence is 1)
- Maximum length of sequence that can be handled
- Noisy signal on LeNet
