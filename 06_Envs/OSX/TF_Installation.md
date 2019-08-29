# TensorFlow Installation

- Error

  `Your CPU supports instructions that his TensorFlow binary was not compiled to use`

  Machine Instruction Set이 맞지 않아, 목표하는 성능을 낼 수 없다는 Warning

  1. 무시

     ```python
     import os
     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
     ```

  2. Build from Source

