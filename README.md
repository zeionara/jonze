# Jonze
Joint Slot and Intent Extraction implementation in Tensorflow2.0

Contains restructured code from the following repo:
https://github.com/shubham8111/Joint-NLU

Implementation of Bi-LSTM based NLU baseline and SlotGated-SLU  (Goo et al, 2018)(https://www.csie.ntu.edu.tw/~yvchen/doc/NAACL18_SlotGated.pdf) 
Models are evaulated on Snips and ATIS datasets.

Experiments did not reproduce improvements by SlotGated model over Basline model, on snips dataset.


Preprocessing modules reused from following repo:
https://github.com/MiuLab/SlotGated-SLU/

## Usage
To install package:  
```sh
pip install jonze
```  
To train model:  
```python
from jonze import train  
train(dataset = "joint-nlu", datasets_root = "dataset", models_root = "model", layer_size=12)
```
To test model:  
```python
from jonze import test  
test(dataset = "joint-nlu", datasets_root = "dataset", models_root = "model", layer_size=12, batch_size=46)
``` 
## Results

### Snips Dataset:


| Model      | Slot F1 | Intent accuracy | Semantic Accuracy |
|------------|---------|-----------------|-------------------|
| Baseline   | 84.30   | 96.57           | 66.43             |
| Slot Gated | 83.5    | 95.57           | 66.85             |

### Atis Dataset:

| Model      | Slot F1 | Intent accuracy | Semantic Accuracy |
|------------|---------|-----------------|-------------------|
| Baseline   | 95.08   | 94.62           | 81.97             |
| Slot Gated | 94.57   | 96.41           | 83.65             |

P.S.  Sometimes Slot F1 might get stuck at zero during training, better weight intialization or training a few epochs only on slot loss can resolve the issue. 
