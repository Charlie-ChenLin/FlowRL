# FlowRL: Matching Reward Distributions for LLM Reasoning

Official implementation of **FlowRL: Matching Reward Distributions for LLM Reasoning** based on [VERL](https://github.com/volcengine/verl).

## Quick Start

### Installation
```bash
cd verl@FlowRL
pip install -r requirements.txt
pip install -e .
```

### Training
```bash
cd verl@FlowRL/command/training/math
bash flowrl_7B_math.sh
```

### Testing
```bash
cd verl@FlowRL
python -m pytest tests/
```

## Structure
- `verl@FlowRL/command/training/` - Training scripts
- `verl@FlowRL/tests/` - Test suite
- `data_preprocess/` - Data preprocessing utilities

## Citation
```bibtex
@article{flowrl2024,
  title={FlowRL: Matching Reward Distributions for LLM Reasoning},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
