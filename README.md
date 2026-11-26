## Scaling Full Conformal Image Classifiers.

The official implementation of *Targeted Full Conformal Prediction*, a conformal procedure that scales full conformal strategies to large label spaces in image classification, improving efficiency and coverage stability.<br/>
📜 <span style="color:violet">*Conference on Neural Information Processing Systems (NeurIPS)*</span> \
[Julio Silva-Rodríguez](https://scholar.google.es/citations?user=1UMYgHMAAAAJ&hl) ⋅
[Ender Konukoglu<sup>](https://scholar.google.es/citations?user=OeEMrhQAAAAJ&hl) <br/>
Computer Vision Lab, ETH Zürich.<br/>
[ [Manuscript](https://arxiv.org/pdf/2609.37298) ] [ [OpenReview](https://openreview.net/forum?id=pwt7abXpU8) ] 
<br/>


### Install

* Install in your enviroment a compatible torch version with your GPU. For example:
```
conda create -n tfcp python=3.11 -y
conda activate tfcp
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

```
git clone https://github.com/jusiro/T-FCP.git
cd T-FCP
pip install -r requirements.txt
```

### Usage

1. Adequate dataset. Check [`./local_data/datasets/README.md`](./local_data/datasets/README.md) for additional comments.

2. Define dataset path at [`./local_data/constants.py`](./local_data/constants.py).

3. Use [`run_features.py`](run_features.py) to pre-compute the visual representations. 

4. Run inductive conformal prediction (ICP).
- `python run_conformal.py --K 16 --procedure icp --alpha_icp 0.10`

5. Run split conformal prediction (SCP).
- `python run_conformal.py --K 16 --procedure scp --alpha_icp 0.10 --solver gd`

6. Run targeted full conformal prediciton (T-FCP).
- `python run_conformal.py --K 16 --procedure tfcp --alpha_icp 0.005 --alpha_fcp 0.095 --solver lda`

## Citation

If you find this repository useful, please consider citing this paper:
```
@inproceedings{tfcp26,
    author    = {Silva-Rodríguez, Julio and Konukoglu, Ender},
    title     = {Scaling Full Conformal Image Classifiers},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2026},
    volume    = {39},
    pages     = {TBD}
}
```
