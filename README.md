# General Greedy De-bias for Dataset Biases
This is an extention of ["Greedy Gradient Ensemble for Robust Visual Question Answering" (ICCV 2021, Oral)](https://github.com/GeraldHan/GGE). 
The prerequisites can refer to [GGE](https://github.com/GeraldHan/GGE).

If you find this repo helpful, please cite
```
@article{han2021general,
	title={General Greedy De-bias Learning},
	author={Han, Xinzhe and Wang, Shuhui and Su, Chi and Huang, Qingming and Tian, Qi},
	journal={arXiv preprint arXiv:2112.10572 },
	year={2021}
}
```
```
@inproceedings{han2021greedy,
  title={Greedy gradient ensemble for robust visual question answering},
  author={Han, Xinzhe and Wang, Shuhui and Su, Chi and Huang, Qingming and Tian, Qi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1584--1593},
  year={2021}
}
```

## GGD

![figure1](figure/gge.pdf)
![figure2](figure/cr.pdf)

The core of GGD is to decompose the CE loss of GGE. The pytorch implementation is
```
loss = -(logits.log_softmax(-1) * labels).mean() + weight * (logits.log_softmax(-1) * bias * labels).mean()
```

## Biased-MNIST

Biased-MNIST is produced the same with [Re-bias](https://github.com/clovaai/rebias). Run the notebook in the `MNIST` folder. Modify `data_label_correlation` for different bias level.

## VQA-CP

Codes for VQA-CP is updated in [GGE](https://github.com/GeraldHan/GGE). To train a model with new GGD-CR, set `import base_model_sfce` in `mian.py` and run
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode gge_iter --debias ce_decompose --topq 1 --topv -1 --qvp 5 --output [] 
```

## Adversarial SQuAD

The experiment for AdQA is provided in `squad` folder. The code is modified from [squad](https://github.com/chrischute/squad).

## Acknowledgements

Some codes are modified from [CSS](https://github.com/yanxinzju/CSS-VQA), [UpDn](https://github.com/chrisc36/bottom-up-attention-vqa), [squad](https://github.com/chrischute/squad), and [Re-bias](https://github.com/clovaai/rebias). Thanks for their open source.



