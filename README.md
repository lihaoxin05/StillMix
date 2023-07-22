# StillMix
This repo is the implementation of StillMix (ICCV2023): [**Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground**](https://arxiv.org/abs/2211.12883).

The codes are organized into three folders:
1. The [benchmark_synthesis](benchmark_synthesis) supports OOD benchmark synthesis.
2. The [reference_network](reference_network) supports the training of the reference networks.
3. The [main_network](main_network) supports the training of the main networks.

## Datasets
### Training and IID evaluation
We use [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) and [Kinetics400](https://www.deepmind.com/open-source/kinetics) in our training and IID evaluation. You can prepare these datasets following [MMAction2](https://github.com/open-mmlab/mmaction2).

We provide the dataset splits used in our paper on [Google Drive](https://drive.google.com/drive/folders/1H2_rfwImw6HtUd2PfiASInAy9WylnkSQ?usp=drive_link).

### OOD evaluation
We synthesize benchmarks for OOD evaluation. For details, please refer to our paper. We include the code for OOD benchmark synthesis in the [benchmark_synthesis](benchmark_synthesis) folder.

We release SCUBA and ConflFG on [OneDrive](https://onedrive.live.com/?authkey=%21AH8Ach9qjnQQh4Q&id=C88D845F827102A9%21111362&cid=C88D845F827102A9). Since SCUFO is randomly sampled from SCUBA, we do not release SCUFO due to the limited storage space. Our experiments show that constructing SCOFO by sampling SCUBA with different random seeds only leads to slight performance difference (<1.0%) on SCUFO.

We provide the dataset splits used in our paper on [Google Drive](https://drive.google.com/drive/folders/1H2_rfwImw6HtUd2PfiASInAy9WylnkSQ?usp=drive_link).

## Reference Network
For training the reference network, please go to [reference_network](reference_network).

We save the results as pkl files. They are released on [Google Drive](https://drive.google.com/drive/folders/1vL0IckepbfUmZh7vEXlp7LIH4BeHhuzm?usp=drive_link).

## Main Network
For training the main network, please go to [main_network](main_network).

## Contact
If you have any problem please email me at haoxin003@e.ntu.edu.sg or lihaoxin05@gmail.com.