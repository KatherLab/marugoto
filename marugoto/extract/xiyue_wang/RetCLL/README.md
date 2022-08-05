# RetCCL (This paper is under review)
####

### A better and stronger pre-trained model was built for various histopathological image applications. This model outperforms ImageNet pre-trained features by a large margin. We release our best model and invite researchers to test it on your computational pathology tasks.

#### Hardware

* 128GB of RAM
* 32*Nvidia V100 32G GPUs

### Preparations
1.Download all [TCGA](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D) 32000 WSIs. 

2.Download all [PAIP](http://wisepaip.org/paip) 2,457 WSIs. So, there will be about 15,000,000 images(~100T). It costs us $400,000 to advance the progress of digital pathology.



## Pre-trained models for histopathological image tasks
This pre-train model is [here](https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL?usp=sharing)
### 1.Classification through search
It is the most obvious and direct way to evaluate the distinctive power of the provided features.

|          |         | [TissueNet](https://www.drivendata.org/competitions/67/competition-cervicalbiopsy/page/254/) |    |    |
|----------:|--------:|------:|--------:|-------------:|
|         | Acc@1   | Acc@3    |   Acc@5 |   mMV@5 |
| ImageNet | 50.35   | 77.65 |    87.68 |         46.15 |                
| CCL (ours) | **67.09**  | **87.81**  |    **93.4** |         **70.1** |                  


|          |         | [UniToPatho](https://ieee-dataport.org/open-access/unitopatho) |    |    |
|----------:|--------:|------:|--------:|-------------:|
|         | Acc@1   | Acc@3    |   Acc@5 |   mMV@5 |
| ImageNet | 58.17   | 82.89 |    89.45 |         59.01 |                
| CCL (ours) | **66.55**  | **84.32**  |    **90.31** |         **68.35** | 



### 2.Multiple Instance Learning for Whole Slide Image Classification
This task is currently based on ImageNet pretrained features, which can also verify the superiority of our feature extractor.

|          |         | TCGA-NSCLC |
|----------:|--------:|------:|
|         |  Accuracy   | AUC    |   
| [ABMIL](https://arxiv.org/abs/1802.04712) |  0.7719   | 0.8656 |
| [MIL-RNN](https://www.nature.com/articles/s41591-019-0508-1) | 0.8619   |  0.9107 |
| [DSMIL](https://arxiv.org/abs/2011.08939) | 0.8058   | 0.8925 |
| [TransMIL](https://arxiv.org/abs/2106.00908) | 0.8835   |  0.9603 |
| [CLAM](https://www.nature.com/articles/s41551-020-00682-w) |  0.8422   |   0.9377 |
| CLAM+CCL (ours) |  **0.911**   |   **0.967**  |


### 3.Classification based on features using SVM
This task follows [KimiaNet](https://www.sciencedirect.com/science/article/pii/S1361841521000785)

|          |          [Colorectal cancer dataset](https://zenodo.org/record/53169#.YRfeKYgzbmE) |
|----------------:|-------------:|
|              |  Accuracy   |    
| [Combined features](https://www.nature.com/articles/srep27988) |  87.40   |
| [Fine-tuned VGG-19](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2184-4) | 86.19   | 
| [Ensemble of CNNs](https://www.sciencedirect.com/science/article/pii/S095219762100049X?dgcid=rss_sd_all) |  92.83   | 
| [KamiaNet](https://www.sciencedirect.com/science/article/pii/S1361841521000785) | 96.80   |
| CCL (ours) |  **98.40**   |   


#### If you want to compute the features.
```
python get_feature.py
```
It is recommended to first try to extract features at 1.0mpp, and then try other magnifications


#### If you want to fine-tune model.
```
python resnet_lincls.py
```

## License

RetCCL is released under the GPLv3 License and is available for non-commercial academic purposes.

### Citation
Please use below to cite this paper if you find our work useful in your research.
```
@inproceedings{wang2022RetCCL,
  title={RetCCL: Clustering-guided Contrastive Learning for Whole-slide Image Retrieval}

}
``` 

