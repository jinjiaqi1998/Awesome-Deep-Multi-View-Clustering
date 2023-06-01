[stars-img]: https://img.shields.io/github/stars/jinjiaqi1998/Awesome-Deep-Multiview-Clustering?color=yellow
[stars-url]: https://github.com/jinjiaqi1998/Awesome-Deep-Multiview-Clustering/stargazers
[fork-img]: https://img.shields.io/github/forks/jinjiaqi1998/Awesome-Deep-Multiview-Clustering?color=lightblue&label=fork
[fork-url]: https://github.com/jinjiaqi1998/Awesome-Deep-Multiview-Clustering/network/members


# Awesome-Deep-Multiview-Clustering
Collections for state-of-the-art and novel deep neural network-based multi-view clustering approaches (papers & codes). According to the integrity of multi-view data, such methods can be further subdivided into Deep Multi-view Clustering(**DMVC**) and Deep Incomplete Multi-view Clustering(**DIMVC**).

We are looking forward for other participants to share their papers and codes. If interested or any question about the listed papers and codes, please contanct <jinjiaqi@nudt.edu.cn>. If you find this repository useful to your research or work, it is really appreciated to star this repository. :sparkles: If you use our code or the processed datasets in this repository for your research, please cite 1-2 papers in the citation part [here](#jump4). :heart:

[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]



##  Table of Contents
- [What's Deep Multi-view Clustering?](#jump1) 
- [Surveys](#jump2) 
- [Papers & Codes](#jump3)
    - [Deep Multi-view Clustering(DMVC)](#jump31)
    - [Deep Incomplete Multi-view Clustering(DIMVC)](#jump32)
- [Citation](#jump4)



--------------

## <span id="jump1">What's Deep Multi-view Clustering? </span>
Deep multi-view clustering aims to reveal the potential complementary information of multiple features or modalities through deep neural networks, and finally divide samples into different groups in unsupervised scenarios.

<div  align="center">    
    <img src="./DMVC_frame.png" width=70% />
</div>

##  <span id="jump2">Surveys </span>

| Year | Title                                                        |    Venue    |                            Paper                             |
| ---- | ------------------------------------------------------------ | :---------: | :----------------------------------------------------------: |
| 2023 | **A Comprehensive Survey on Multi-view Clustering** |    TKDE   | [paper](https://ieeexplore.ieee.org/abstract/document/10108535) |
| 2022 | **Representation Learning in Multi-view Clustering: A Literature Review** |    DSE    | [Link](https://arxiv.org/abs/2211.12875) |
| 2022 | **A Comprehensive Survey on Community Detection with Deep Learning** |    TNNLS    | [Link](https://arxiv.org/pdf/2105.12584.pdf?ref=https://githubhelp.com) |
| 2021 | **A Comprehensive Survey on Graph Neural Networks**          |    TNNLS    | [Link](https://ieeexplore.ieee.org/abstract/document/9046288) |
| 2018 | **Deep Learning for Community Detection: Progress, Challenges and Opportunities** |    IJCAI    |           [Link](https://arxiv.org/pdf/2005.08225)           |
| 2018 | **A survey of clustering with deep learning: From the perspective of network architecture** | IEEE Access | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8412085) |
| 2017 | **A survey of clustering with deep learning: From the perspective of network architecture** | IEEE Access | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8412085) |
| 2013 | **A survey of clustering with deep learning: From the perspective of network architecture** | IEEE Access | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8412085) |

1. **CVPR**: Deep Safe Multi-View Clustering(**DSMVC**).[![paper](https://img.shields.io/badge/%20-paper-blue)](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Deep_Safe_Multi-View_Clustering_Reducing_the_Risk_of_Clustering_Performance_CVPR_2022_paper.pdf)




1. 2023 **TKDE**: A Comprehensive Survey on Multi-view Clustering.[<a href= "https://ieeexplore.ieee.org/abstract/document/10108535" target="_blank">Paper</a>]

1. 2022 **DSE**: Representation Learning in Multi-view Clustering: A Literature Review.[<a href= "https://link.springer.com/article/10.1007/s41019-022-00190-8" target="_blank">Paper</a>]

1. 2022 **Arxiv**: Foundations and Recent Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions.[<a href= "https://arxiv.org/pdf/2209.03430" target="_blank">Paper</a>]

1. 2021 **Neurocom**: Deep Multi-view Learning Methods: A Review.[<a href= "https://researchportal.port.ac.uk/files/26919776/Manuscript_R_pp.pdf" target="_blank">Paper</a>]

1. 2018 **BDMA**: Multi-view Clustering: A Survey.[<a href= "https://ieeexplore.ieee.org/iel7/8254253/8336843/08336846.pdf" target="_blank">Paper</a>]

1. 2018 **TPAMI**: Multimodal Machine Learning: A Survey and Taxonomy.[<a href= "https://arxiv.org/pdf/1705.09406" target="_blank">Paper</a>]

1. 2017 **Arxiv**: Multi-view Learning Overview：Recent Progress and New Challenges.[<a href= "https://shiliangsun.github.io/pubs/MVLoverviewIF17.pdf" target="_blank">Paper</a>]

1. 2013 **Arxiv**: A Survey on Multi-view Learning.[<a href= "https://arxiv.org/pdf/1304.5634" target="_blank">Paper</a>]

---

## <span id="jump3">Papers & Codes </span>
According to the integrity of multi-view data, the paper is divided into deep multi-view clustering methods and deep incomplete multi-view clustering approaches.

### <span id="jump31">Deep Multi-view Clustering(DMVC)</span> 

#### **2023**
1. **TIP**: Self-Supervised Information Bottleneck for Deep Multi-View Subspace Clustering(**SIB-MSC**).[<a href= "https://arxiv.org/pdf/2204.12496.pdf" target="_blank">Paper</a>]

1. **TNSE**: Multi-channel Augmented Graph Embedding Convolutional Network for Multi-view Clustering(**MAGEC-Net**).[<a href= "https://ieeexplore.ieee.org/abstract/document/10043740/" target="_blank">Paper</a>]

#### **2022**
1. **CVPR**: Deep Safe Multi-View Clustering：Reducing the Risk of Clustering Performance Degradation Caused by View Increase(**DSMVC**).[<a href= "https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Deep_Safe_Multi-View_Clustering_Reducing_the_Risk_of_Clustering_Performance_CVPR_2022_paper.pdf" target="_blank">Paper</a>] [<a href="https://github.com/Gasteinh/DSMVC" target="_blank">Code</a>]

1. **CVPR**: Multi-level Feature Learning for Contrastive Multi-view Clustering(**MFLVC**).[<a href= "https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Multi-Level_Feature_Learning_for_Contrastive_Multi-View_Clustering_CVPR_2022_paper.pdf" target="_blank">Paper</a>] [<a href="https://github.com/SubmissionsIn/MFLVC" target="_blank">Code</a>]

1. **AAAI**: Stationary Diffusion State Neural Estimation for Multiview Clustering(**SDSNE**).[<a href= "https://www.aaai.org/AAAI22Papers/AAAI-184.LiuC.pdf" target="_blank">Paper</a>] [<a href="https://github.com/kunzhan/SDSNE" target="_blank">Code</a>]

1. **TNNLS**: Multi-View Subspace Clustering via Structured Multi-Pathway Network(**SMpNet**).[<a href= "http://cic.tju.edu.cn/faculty/huqinghua/pdf/GeneralizedLatentMulti-ViewSubspaceClustering.pdf" target="_blank">Paper</a>] [<a href="http://cic.tju.edu.cn/faculty/zhangchangqing/code.html" target="_blank">Code</a>]

1. **TNNLS**: Self-Supervised Deep Multiview Spectral Clustering(**SDMvSC**).[<a href= "https://ieeexplore.ieee.org/abstract/document/9853217/" target="_blank">Paper</a>]

1. **IJCAI**: Contrastive Multi-view Hyperbolic Hierarchical Clustering(**CMHHC**).[<a href= "https://arxiv.org/pdf/2205.02618.pdf" target="_blank">Paper</a>]

1. **NN**: Multi-view Graph Embedding Clustering Network：Joint Self-supervision and Block Diagonal Representation(**MVGC**).[<a href= "https://www.sciencedirect.com/science/article/pii/S089360802100397X" target="_blank">Paper</a>] [<a href="https://github.com/xdweixia/NN-2022-MVGC" target="_blank">Code</a>]

1. **APPL INTELL**: Efficient Multi‑view Clustering Networks(**EMC-Nets**).[<a href= "https://link.springer.com/article/10.1007/s10489-021-03129-0" target="_blank">Paper</a>] [<a href="https://github.com/Guanzhou-Ke/EMC-Nets" target="_blank">Code</a>]

#### **2021**
1. **AAAI**: Deep Mutual Information Maximin for Cross-Modal Clustering(**DMIM**).[<a href= "https://ojs.aaai.org/index.php/AAAI/article/view/17076/16883" target="_blank">Paper</a>]

1. **CVPR**: Reconsidering Representation Alignment for Multi-view Clustering(**SiMVC&CoMVC**).[<a href= "https://openaccess.thecvf.com/content/CVPR2021/papers/Trosten_Reconsidering_Representation_Alignment_for_Multi-View_Clustering_CVPR_2021_paper.pdf" target="_blank">Paper</a>] [<a href="https://github.com/AllenWrong/mvc" target="_blank">Code</a>]

1. **DSE**: Deep Multiple Auto-Encoder-Based Multi-view Clustering(**MVC_MAE**).[<a href= "https://link.springer.com/article/10.1007/s41019-021-00159-z" target="_blank">Paper</a>] [<a href="https://github.com/dugzzuli/Deep-Multiple-Auto-Encoder-Based-Multi-view-Clustering" target="_blank">Code</a>]

1. **ICCV**: Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos(**MCN**).[<a href= "https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Multimodal_Clustering_Networks_for_Self-Supervised_Learning_From_Unlabeled_Videos_ICCV_2021_paper.pdf" target="_blank">Paper</a>] [<a href="https://github.com/brian7685/Multimodal-Clustering-Network" target="_blank">Code</a>]

1. **ICCV**: Multi-VAE: Learning Disentangled View-common and View-peculiar Visual Representations for Multi-view Clustering(**Multi-VAE**).[<a href= "https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Multi-VAE_Learning_Disentangled_View-Common_and_View-Peculiar_Visual_Representations_for_Multi-View_ICCV_2021_paper.pdf" target="_blank">Paper</a>] [<a href="https://github.com/SubmissionsIn/Multi-VAE" target="_blank">Code</a>]

1. **IJCAI**: Graph Filter-based Multi-view Attributed Graph Clustering(**MvAGC**).[<a href= "https://www.ijcai.org/proceedings/2021/0375.pdf" target="_blank">Paper</a>] [<a href="https://github.com/sckangz/MvAGC" target="_blank">Code</a>]

1. **Neurcomputing**: Multi-view Subspace Clustering Networks with Local and Global Graph Information(**MSCNGL**).[<a href= "https://arxiv.53yu.com/pdf/2010.09323" target="_blank">Paper</a>] [<a href="https://github.com/qinghai-zheng/MSCNLG" target="_blank">Code</a>]

1. **NeurIPS**: Multi-view Contrastive Graph Clustering(**MCGC**).[<a href= "https://proceedings.neurips.cc/paper/2021/file/10c66082c124f8afe3df4886f5e516e0-Paper.pdf" target="_blank">Paper</a>] [<a href="https://github.com/panern/mcgc" target="_blank">Code</a>]

1. **TKDE**: Self-supervised Discriminative Feature Learning for Deep Multi-view Clustering(**SDMVC**).[<a href= "https://arxiv.org/pdf/2103.15069.pdf" target="_blank">Paper</a>] [<a href="https://github.com/SubmissionsIn/SDMVC" target="_blank">Code</a>]

1. **TKDE**: Multi-view Attributed Graph Clustering(**MAGC**).[<a href= "https://www.researchgate.net/publication/353747180_Multi-view_Attributed_Graph_Clustering" target="_blank">Paper</a>] [<a href="https://github.com/sckangz/MAGC" target="_blank">Code</a>]

1. **TMM**: Deep Multi-view Subspace Clustering with Unified and Discriminative Learning(**DMSC-UDL**).[<a href= "https://ieeexplore.ieee.org/abstract/document/9204408/" target="_blank">Paper</a>] [<a href="https://github.com/IMKBLE/DMSC-UDL" target="_blank">Code</a>]

1. **TMM**: Self-supervised Graph Convolutional Network for Multi-view Clustering(**SGCMC**).[<a href= "https://ieeexplore.ieee.org/abstract/document/9472979/" target="_blank">Paper</a>] [<a href="https://github.com/xdweixia/SGCMC" target="_blank">Code</a>]

1. **TMM**: Consistent Multiple Graph Embedding for Multi-View Clustering(**CMGEC**).[<a href= "https://arxiv.org/pdf/2105.04880" target="_blank">Paper</a>] [<a href="https://github.com/wangemm/CMGEC-TMM-2021" target="_blank">Code</a>]

1. **TNNLS**: Deep Multiview Collaborative Clustering(**DMCC**).[<a href= "https://see.xidian.edu.cn/faculty/chdeng/Welcome%20to%20Cheng%20Deng's%20Homepage_files/Papers/Journal/TNNLS2021_Xu.pdf" target="_blank">Paper</a>]

1. **ACM MM**: Consistent Multiple Graph Embedding for Multi-View Clustering(**CMGEC**).[<a href= "https://arxiv.org/pdf/2105.04880.pdf" target="_blank">Paper</a>] [<a href="https://github.com/wangemm/CMGEC" target="_blank">Code</a>]

#### **2020**
1. **AAAI**: Cross-modal Subspace Clustering via Deep Canonical Correlation Analysis(**CMSC-DCCA**).[<a href= "https://ojs.aaai.org/index.php/AAAI/article/view/5808/5664" target="_blank">Paper</a>]

1. **AAAI**: Shared Generative Latent Representation Learning for Multi-View Clustering(**DMVCVAE**).[<a href= "https://ojs.aaai.org/index.php/AAAI/article/download/6146/6002" target="_blank">Paper</a>] [<a href="https://github.com/whytin95/DMVCVAE" target="_blank">Code</a>]

1. **CVPR**: End-to-End Adversarial-Attention Network for Multi-Modal Clustering(**EAMC**).[<a href= "https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_End-to-End_Adversarial-Attention_Network_for_Multi-Modal_Clustering_CVPR_2020_paper.pdf" target="_blank">Paper</a>] [<a href="https://github.com/AllenWrong/mvc" target="_blank">Code</a>]

1. **IJCAI**: Multi-View Attribute Graph Convolution Networks for Clustering(**MAGCN**).[<a href= "https://www.ijcai.org/proceedings/2020/0411.pdf" target="_blank">Paper</a>] [<a href="https://github.com/IMKBLE/MAGCN" target="_blank">Code</a>]

1. **ICME**: End-To-End Deep Multimodal Clustering(**DMMC**).[<a href= "https://ieeexplore.ieee.org/abstract/document/9102921/" target="_blank">Paper</a>] [<a href="https://github.com/Guanzhou-Ke/DMMC-zoo" target="_blank">Code</a>]

1. **IS**: Deep Embedded Multi-view Clustering with Collaborative Training(**DEMVC**).[<a href= "https://arxiv.org/pdf/2007.13067.pdf" target="_blank">Paper</a>] [<a href="https://github.com/SubmissionsIn/DEMVC" target="_blank">Code</a>]

1. **TKDE**: Joint Deep Multi-View Learning for Image Clustering(**DMJC**).[<a href= "https://ieeexplore.ieee.org/abstract/document/8999493/" target="_blank">Paper</a>]

1. **WWW**: One2Multi Graph Autoencoder for Multi-view Graph Clustering(**O2MVC**).[<a href= "http://shichuan.org/doc/83.pdf" target="_blank">Paper</a>] [<a href="https://github.com/googlebaba/WWW2020-O2MAC" target="_blank">Code</a>]

#### **2019**
1. **CVPR**: AE^2-Nets: Autoencoder in Autoencoder Networks(**AE^2-Nets**).[<a href= "http://cic.tju.edu.cn/faculty/zhangchangqing/pub/AE2_Nets.pdf" target="_blank">Paper</a>] [<a href="https://github.com/willow617/AE2-Nets" target="_blank">Code</a>]

1. **ICML**: COMIC: Multi-view Clustering Without Parameter Selection(**COMIC**).[<a href= "http://proceedings.mlr.press/v97/peng19a/peng19a.pdf" target="_blank">Paper</a>] [<a href="https://github.com/limit-scu/2019-ICML-COMIC" target="_blank">Code</a>]

1. **IJCAI**: Deep Adversarial Multi-view Clustering Network(**DAMC**).[<a href= "https://www.researchgate.net/publication/334844473_Deep_Adversarial_Multi-view_Clustering_Network" target="_blank">Paper</a>] [<a href="https://github.com/IMKBLE/DAMC" target="_blank">Code</a>]

1. **IJCAI**: Multi-view Spectral Clustering Network(**MvSCN**).[<a href= "https://www.ijcai.org/Proceedings/2019/0356.pdf">Paper</a>] [<a href="https://github.com/limit-scu/2019-IJCAI-MvSCN" target="_blank">Code</a>]

1. **TIP**: Multi-view Deep Subspace Clustering Networks(**MvDSCN**).[<a href= "https://arxiv.org/abs/1908.01978" target="_blank">Paper</a>] [<a href="https://github.com/huybery/MvDSCN" target="_blank">Code</a>]

#### **2018**
1. **TPAMI**: Generalized Latent Multi-View Subspace Clustering(**gLMSC**).[<a href= "http://cic.tju.edu.cn/faculty/huqinghua/pdf/GeneralizedLatentMulti-ViewSubspaceClustering.pdf" target="_blank">Paper</a>] [<a href="http://cic.tju.edu.cn/faculty/zhangchangqing/code.html" target="_blank">Code</a>]

1. **STSP**: Deep Multimodal Subspace Clustering Networks(**DMSC**).[<a href= "https://arxiv.org/pdf/1804.06498.pdf" target="_blank">Paper</a>] [<a href="https://github.com/mahdiabavisani/Deep-multimodal-subspace-clustering-networks" target="_blank">Code</a>]

1. **CoRR**: Deep Multi-View Clustering via Multiple Embedding(**DMVC-ME**).[<a href= "https://deepai.org/publication/deep-multi-view-clustering-via-multiple-embedding" target="_blank">Paper</a>]

---

### <span id="jump32">Deep Incomplete Multi-view Clustering(DIMVC)</span> 
#### **2023**
1. **CVPR**: Deep Incomplete Multi-view Clustering with Cross-view Partial Sample and Prototype Alignment(**CPSPAN**).[<a href= "http://openaccess.thecvf.com/content/CVPR2023/papers/Jin_Deep_Incomplete_Multi-View_Clustering_With_Cross-View_Partial_Sample_and_Prototype_CVPR_2023_paper.pdf" target="_blank">Paper</a>]

1. **TIP**: Adaptive Feature Projection with Distribution Alignment for Deep Incomplete Multi-view Clustering(**APADC**).[<a href= "https://ieeexplore.ieee.org/abstract/document/10043822/" target="_blank">Paper</a>] [<a href="https://github.com/SubmissionsIn/APADC" target="_blank">Code</a>]

1. **NN**: Incomplete Multi-view Clustering Network via Nonlinear Manifold Embedding and Probability-Induced Loss(**IMCNet-MP**).[<a href= "https://www.sciencedirect.com/science/article/pii/S0893608023001302" target="_blank">Paper</a>]

#### **2022**
1. **TPAMI**: Robust Multi-view Clustering with Incomplete Information(**SURE**).[<a href= "https://ieeexplore.ieee.org/abstract/document/9723577/" target="_blank">Paper</a>] [<a href="https://github.com/XLearning-SCU/2022-TPAMI-SURE" target="_blank">Code</a>]

1. **TPAMI**: Dual Contrastive Prediction for Incomplete Multi-view Representation Learning(**DCP**).[<a href= "http://pengxi.me/wp-content/uploads/2022/08/DCP.pdf" target="_blank">Paper</a>] [<a href="https://github.com/XLearning-SCU/2021-CVPR-Completer" target="_blank">Code</a>]

1. **ICML**: Deep Safe Incomplete Multi-view Clustering: Theorem and Algorithm(**DSIMVC**).[<a href= "https://proceedings.mlr.press/v162/tang22c/tang22c.pdf" target="_blank">Paper</a>] [<a href="https://github.com/Gasteinh/DSIMVC" target="_blank">Code</a>]

1. **AAAI**: Deep Incomplete Multi-view Clustering via Mining Cluster Complementarity(**DIMVC**).[<a href= "https://ojs.aaai.org/index.php/AAAI/article/download/20856/20615" target="_blank">Paper</a>] [<a href="https://github.com/SubmissionsIn/DIMVC" target="_blank">Code</a>]

1. **ACM MM**: Robust Diversified Graph Contrastive Network for Incomplete Multi-view Clustering(**RDGC**).[<a href= "https://dl.acm.org/doi/abs/10.1145/3503161.3547894" target="_blank">Paper</a>] [<a href="https://github.com/zh-hike/RDGC" target="_blank">Code</a>]

1. **TCSVT**: Incomplete Multi-view Clustering via Cross-view Relation Transfer(**CRTC**).[<a href= "https://arxiv.org/pdf/2112.00739" target="_blank">Paper</a>]

1. **TMM**: Graph Contrastive Partial Multi-view Clustering(**AGCL**).[<a href= "https://ieeexplore.ieee.org/abstract/document/9904927/" target="_blank">Paper</a>] [<a href="https://github.com/wangemm/AGCL-TMM-2022" target="_blank">Code</a>]

#### **2021**
1. **CVPR**: COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction(**COMPLETER**).[<a href= "http://pengxi.me/wp-content/uploads/2021/03/2021CVPR-completer.pdf" target="_blank">Paper</a>] [<a href="https://github.com/XLearning-SCU/2021-CVPR-Completer" target="_blank">Code</a>]

1. **TIP**: iCmSC: Incomplete Cross-modal Subspace Clustering(**iCmSC**).[<a href= "https://ieeexplore.ieee.org/abstract/document/9259207" target="_blank">Paper</a>] [<a href="https://github.com/IMKBLE/iCmSC" target="_blank">Code</a>]

1. **TIP**: Generative Partial Multi-View Clustering With Adaptive Fusion and Cycle Consistency(**GP-MVC**).[<a href= "https://ieeexplore.ieee.org/abstract/document/9318542/" target="_blank">Paper</a>] [<a href="https://github.com/IMKBLE/GP-MVC" target="_blank">Code</a>]

1. **IJCAI**: Clustering-Induced Adaptive Structure Enhancing Network for Incomplete Multi-View Data(**CASEN**).[<a href= "https://www.ijcai.org/proceedings/2021/0445.pdf" target="_blank">Paper</a>]

1. **CIKM**: Structural Deep Incomplete Multi-view Clustering Network(**SDIMC-net**).[<a href= "https://dl.acm.org/doi/abs/10.1145/3459637.3482192" target="_blank">Paper</a>]

1. **SPL**: Dual Alignment Self-Supervised Incomplete Multi-View Subspace Clustering Network(**DASIMSC**).[<a href= "https://ieeexplore.ieee.org/abstract/document/9573269/" target="_blank">Paper</a>]

#### **2020**
1. **TPAMI**: Deep Partial Multi-View Learning(**DPML**).[<a href= "https://ieeexplore.ieee.org/document/9258396" target="_blank">Paper</a>] [<a href="http://cic.tju.edu.cn/faculty/zhangchangqing/code/DPML.zip" target="_blank">Code</a>]

1. **IJCAI**: CDIMC-net：Cognitive Deep Incomplete Multi-view Clustering Network(**CDIMC-net**).[<a href= "https://www.ijcai.org/proceedings/2020/447" target="_blank">Paper</a>] [<a href="https://github.com/DarrenZZhang/CDIMC-Net" target="_blank">Code</a>]

1. **ACM MM**: DIMC-net：Deep Incomplete Multi-view Clustering Network(**DIMC-net**).[<a href= "https://dl.acm.org/doi/10.1145/3394171.3413807" target="_blank">Paper</a>]

1. **ICDM**: Deep Incomplete Multi-View Multiple Clusterings(**DiMVMC**).[<a href= "https://arxiv.org/pdf/2010.02024" target="_blank">Paper</a>]

#### **2019**
1. **NeurIPS**: CPM-Nets: Cross Partial Multi-View Networks(**CPM-Nets**).[<a href= "https://papers.nips.cc/paper/2019/file/11b9842e0a271ff252c1903e7132cd68-Paper.pdf" target="_blank">Paper</a>] [<a href="https://github.com/hanmenghan/CPM_Nets" target="_blank">Code</a>]

1. **IJCAI**: Adversarial Incomplete Multi-view Clustering(**AIMC**).[<a href= "https://www.ijcai.org/proceedings/2019/0546.pdf" target="_blank">Paper</a>]

#### **2018**
1. **ICDM**: Partial Multi-View Clustering via Consistent GAN(**PVC-GAN**).[<a href= "https://drive.google.com/file/d/1RrVeq_FHkLSgltNd1bVfyaHhtIclV5ZG/view" target="_blank">Paper</a>] [<a href="https://github.com/IMKBLE/PVC-GAN" target="_blank">Code</a>]

---

## <span id="jump4">Citation </span>
```
@inproceedings{jin2023deep,
  title={Deep Incomplete Multi-view Clustering with Cross-view Partial Sample and Prototype Alignment},
  author={Jin, Jiaqi and Wang, Siwei and Dong, Zhibin and Liu, Xinwang and Zhu, En},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11600--11609},
  year={2023}
}
```
