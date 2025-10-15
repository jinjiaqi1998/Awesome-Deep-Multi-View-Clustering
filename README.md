[stars-img]: https://img.shields.io/github/stars/jinjiaqi1998/Awesome-Deep-Multi-View-Clustering?color=yellow
[stars-url]: https://github.com/jinjiaqi1998/Awesome-Deep-Multi-View-Clustering/stargazers
[fork-img]: https://img.shields.io/github/forks/jinjiaqi1998/Awesome-Deep-Multi-View-Clustering?color=lightblue&label=fork
[fork-url]: https://github.com/jinjiaqi1998/Awesome-Deep-Multi-View-Clustering/network/members



# Awesome-Deep-Multi-View-Clustering
Collections for state-of-the-art and novel deep neural network-based multi-view clustering approaches (papers & codes). According to the integrity of multi-view data, such methods can be further subdivided into Deep Multi-view Clustering(**DMVC**) and Deep Incomplete Multi-view Clustering(**DIMVC**).

We are looking forward for other participants to share their papers and codes. If interested or any question about the listed papers and codes, please contact <jinjiaqi@nudt.edu.cn>. If you find this repository useful to your research or work, it is really appreciated to star this repository. :sparkles: If you use our code or the processed datasets in this repository for your research, please cite 1-2 papers in the citation part [here](#jump4). :heart:

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
| Year | Title                                                                 |    Venue    |    Paper    |
| ---- | --------------------------------------------------------------------- | :---------: | :---------: |
| 2025 | **Deep Multi-view Clustering：A Comprehensive Survey of the Contemporary Techniques** |    IF   | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S1566253525000855) |
| 2025 | **Advanced Unsupervised Learning：A Comprehensive Overview of Multi-view Clustering Techniques** |    AIR   | [![](https://img.shields.io/badge/-paper-blue)](https://link.springer.com/content/pdf/10.1007/s10462-025-11240-8.pdf) |
| 2024 | **A Survey and an Empirical Evaluation of Multi-view Clustering Approaches** |    ACM CS   | [![](https://img.shields.io/badge/-paper-blue)](https://www.researchgate.net/profile/Guowang-Du/publication/378091506_A_Survey_and_an_Empirical_Evaluation_of_Multi-view_Clustering_Approaches/links/65cc1c261bed776ae34f5e80/A-Survey-and-an-Empirical-Evaluation-of-Multi-View-Clustering-Approaches.pdf) |
| 2024 | **Self‐Supervised Multi‐View Clustering in Computer Vision: A Survey** |    IET CV   | [![](https://img.shields.io/badge/-paper-blue)](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/cvi2.12299) |
| 2024 | **The Methods for Improving Large-Scale Multi-View Clustering Efficiency: A Survey** |    AIR   | [![](https://img.shields.io/badge/-paper-blue)](https://link.springer.com/content/pdf/10.1007/s10462-024-10785-4.pdf) |
| 2024 | **Deep Clustering：A Comprehensive Survey** |    TNNLS   | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2210.04142) |
| 2024 | **Breaking Down Multi-view Clustering：A Comprehensive Review of Multi-view Approaches for Complex Data Structures** |    EAAI   | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0952197624000150) |
| 2024 | **Incomplete Multi-view Learning: Review, Analysis, and Prospects** |    ASC   | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S1568494624000528) |
| 2023 | **A Comprehensive Survey on Multi-view Clustering** |    TKDE   | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10108535) |
| 2022 | **Representation Learning in Multi-view Clustering: A Literature Review** | DSE | [![](https://img.shields.io/badge/-paper-blue)](https://link.springer.com/article/10.1007/s41019-022-00190-8) |
| 2022 | **Foundations and Recent Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions** | Arxiv | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2209.03430) |
| 2021 | **Survey on Deep Multi-modal Data Analytics: Collaboration, Rivalry, and Fusion** | TOMM | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2006.08159.pdf) |
| 2021 | **Deep Multi-view Learning Methods: A Review** | Neurocom | [![](https://img.shields.io/badge/-paper-blue)](https://researchportal.port.ac.uk/files/26919776/Manuscript_R_pp.pdf) |
| 2018 | **A Survey of Multi-View Representation Learning** | TKDE | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/1610.01206) |
| 2018 | **Multi-view Clustering: A Survey** | BDMA | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/iel7/8254253/8336843/08336846.pdf) |
| 2018 | **Multimodal Machine Learning: A Survey and Taxonomy** | TPAMI | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/1705.09406) |
| 2018 | **A Survey on Multi-View Clustering** | Arxiv | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/1712.06246.pdf) |
| 2017 | **Multi-view Learning Overview：Recent Progress and New Challenges** | IF | [![](https://img.shields.io/badge/-paper-blue)](https://shiliangsun.github.io/pubs/MVLoverviewIF17.pdf) |
| 2013 | **A Survey on Multi-view Learning** | Arxiv | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/1304.5634) |

---

## <span id="jump3">Papers & Codes </span>
According to the integrity of multi-view data, the paper is divided into deep multi-view clustering methods and deep incomplete multi-view clustering approaches.

### <span id="jump31">Deep Multi-view Clustering(DMVC)</span> 

| Year | Title                                                        | Abbreviation |    Venue    |    Paper    |     Code    |
| ---- | ------------------------------------------------------------ | :----------: | :---------: | :---------: | :---------: |
| 2025 | **Multi-View Graph Clustering via Node-Guided Contrastive Encoding** | **NGCE** | ICML | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=Ae5qnQxAxQ) |[![](https://img.shields.io/badge/-code-red)](https://github.com/Rirayh/NGCE)|
| 2025 | **Automatically Identify and Rectify: Robust Deep Contrastive Multi-view Clustering in Noisy Scenarios** | **AIRMVC** | ICML | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=iFOXz5H2gB) |[![](https://img.shields.io/badge/-code-red)](https://github.com/xihongyang1999/AIRMVC)|
| 2025 | **PROTOCOL: Partial Optimal Transport-enhanced Contrastive Learning for Imbalanced Multi-view Clustering** | **PROTOCOL** | ICML | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=Pm8LUCx6Mb) |[![](https://img.shields.io/badge/-code-red)](https://github.com/Scarlett125/PROTOCOL)|
| 2025 | **A Hubness Perspective on Representation Learning for Graph-Based Multi-View Clustering** | **hubREP** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_A_Hubness_Perspective_on_Representation_Learning_for_Graph-Based_Multi-View_Clustering_CVPR_2025_paper.pdf) |[![](https://img.shields.io/badge/-code-red)](https://github.com/zmxu196/hubREP)|
| 2025 | **Deep Fair Multi-View Clustering with Attention KAN** | **DFMVC-AKAN** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Deep_Fair_Multi-View_Clustering_with_Attention_KAN_CVPR_2025_paper.pdf) | - |
| 2025 | **EASEMVC:Efficient Dual Selection Mechanism for Deep Multi-View Clustering** | **EASEMVC** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_EASEMVCEfficient_Dual_Selection_Mechanism_for_Deep_Multi-View_Clustering_CVPR_2025_paper.pdf) | - |
| 2025 | **Enhanced then Progressive Fusion with View Graph for Multi-View Clustering** | **EPFMVC** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2025/papers/Dong_Enhanced_then_Progressive_Fusion_with_View_Graph_for_Multi-View_Clustering_CVPR_2025_paper.pdf) | - |
| 2025 | **Medusa: A Multi-Scale High-order Contrastive Dual-Diffusion Approach for Multi-View Clustering** | **Medusa** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Medusa_A_Multi-Scale_High-order_Contrastive_Dual-Diffusion_Approach_for_Multi-View_Clustering_CVPR_2025_paper.pdf) | - |
| 2025 | **ROLL: Robust Noisy Pseudo-label Learning for Multi-View Clustering with Noisy Correspondence** | **ROLL** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_ROLL_Robust_Noisy_Pseudo-label_Learning_for_Multi-View_Clustering_with_Noisy_CVPR_2025_paper.pdf) |[![](https://img.shields.io/badge/-code-red)](https://github.com/sunyuan-cs/2025-CVPR-ROLL)|
| 2025 | **Deep Multi-View Contrastive Clustering via Graph Structure Awareness** | **DMvCGSA** | TIP | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/11021328/) | - |
| 2025 | **DFL-Net: Disentangled Feature Learning Network for Multi-view Clustering** | **DFL-Net** | TKDE | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/11034651/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/chenzhe207/DFL-NET)|
| 2025 | **Pseudo-Label Guided Bidirectional Discriminative Deep Multi-View Subspace Clustering** | **PBDMSC** | TKDE | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10971281/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/usualheart/PBDMSC)|
| 2025 | **Multigranularity Information Fused Contrastive Learning With Multiview Clustering** | **MGCMVC** | TNNLS | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/11030281/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/Luyangabc/MGCMVC)|
| 2025 | **Multilevel Reliable Guidance for Unpaired Multiview Clustering** | **MRG-UMC** | TNNLS | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2407.01247) |[![](https://img.shields.io/badge/-code-red)](https://github.com/LikeXin94/MRG-UMC)|
| 2025 | **Learn from Global Rather Than Local: Consistent Context-Aware Representation Learning for Multi-View Graph Clustering** | **CCARL** | IJCAI | [![](https://img.shields.io/badge/-paper-blue)]([https://arxiv.org/pdf/2407.01247](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/3575.pdf)) | - |
| 2025 | **Graph Embedded Contrastive Learning for Multi-View Clustering** | **GMVC** | IJCAI | [![](https://img.shields.io/badge/-paper-blue)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/2286.pdf) |[![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/GMVC)|
| 2025 | **MASTER: A Multi-granularity Invariant Structure Clustering Scheme for Multi-view Clustering** | **MASTER** | IJCAI | [![](https://img.shields.io/badge/-paper-blue)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/4274.pdf) | - |
| 2025 | **Efficient Multi-view Clustering via Reinforcement Contrastive Learning** | **EMVC-RCL** | IJCAI | [![](https://img.shields.io/badge/-paper-blue)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/5103.pdf) | - |
| 2025 | **COPER: Correlation-based Permutations for Multi-View Clustering** | **COPER** | ICLR | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=5ZEbpBYGwH) |[![](https://img.shields.io/badge/-code-red)](https://github.com/LindenbaumLab/COPER)|
| 2025 | **Self-supervised Trusted Contrastive Multi-view Clustering with Uncertainty Refined** | **STCMC-UR** | AAAI | [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/33902/36057) |[![](https://img.shields.io/badge/-code-red)](https://github.com/ShizheHu)|
| 2025 | **Mixture of Experts as Representation Learner for Deep Multi-View Clustering** | **DMVC-CE** | AAAI | [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/download/34430/36585) | - |
| 2025 | **Structure-Adaptive Multi-View Graph Clustering for Remote Sensing Data** | **SAMVGC** | AAAI | [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/33861/36016) | - |
| 2025 | **Multi-aspect Self-guided Deep Information Bottleneck for Multi-modal Clustering** | **MSDIB** | AAAI | [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/33903/36058) |[![](https://img.shields.io/badge/-code-red)](https://github.com/ShizheHu/AAAI25_Code_MSDIB)|
| 2025 | **Multi-view Granular-ball Contrastive Clustering** | **MGBCC** | AAAI | [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/34274) |[![](https://img.shields.io/badge/-code-red)](https://github.com/Duo-laimi/mgbcc_main)|
| 2025 | **Selective Contrastive Learning for Unpaired Multi-View Clustering** | **scl-UMC** | TNNLS | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10327758/) | - |
| 2025 | **EMLFCL: An Efficient Multilevel Fusion Contrastive Learning for Multiview Clustering** | **EMLFCL** | TNNLS | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10949655/) | - |
| 2025 | **Anchor-Sharing and Cluster-Wise Contrastive Network for Multiview Representation Learning** | **CwCL** | TNNLS | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10430449/) | - |
| 2025 | **Multilevel Contrastive Multiview Clustering With Dual Self-Supervised Learning** | **MCMC** | TNNLS | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10963906/) | - |
| 2025 | **Variational Graph Generator for Multiview Graph Clustering** | **VGMGC** | TNNLS | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10833915/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/cjpcool/VGMGC)|
| 2025 | **Dynamic Graph Guided Progressive Partial View-Aligned Clustering** | **DGPPVC** | TNNLS | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10606529/) | - |
| 2025 | **Deep Discriminative Multi-view Clustering** | **DDMvC** | TCSVT | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10884978/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/chenzhe207/DDMvC)|
| 2025 | **Graph Variational Multi-view Clustering** | **GVMVC** | TCSVT | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/11006709/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/WenB777/GVMVC)|
| 2025 | **A Novel Approach for Effective Partially View-Aligned Clustering with Triple-Consistency** | **TCLPVC** | TCSVT | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/11005507/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/kongyiH/TCLPVC)|
| 2025 | **Learnable Graph Guided Deep Multi-view Representation Learning via Information Bottleneck** | **LGG-DMRL** | TCSVT | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10772242/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/AlanWang2000/LGG-DMRL)|
| 2025 | **Self-supervised Semantic Soft Label Learning Network for Deep Multi-view Clustering** | **SSLNMVC** | TMM | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10891409/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/shayeyty/SSLNMVC)|
| 2025 | **Learning Uniform Latent Representation via Alternating Adversarial Network for Multi-View Clustering** | **Deep-A2MC** | TETCI | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10909256/) | - |
| 2025 | **Multi-layer Multi-level Comprehensive Learning for Deep Multi-view Clustering** | **3MC** | IF | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S1566253524005633) |[![](https://img.shields.io/badge/-code-red)](https://github.com/chenzhe207/3MC)|
| 2025 | **Interpretable Multi-view Clustering** | **-** | PR | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0031320325000780) | - |
| 2025 | **Deep Multi-view Clustering with Diverse and Discriminative Feature Learning** | **DDMVC** | PR | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0031320324010732) |[![](https://img.shields.io/badge/-code-red)](https://github.com/xujunpeng832/DDMVC)|
| 2024 | **Adversarially Robust Deep Multi-View Clustering: A Novel Attack and Defense Framework** | **AR-DMVC-AM** | ICML | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=D9EfAkQCzh) |[![](https://img.shields.io/badge/-code-red)](https://github.com/libertyhhn/AR-DMVC)|
| 2024 | **Bridging Gaps: Federated Multi-View Clustering in Heterogeneous Hybrid Views** | **FMCSC** | NeurIPS | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2410.09484) |[![](https://img.shields.io/badge/-code-red)](https://github.com/5Martina5/FMCSC)|
| 2024 | **Robust Contrastive Multi-view Clustering against Dual Noisy Correspondence** | **CANDY** | NeurIPS | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=6OvTbDClUn) |[![](https://img.shields.io/badge/-code-red)](https://github.com/XLearning-SCU/2024-NeurIPS-CANDY)|
| 2024 | **Evaluate then Cooperate: Shapley-based View Cooperation Enhancement for Multi-view Clustering** | **SCE-MVC** | NeurIPS | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=xoc4QOvbDs) | - |
| 2024 | **Investigating and Mitigating the Side Effects of Noisy Views for Self-Supervised Clustering Algorithms in Practical Multi-View Scenarios** | **MVCAN** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Investigating_and_Mitigating_the_Side_Effects_of_Noisy_Views_for_CVPR_2024_paper.pdf) |[![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/MVCAN)|
| 2024 | **Rethinking Multi-view Representation Learning via Distilled Disentangling** | **MRDD** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/abs/2403.10897) |[![](https://img.shields.io/badge/-code-red)](https://github.com/Guanzhou-Ke/MRDD)|
| 2024 | **Differentiable Information Bottleneck for Deterministic Multi-view Clustering** | **DIB** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/abs/2403.15681) | - |
| 2024 | **Deep Generative Clustering with Multimodal Diffusion Variational Autoencoders** | **CMVAE** | ICLR | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=k5THrhXDV3) | - |
| 2024 | **Learning Common Semantics via Optimal Transport for Contrastive Multi-view Clustering** | **CSOT** | TIP | [![](https://img.shields.io/badge/-paper-blue)](https://research.edgehill.ac.uk/files/93038512/TIP_MVC_Final.pdf) |[![](https://img.shields.io/badge/-code-red)](https://github.com/vsislab/CSOT_MVC)|
| 2024 | **Dual Contrast-Driven Deep Multi-View Clustering** | **DCMVC** | TIP | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10648641/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/tweety1028/DCMVC)|
| 2024 | **Multiview Deep Subspace Clustering Networks** | **MvDSCN** | TCYB | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10478097/) |-|
| 2024 | **Deep Contrastive Multi-View Subspace Clustering With Representation and Cluster Interactive Learning** | **DCMVSC** | TKDE | [![](https://img.shields.io/badge/-paper-blue)](https://www.computer.org/csdl/journal/tk/5555/01/10726614/21dMsXWBxyE) | [![](https://img.shields.io/badge/-code-red)](https://github.com/YUKI-HIT/DCMVSC) |
| 2024 | **Robust Multi-View Clustering with Noisy Correspondence** | **RMCNC** | TKDE | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10595464/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/sunyuan-cs/2024-TKDE-RMCNC)|
| 2024 | **Integrating Vision-Language Semantic Graphs in Multi-View Clustering** | **IVSGMV** | IJCAI | [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/proceedings/2024/0472.pdf) | - |
| 2024 | **Simple Contrastive Multi-View Clustering with Data-Level Fusion** | **SCM** | IJCAI | [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/proceedings/2024/0519.pdf) |[![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/SCM)|
| 2024 | **Dynamic Weighted Graph Fusion for Deep Multi-View Clustering** | **DFMVC** | IJCAI | [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/proceedings/2024/0535.pdf) | - |
| 2024 | **Contrastive and View-Interaction Structure Learning for Multi-view Clustering** | **SERIES** | IJCAI | [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/proceedings/2024/0559.pdf) | - |
| 2024 | **Active Deep Multi-view Clustering** | **ADMC** | IJCAI | [![](https://img.shields.io/badge/-paper-blue)](https://doctor-nobody.github.io/papers/IJCAI2024.pdf) |[![](https://img.shields.io/badge/-code-red)](https://github.com/wodedazhuozi/ADMC)|
| 2024 | **Homophily-Related: Adaptive Hybrid Graph Filter for Multi-View Graph Clustering** | **AHGFC** | AAAI | [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/29514/30852) | - |
| 2024 | **SURER: Structure-Adaptive Unified Graph Neural Network for Multi-View Clustering** | **SURER** | AAAI | [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/29478/30785) | [![](https://img.shields.io/badge/-code-red)](https://github.com/Wjing-bjtu/SURER) |
| 2024 | **Graph based Consistency Learning for Contrastive Multi-View Clustering** | **GC-CMVC** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=AzwkOHpgqA) | - |
| 2024 | **Heterogeneity-Aware Federated Deep Multi-View Clustering towards Diverse Feature Representations** | **HFMVC** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=3fgY4qOhoO) | [![](https://img.shields.io/badge/-code-red)](https://github.com/xiaorui-jiang/HFMVC) |
| 2024 | **EMVCC: Enhanced Multi-View Contrastive Clustering for Hyperspectral Images** | **EMVCC** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=Twe5GWM0Hl) | [![](https://img.shields.io/badge/-code-red)](https://github.com/YiLiu1999/EMVCC) |
| 2024 | **DFMVC: Deep Fair Multi-view Clustering** | **DFMVC** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=IgQ3xrYXTc) | - |
| 2024 | **View Gap Matters: Cross-view Topology and Information Decoupling for Multi-view Clustering** | **TGM-MVC** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=DgNJ18FQLe) | - |
| 2024 | **Learning Dual Enhanced Representation for Contrastive Multi-view Clustering** | **LUCE-CMC** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=8uTMi4dQFK) | [![](https://img.shields.io/badge/-code-red)](https://github.com/ShizheHu/ACMMM24_Code_LUCE-CMC) |
| 2024 | **Contrastive Graph Distribution Alignment for Partially View-Aligned Clustering** | **CGDA** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=QoBHdPW1pM) | - |
| 2024 | **Dual-Optimized Adaptive Graph Reconstruction for Multi-View Graph Clustering** | **DOAGC** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2410.22983) | - |
| 2024 | **Robust Variational Contrastive Learning for Partially View-unaligned Clustering** | **VITAL** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=eZpm234cw2) | [![](https://img.shields.io/badge/-code-red)](https://github.com/He-Changhao/2024-MM-VITAL) |
| 2024 | **Self-Weighted Contrastive Fusion for Deep Multi-View Clustering** | **SCMVC** | TMM | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10499831/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/SongwuJob/SCMVC) |
| 2024 | **Subspace-Contrastive Multi-View Clustering** | **SCMC** | TKDD | [![](https://img.shields.io/badge/-paper-blue)](https://dl.acm.org/doi/abs/10.1145/3674839) | - |
| 2024 | **Multi-view contrastive clustering via integrating graph aggregation and confidence enhancement** | **MAGA** | IF  | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S1566253524001714) |[![](https://img.shields.io/badge/-code-red)](https://github.com/BJT-bjt/MAGA)|
| 2024 | **Trustworthy multi-view clustering via alternating generative adversarial representation learning and fusion** | **AGARL** | IF  | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S1566253524001015) | - |
| 2024 | **Structural deep multi-view clustering with integrated abstraction and detail** | **SMVC** | NN | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0893608024002119) | - |
| 2024 | **Progressive Neighbor-masked Contrastive Learning for Fusion-style Deep Multi-View Clustering** | **PNCL-FDMC** | NN | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0893608024004271?dgcid=rss_sd_all) | - |
| 2024 | **Composite Attention Mechanism Network for Deep Contrastive Multi-view Clustering** | **CAMVC** | NN | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0893608024002855) | - |
| 2024 | **Asymmetric Double-Winged Multi-View Clustering Network for Exploring Diverse and Consistent Information** | **CodingNet** | NN | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0893608024004878) | - |
| 2024 | **Decomposed deep multi-view subspace clustering with self-labeling supervision** | **D2MVSC** | IS | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S002002552301383X) | - |
| 2024 | **Structure-guided feature and cluster contrastive learning for multi-view clustering** | **SGFCC** | Neurcom | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0925231224003266) | - |
| 2024 | **Learning consensus representations in multi-latent spaces for multi-view clustering** | **DMCC** | Neurcom | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0925231224006702) | - |
| 2024 | **MCoCo: Multi-level Consistency Collaborative Multi-view Clustering** | **MCoCo** | ESA | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2302.13339) | - |
| 2024 | **Graph-Driven Deep Multi-View Clustering with Self-Paced Learning** | **GDMVC** | KBS | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0950705124005057) |[![](https://img.shields.io/badge/-code-red)](https://github.com/yff-java/GDMVC/)|
| 2024 | **Information Bottleneck Fusion for Deep Multi-view Clustering** | **IBFDMVC** | KBS | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0950705124001862) | - |
| 2024 | **Separable Consistency and Diversity Feature Learning for Multi-View Clustering** | **SCDFL** | SPL | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10545549/) | - |
| 2023 | **Graph Embedding Contrastive Multi-Modal Representation Learning for Clustering** | **GECMC** | TIP | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10036442/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/xdweixia/GECMC)|
| 2023 | **Neighbor-aware deep multi-view clustering via graph convolutional network** | **NMvC-GCN** | IF | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S1566253523000015) |[![](https://img.shields.io/badge/-code-red)](https://github.com/dugzzuli/NMvC-GCN)|
| 2023 | **Joint contrastive triple-learning for deep multi-view clustering** | **JCT** | IPM | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0306457323000213) |[![](https://img.shields.io/badge/-code-red)](https://github.com/ShizheHu/Joint-Contrastive-Triplelearning)|
| 2023 | **Auto-attention mechanism for multi-view deep embedding clustering** | **MDEC** | PR | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0031320323004624) | - |
| 2023 | **Deep multi-view spectral clustering via ensemble** | **DMCE** | PR | [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0031320323005344) | - |
| 2023 | **Unified Representation Learning for Multi-View Clustering by Between/Within View Deep Majorization** | **deepURL** | TETCI | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10261281/) |[![](https://img.shields.io/badge/-code-red)](https://github.com/PureRRR/deepURL)|
| 2023 | **Dropping pathways towards deep multi-view graph subspace clustering networks** | **DPMGSC** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://dl.acm.org/doi/10.1145/3581783.3612332) | - |
| 2023 | **Triple-granularity contrastive learning for deep multi-view subspace clustering** | **TRUST** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://dl.acm.org/doi/10.1145/3581783.3611844) | - |
| 2023 | **Deep multiview adaptive clustering with semantic invariance** | **DMAC-SI** | TNNLS | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10115048) |[![](https://img.shields.io/badge/-code-red)](https://github.com/TriMates/DMAC-SI)|
| 2023 | **Generalized Information-theoretic Multi-view Clustering** | **IMC** | NeurIPS | [![](https://img.shields.io/badge/-paper-blue)](https://proceedings.neurips.cc/paper_files/paper/2023/file/b7aa34d2d24f9bab3056993b7bfa0f1b-Paper-Conference.pdf) | - |
| 2023 | **Self-Weighted Contrastive Learning among Multiple Views for Mitigating Representation Degeneration** | **SEM** | NeurIPS | [![](https://img.shields.io/badge/-paper-blue)]([https://proceedings.neurips.cc/paper_files/paper/2023/file/8c64bc3f7796d31caa7c3e6b969bf7da-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/03b13b0db740b95cb741e007178ef5e5-Paper-Conference.pdf)) | [![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/SEM/archive/refs/heads/main.zip) |
| 2023 | **A Novel Approach for Effective Multi-View Clustering with Information-Theoretic Perspective** | **SUMVC** | NeurIPS | [![](https://img.shields.io/badge/-paper-blue)](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c64bc3f7796d31caa7c3e6b969bf7da-Paper-Conference.pdf) | [![](https://img.shields.io/badge/-code-red)](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c64bc3f7796d31caa7c3e6b969bf7da-Supplemental-Conference.zip) |
| 2023 | **Dual Label-Guided Graph Refnement for Multi-View Graph Clustering** | **DuaLGR** | AAAI | [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/26057/25829) | [![](https://img.shields.io/badge/-code-red)](https://github.com/YwL-zhufeng/DuaLGR/archive/refs/heads/main.zip) |
| 2023 | **Cross-view Topology Based Consistent and Complementary Information for Deep Multi-view Clustering** | **CTCC** | ICCV | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/ICCV2023/papers/Dong_Cross-view_Topology_Based_Consistent_and_Complementary_Information_for_Deep_Multi-view_ICCV_2023_paper.pdf) | - |
| 2023 | **MHCN: A Hyperbolic Neural Network Model for Multi-view Hierarchical Clustering** | **MHCN** | ICCV | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/ICCV2023/papers/Lin_MHCN_A_Hyperbolic_Neural_Network_Model_for_Multi-view_Hierarchical_Clustering_ICCV_2023_paper.pdf) | - |
| 2023 | **Deep Multiview Clustering by Contrasting Cluster Assignments** | **CVCL** | ICCV | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2304.10769) | [![](https://img.shields.io/badge/-code-red)](https://github.com/chenjie20/CVCL/) |
| 2023 | **DealMVC: Dual Contrastive Calibration for Multi-view Clustering** | **DealMVC** | ACM MM | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2308.09000) | [![](https://img.shields.io/badge/-code-red)](https://github.com/xihongyang1999/DealMVC) |
| 2023 | **Self-Supervised Graph Attention Networks for Deep Weighted Multi-View Clustering** | **SGDMC** | AAAI | [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/25960/25732) | - |
| 2023 | **Dual Fusion-Propagation Graph Neural Network for Multi-view Clustering** | **DFP-GNN** | TMM | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10050836/) | - |
| 2023 | **Joint Shared-and-Specific Information for Deep Multi-View Clustering** | **JSSI** | TCSVT | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10130402) | - |
| 2023 | **On the Effects of Self-supervision and Contrastive Alignment in Deep Multi-view Clustering** | **DeepMVC** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](http://openaccess.thecvf.com/content/CVPR2023/papers/Trosten_On_the_Effects_of_Self-Supervision_and_Contrastive_Alignment_in_Deep_CVPR_2023_paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/DanielTrosten/DeepMVC) |
| 2023 | **GCFAgg：Global and Cross-view Feature Aggregation for Multi-view Clustering** | **GCFAgg** | CVPR | [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_GCFAgg_Global_and_Cross-View_Feature_Aggregation_for_Multi-View_Clustering_CVPR_2023_paper.pdf) |  -  |
| 2023 | **Self-Supervised Information Bottleneck for Deep Multi-View Subspace Clustering** | **SIB-MSC** | TIP   | [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2204.12496.pdf) |  -  |
| 2023 | **Multi-channel Augmented Graph Embedding Convolutional Network for Multi-view Clustering** |    **MAGEC-Net**    | TNSE |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10043740/) | - |
| 2022 | **Deep Safe Multi-View Clustering：Reducing the Risk of Clustering Performance Degradation Caused by View Increase** | **DSMVC** | CVPR |  [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Deep_Safe_Multi-View_Clustering_Reducing_the_Risk_of_Clustering_Performance_CVPR_2022_paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/Gasteinh/DSMVC) |
| 2022 | **Multi-level Feature Learning for Contrastive Multi-view Clustering** | **MFLVC** | CVPR |  [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Multi-Level_Feature_Learning_for_Contrastive_Multi-View_Clustering_CVPR_2022_paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/MFLVC) |
| 2022 | **Stationary Diffusion State Neural Estimation for Multiview Clustering** | **SDSNE** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://www.aaai.org/AAAI22Papers/AAAI-184.LiuC.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/kunzhan/SDSNE) |
| 2022 | **Multi-View Subspace Clustering via Structured Multi-Pathway Network** | **SMpNet** | TNNLS |  [![](https://img.shields.io/badge/-paper-blue)](http://cic.tju.edu.cn/faculty/huqinghua/pdf/GeneralizedLatentMulti-ViewSubspaceClustering.pdf) | [![](https://img.shields.io/badge/-code-red)](http://cic.tju.edu.cn/faculty/zhangchangqing/code.html) |
| 2022 | **Multiview Subspace Clustering With Multilevel Representations and Adversarial Regularization** | **MvSC-MRAR** | TNNLS |  [![](https://img.shields.io/badge/-paper-blue)](https://www.researchgate.net/profile/Guowang-Du/publication/360244524_Multiview_Subspace_Clustering_With_Multilevel_Representations_and_Adversarial_Regularization/links/6291812e8d19206823e10ac7/Multiview-Subspace-Clustering-With-Multilevel-Representations-and-Adversarial-Regularization.pdf) | - |
| 2022 | **Self-Supervised Deep Multiview Spectral Clustering** | **SDMvSC** | TNNLS |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9853217/) | - |
| 2022 | **Contrastive Multi-view Hyperbolic Hierarchical Clustering** | **CMHHC** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2205.02618.pdf) | - |
| 2022 | **Multi-view Graph Embedding Clustering Network：Joint Self-supervision and Block Diagonal Representation** | **MVGC** | NN |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S089360802100397X) | [![](https://img.shields.io/badge/-code-red)](https://github.com/xdweixia/NN-2022-MVGC) |
| 2022 | **Efficient Multi‑view Clustering Networks** | **EMC-Nets** | APPL INTELL |  [![](https://img.shields.io/badge/-paper-blue)](https://link.springer.com/article/10.1007/s10489-021-03129-0) | [![](https://img.shields.io/badge/-code-red)](https://github.com/Guanzhou-Ke/EMC-Nets) |
| 2021 | **Deep Mutual Information Maximin for Cross-Modal Clustering** | **DMIM** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/17076/16883) | - |
| 2021 | **Uncertainty-Aware Multi-View Representation Learning** | **DUA-Nets** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/16924/16731) | [![](https://img.shields.io/badge/-code-red)](http://cic.tju.edu.cn/faculty/zhangchangqing/code/daunet.zip) |
| 2021 | **Learning Deep Sparse Regularizers With Applications to Multi-View Clustering and Semi-Supervised Classification** | **DSRL** | TPAMI |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9439159/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/chenzl23/DSRL) |
| 2021 | **Reconsidering Representation Alignment for Multi-view Clustering** | **SiMVC&CoMVC** | CVPR |  [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2021/papers/Trosten_Reconsidering_Representation_Alignment_for_Multi-View_Clustering_CVPR_2021_paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/AllenWrong/mvc) |
| 2021 | **Deep Multiple Auto-Encoder-Based Multi-view Clustering** | **MVC_MAE** | DSE |  [![](https://img.shields.io/badge/-paper-blue)](https://link.springer.com/article/10.1007/s41019-021-00159-z) | [![](https://img.shields.io/badge/-code-red)](https://github.com/dugzzuli/Deep-Multiple-Auto-Encoder-Based-Multi-view-Clustering) |
| 2021 | **Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos** | **MCN** | ICCV |  [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Multimodal_Clustering_Networks_for_Self-Supervised_Learning_From_Unlabeled_Videos_ICCV_2021_paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/brian7685/Multimodal-Clustering-Network) |
| 2021 | **Multi-VAE: Learning Disentangled View-common and View-peculiar Visual Representations for Multi-view Clustering** | **Multi-VAE** | ICCV |  [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Multi-VAE_Learning_Disentangled_View-Common_and_View-Peculiar_Visual_Representations_for_Multi-View_ICCV_2021_paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/Multi-VAE) |
| 2021 | **Graph Filter-based Multi-view Attributed Graph Clustering** | **MvAGC** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/proceedings/2021/0375.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/sckangz/MvAGC) |
| 2021 | **Multi-view Subspace Clustering Networks with Local and Global Graph Information** | **MSCNGL** | Neurocom |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.53yu.com/pdf/2010.09323) | [![](https://img.shields.io/badge/-code-red)](https://github.com/qinghai-zheng/MSCNLG) |
| 2021 | **Attentive Multi-View Deep Subspace Clustering Net** | **AMVDSN** | Neurocom |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2112.12506.pdf) | - |
| 2021 | **Multi-view Contrastive Graph Clustering** | **MCGC** | NeurIPS |  [![](https://img.shields.io/badge/-paper-blue)](https://proceedings.neurips.cc/paper/2021/file/10c66082c124f8afe3df4886f5e516e0-Paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/panern/mcgc) |
| 2021 | **Self-supervised Discriminative Feature Learning for Deep Multi-view Clustering** | **SDMVC** | TKDE |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2103.15069.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/SDMVC) |
| 2021 | **Multi-view Attributed Graph Clustering** | **MAGC** | TKDE |  [![](https://img.shields.io/badge/-paper-blue)](https://www.researchgate.net/publication/353747180_Multi-view_Attributed_Graph_Clustering) | [![](https://img.shields.io/badge/-code-red)](https://github.com/sckangz/MAGC) |
| 2021 | **Deep Multi-view Subspace Clustering with Unified and Discriminative Learning** | **DMSC-UDL** | TMM |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9204408/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/IMKBLE/DMSC-UDL) |
| 2021 | **Self-supervised Graph Convolutional Network for Multi-view Clustering** | **SGCMC** | TMM |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9472979/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/xdweixia/SGCMC) |
| 2021 | **Consistent Multiple Graph Embedding for Multi-View Clustering** | **CMGEC** | TMM |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2105.04880) | [![](https://img.shields.io/badge/-code-red)](https://github.com/wangemm/CMGEC-TMM-2021) |
| 2021 | **Deep Multiview Collaborative Clustering** | **DMCC** | TNNLS |  [![](https://img.shields.io/badge/-paper-blue)](https://see.xidian.edu.cn/faculty/chdeng/Welcome%20to%20Cheng%20Deng's%20Homepage_files/Papers/Journal/TNNLS2021_Xu.pdf) | - |
| 2020 | **Partially View-aligned Clustering** | **PVC** | NeurIPS |  [![](https://img.shields.io/badge/-paper-blue)](https://proceedings.neurips.cc/paper/2020/file/1e591403ff232de0f0f139ac51d99295-Paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/hi-zhenyu/PVC) |
| 2020 | **Cross-modal Subspace Clustering via Deep Canonical Correlation Analysis** | **CMSC-DCCA** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/5808/5664) | - |
| 2020 | **Shared Generative Latent Representation Learning for Multi-View Clustering** | **DMVCVAE** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/download/6146/6002) | [![](https://img.shields.io/badge/-code-red)](https://github.com/whytin95/DMVCVAE) |
| 2020 | **End-to-End Adversarial-Attention Network for Multi-Modal Clustering** | **EAMC** | CVPR |  [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_End-to-End_Adversarial-Attention_Network_for_Multi-Modal_Clustering_CVPR_2020_paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/AllenWrong/mvc) |
| 2020 | **Multi-View Attribute Graph Convolution Networks for Clustering** | **MAGCN** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/proceedings/2020/0411.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/IMKBLE/MAGCN) |
| 2020 | **End-To-End Deep Multimodal Clustering** | **DMMC** | ICME |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9102921/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/Guanzhou-Ke/DMMC-zoo) |
| 2020 | **Deep Embedded Multi-view Clustering with Collaborative Training** | **DEMVC** | IS |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2007.13067.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/DEMVC) |
| 2020 | **Joint Deep Multi-View Learning for Image Clustering** | **DMJC** | TKDE |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/8999493/) | - |
| 2020 | **One2Multi Graph Autoencoder for Multi-view Graph Clustering** | **O2MVC** | WWW |  [![](https://img.shields.io/badge/-paper-blue)](http://shichuan.org/doc/83.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/googlebaba/WWW2020-O2MAC) |
| 2019 | **AE^2-Nets: Autoencoder in Autoencoder Networks** | **AE^2-Nets** | CVPR |  [![](https://img.shields.io/badge/-paper-blue)](http://cic.tju.edu.cn/faculty/zhangchangqing/pub/AE2_Nets.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/willow617/AE2-Nets) |
| 2019 | **COMIC: Multi-view Clustering Without Parameter Selection** | **COMIC** | ICML |  [![](https://img.shields.io/badge/-paper-blue)](http://proceedings.mlr.press/v97/peng19a/peng19a.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/limit-scu/2019-ICML-COMIC) |
| 2019 | **Deep Adversarial Multi-view Clustering Network** | **DAMC** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://www.researchgate.net/publication/334844473_Deep_Adversarial_Multi-view_Clustering_Network) | [![](https://img.shields.io/badge/-code-red)](https://github.com/IMKBLE/DAMC) |
| 2019 | **Multi-view Spectral Clustering Network** | **MvSCN** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/Proceedings/2019/0356.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/limit-scu/2019-IJCAI-MvSCN) |
| 2019 | **Multi-view Deep Subspace Clustering Networks** | **MvDSCN** | TIP |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/abs/1908.01978) | [![](https://img.shields.io/badge/-code-red)](https://github.com/huybery/MvDSCN) |
| 2018 | **Generalized Latent Multi-View Subspace Clustering** | **gLMSC** | TPAMI |  [![](https://img.shields.io/badge/-paper-blue)](http://cic.tju.edu.cn/faculty/huqinghua/pdf/GeneralizedLatentMulti-ViewSubspaceClustering.pdf) | [![](https://img.shields.io/badge/-code-red)](http://cic.tju.edu.cn/faculty/zhangchangqing/code.html) |
| 2018 | **Deep Multimodal Subspace Clustering Networks** | **DMSC** | STSP |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/1804.06498.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/mahdiabavisani/Deep-multimodal-subspace-clustering-networks) |
| 2018 | **Deep Multi-View Clustering via Multiple Embedding** | **DMVC-ME** | CoRR |  [![](https://img.shields.io/badge/-paper-blue)](https://deepai.org/publication/deep-multi-view-clustering-via-multiple-embedding) | - |


---

### <span id="jump32">Deep Incomplete Multi-view Clustering(DIMVC)</span> 

| Year | Title                                                        | Abbreviation |    Venue    |    Paper    |     Code    |
| ---- | ------------------------------------------------------------ | :----------: | :---------: | :---------: | :---------: |
| 2025 | **An Effective and Secure Federated Multi-View Clustering Method with Information-Theoretic Perspective** | **ESFMC** | ICML |  [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=eLkkXaPFEP) | [![](https://img.shields.io/badge/-code-red)](https://github.com/5Martina5/ESFMC) |
| 2025 | **Federated Incomplete Multi-view Clustering with Globally Fused Graph Guidance** | **FIMCFG** | ICML |  [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=7qvYLnJDRd) | [![](https://img.shields.io/badge/-code-red)](https://github.com/PaddiHunter/FIMCFG) |
| 2025 | **Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs** | **MVP** | ICLR |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2502.11037) | [![](https://img.shields.io/badge/-code-red)](https://openreview.net/forum?id=s4MwstmB8o) |
| 2025 | **Imputation-free and Alignment-free: Incomplete Multi-view Clustering Driven by Consensus Semantic Learning** | **FreeCSL** | CVPR |  [![](https://img.shields.io/badge/-paper-blue)](https://openaccess.thecvf.com/content/CVPR2025/papers/Dai_Imputation-free_and_Alignment-free_Incomplete_Multi-view_Clustering_Driven_by_Consensus_Semantic_CVPR_2025_paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/zoyadai/2025_CVPR_FreeCSL) |
| 2025 | **Selective Cross-view Topology for Deep Incomplete Multi-view Clustering** | **SCVT** | TIP | [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/11091516/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/dzboop/SCVT)|
| 2025 | **Robust Graph Contrastive Learning for Incomplete Multi-view Clustering** | **RGCL** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/1295.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/DYZ163/RGCL) |
| 2025 | **LRGR: Self-Supervised Incomplete Multi-View Clustering via Local Refinement and Global Realignment** | **LRGR** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/3180.pdf) | - |
| 2025 | **Fair Incomplete Multi-View Clustering via Distribution Alignment** | **FIMVC-DA** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/5109.pdf) | - |
| 2025 | **Dual Robust Unbiased Multi-View Clustering for Incomplete and Unpaired Information** | **DRUMVC** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/6021.pdf) | - |
| 2025 | **Imputation-free Incomplete Multi-view Clustering via Knowledge Distillation** | **I2MVC** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/7709.pdf) | - |
| 2025 | **Mask-informed Deep Contrastive Incomplete Multi-view Clustering** | **Mask-IMvC** | TCSVT |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2502.02234?) | [![](https://img.shields.io/badge/-code-red)](https://github.com/guanyuezhen/Mask-IMvC) |
| 2025 | **Incomplete and Unpaired Multi-View Graph Clustering with Cross-View Feature Fusion** | **MGCCFF** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/34439/36594) | [![](https://img.shields.io/badge/-code-red)](https://github.com/AlanWang2000/MGCCFF) |
| 2025 | **Incomplete Multi-view Clustering via Diffusion Contrastive Generation** | **DCG** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/download/34424/36579) | [![](https://img.shields.io/badge/-code-red)](https://github.com/zhangyuanyang21/2025-AAAI-DCG) |
| 2025 | **Global Graph Propagation with Hierarchical Information Transfer for Incomplete Contrastive Multi-view Clustering** | **GHICMC** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/33725/35880) | [![](https://img.shields.io/badge/-code-red)](https://github.com/KelvinXuu/GHICMC) |
| 2025 | **Prototype Matching Learning for Incomplete Multi-view Clustering** | **PMIMC** | TIP |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10847794/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/hl-yuan/PMIMC) |
| 2025 | **Incomplete Multi-View Clustering via Multi-Level Contrastive Learning** | **IMC-MCL** | TKDE |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/11005467/) | - |
| 2025 | **Neighbor-Based Completion for Addressing Incomplete Multiview Clustering** | **NBIMVC** | TNNLS |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10925219/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/WenB777/NBIMVC) |
| 2025 | **Deep Incomplete Multi-View Clustering via Dynamic Imputation and Triple Alignment With Dual Optimization** | **DITA-IMVC** | TCSVT |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/10771953/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/liukanglong/DITA-IMVC) |
| 2025 | **Incomplete Multi-view Clustering Based on Information Fusion with Self-supervised Learning** | **IMCFL** | IF |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S1566253524006274) | - | 
| 2025 | **Deep Incomplete Multi-view Clustering via Multi-level Imputation and Contrastive Alignment** | **MICA** | PR |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0893608024007755) | - |
| 2024 | **Explicit View-Labels Matter: A Multifacet Complementarity Study of Multi-View Clustering** | **MCMVC** | TPAMI |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10816579/) | - |
| 2024 | **Diffusion-based Missing-view Generation With the Application on Incomplete Multi-view Clustering** | **DMVG** | ICML |  [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=OHFxcU9jwW) | [![](https://img.shields.io/badge/-code-red)](https://github.com/Dshijie/DMVG) |
| 2024 | **Deep Variational Incomplete Multi-View Clustering: Exploring Shared Clustering Structures** | **DVIMC** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/29548/30915) | [![](https://img.shields.io/badge/-code-red)](https://github.com/ckghostwj/DVIMC-mindspore) |
| 2024 | **Partial Multi-View Clustering via Self-Supervised Network** | **PVC-SCN** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/download/29086/30054) | | [![](https://img.shields.io/badge/-code-red)](https://github.com/Dshijie/DMVG) |
| 2024 | **Incomplete Contrastive Multi-View Clustering with High-Confidence Guiding** | **ICMVC** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/download/29000/29899) | [![](https://img.shields.io/badge/-code-red)](https://github.com/hannaiiyanggit/MCMVC?tab=readme-ov-file#view-labels-are-important-a-multifacet-complementarity-study-of-deep-multi-view-clustering) |
| 2024 | **Adaptive Feature Imputation with Latent Graph for Deep Incomplete Multi-View Clustering** | **AGDIMC** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/download/29380/30606) | - |
| 2024 | **Decoupled Contrastive Multi-view Clustering with High-order Random Walks** | **DIVIDE** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/download/29330/30509) | [![](https://img.shields.io/badge/-code-red)](https://github.com/XLearning-SCU/2024-AAAI-DIVIDE) |
| 2024 | **Robust Prototype Completion for Incomplete Multi-view Clustering** | **RPCIC** | ACM MM |  [![](https://img.shields.io/badge/-paper-blue)](https://openreview.net/pdf?id=4BrIZo3Ave) | [![](https://img.shields.io/badge/-code-red)](https://github.com/hl-yuan/RPCIC) |
| 2024 | **URRL-IMVC: Unified and Robust Representation Learning for Incomplete Multi-View Clustering** | **URRL-IMVC** | SIGKDD |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2407.09120) | - |
| 2024 | **Subgraph Propagation and Contrastive Calibration for Incomplete Multiview Data Clustering** | **SPCC** | TNNLS |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10405858/) | - |
| 2024 | **Deep Incomplete Multiview Clustering via Local and Global Pseudo-Label Propagation** | **PLP-IMVC** | TNNLS |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10561887/) | - |
| 2024 | **Robust Multi-Graph Contrastive Network for Incomplete Multi-View Clustering** | **RMGC** | TMM |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10378851/) | - |
| 2024 | **A novel Federated Multi-view Clustering Method for Unaligned and Incomplete Data Fusion** | **FUCIF** | IF |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S1566253524001350) | [![](https://img.shields.io/badge/-code-red)](https://github.com/5Martina5/FCUIF) |
| 2024 | **Contrastive and Adversarial Regularized Multi-level Representation Learning for Incomplete Multi-view Clustering** | **MRL_CAL** | NN |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0893608024000169) | [![](https://img.shields.io/badge/-code-red)](https://github.com/xkmaxidian/MRL_CAL) |
| 2024 | **View-interactive Attention Information Alignment-guided Fusion for Incomplete Multi-view Clustering** | **VAIAF** | ESA |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0957417424011242) | - |
| 2024 | **Graph-Guided Imputation-Free Incomplete Multi-View Clustering** | **GIMVC** | ESA |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0957417424020323) | [![](https://img.shields.io/badge/-code-red)](https://github.com/yff-java/GIMVC/) |
| 2024 | **Deep Incomplete Multi-View Clustering via Attention-Based Direct Contrastive Learning** | **ADCL** | ESA |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0957417424016129) | - |
| 2024 | **Incomplete Multi-View Clustering via Diffusion Completion** | **IMVCDC** | MTA |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2305.11489) | - |
| 2024 | **Incomplete Multi-View Clustering Via Inference and Evaluation** | **IMVC-IE** | ICASSP |  [![](https://img.shields.io/badge/-paper-blue)](https://qinghai-zheng.github.io/file/ICASSP_2024.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/Yggdra-sil/IMVC-IE) |
| 2024 | **Incomplete Multi-view Clustering via Self-attention Networks and Feature Reconstruction** | **SNFR** | APPL INTELL |  [![](https://img.shields.io/badge/-paper-blue)](https://link.springer.com/article/10.1007/s10489-024-05299-z) | - |
| 2023 | **UNTIE: Clustering Analysis with Disentanglement in Multi-view Information Fusion** | **UNTIE** | IF |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S1566253523002531) | - |
| 2023 | **Federated Deep Multi-View Clustering with Global Self-Supervision** | **FedDMVC** | ACM MM |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2309.13697) | - |
| 2023 | **Incomplete Multi-view Clustering via Attention-based Contrast Learning** | **MCAC** | IJMLC |  [![](https://img.shields.io/badge/-paper-blue)](https://link.springer.com/article/10.1007/s13042-023-01883-w) | [![](https://img.shields.io/badge/-code-red)](https://github.com/123zyh123/Incomplete-multi-view-clustering-via-attention-based-contrast-learning) |
| 2023 | **Incomplete Multi-View Clustering With Complete View Guidance** | **IMC-CVG** | SPL |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10209192/) | - |
| 2023 | **Information Recovery-driven Deep Incomplete Multiview Clustering Network** | **RecFormer** | TNNLS |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/html/2304.00429v5) | [![](https://img.shields.io/badge/-code-red)](https://github.com/justsmart/Recformer-mindspore) |
| 2023 | **Realize Generative Yet Complete Latent Representation for Incomplete Multi-View Learning** | **CMVAE** | TPAMI |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/iel7/34/4359286/10373887.pdf) | - |
| 2023 | **Semantic Invariant Multi-View Clustering With Fully Incomplete Information** | **SMILE** | TPAMI |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10319403/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/PengxinZeng/2023-TPAMI-SMILE) |
| 2023 | **Deep Incomplete Multi-view Clustering with Cross-view Partial Sample and Prototype Alignment** | **CPSPAN** | CVPR |  [![](https://img.shields.io/badge/-paper-blue)](http://openaccess.thecvf.com/content/CVPR2023/papers/Jin_Deep_Incomplete_Multi-View_Clustering_With_Cross-View_Partial_Sample_and_Prototype_CVPR_2023_paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/jinjiaqi1998/CPSPAN) |
| 2023 | **Adaptive Feature Projection with Distribution Alignment for Deep Incomplete Multi-view Clustering** | **APADC** | TIP |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10043822/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/APADC) |
| 2023 | **Incomplete Multi-view Clustering via Prototype-based Imputation** | **ProImp** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2301.11045) | [![](https://img.shields.io/badge/-code-red)](https://github.com/XLearning-SCU/2023-IJCAI-ProImp) |
| 2023 | **Consistent Graph Embedding Network with Optimal Transport for Incomplete Multi-view Clustering** | **CGEN-OT** | IS |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0020025523010034) | - |
| 2023 | **CCR-Net: Consistent Contrastive Representation Network for Multi-view Clustering** | **CCR-Net** | IS |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0020025523005066) | [![](https://img.shields.io/badge/-code-red)](https://elki-project.github.io/datasets/multi_view) |
| 2023 | **Incomplete Multi-view Clustering Network via Nonlinear Manifold Embedding and Probability-Induced Loss** | **IMCNet-MP** | NN |  [![](https://img.shields.io/badge/-paper-blue)](https://www.sciencedirect.com/science/article/pii/S0893608023001302) | - |
| 2022 | **Robust Multi-view Clustering with Incomplete Information** | **SURE** | TPAMI |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9723577/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/XLearning-SCU/2022-TPAMI-SURE) |
| 2022 | **Dual Contrastive Prediction for Incomplete Multi-view Representation Learning** | **DCP** | TPAMI |  [![](https://img.shields.io/badge/-paper-blue)](http://pengxi.me/wp-content/uploads/2022/08/DCP.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/XLearning-SCU/2021-CVPR-Completer) |
| 2022 | **Deep Safe Incomplete Multi-view Clustering: Theorem and Algorithm** | **DSIMVC** | ICML |  [![](https://img.shields.io/badge/-paper-blue)](https://proceedings.mlr.press/v162/tang22c/tang22c.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/Gasteinh/DSIMVC) |
| 2022 | **Deep Incomplete Multi-view Clustering via Mining Cluster Complementarity** | **DIMVC** | AAAI |  [![](https://img.shields.io/badge/-paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/download/20856/20615) | [![](https://img.shields.io/badge/-code-red)](https://github.com/SubmissionsIn/DIMVC) |
| 2022 | **Robust Diversified Graph Contrastive Network for Incomplete Multi-view Clustering** | **RDGC** | ACM MM |  [![](https://img.shields.io/badge/-paper-blue)](https://dl.acm.org/doi/abs/10.1145/3503161.3547894) | [![](https://img.shields.io/badge/-code-red)](https://github.com/zh-hike/RDGC) |
| 2022 | **Incomplete Multi-view Clustering via Cross-view Relation Transfer** | **CRTC** | TCSVT |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2112.00739) | [![](https://img.shields.io/badge/-code-red)](https://github.com/wangemm/CRTC-TCSVT-2022) |   
| 2022 | **Graph Contrastive Partial Multi-view Clustering** | **AGCL** | TMM |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9904927/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/wangemm/AGCL-TMM-2022) |
| 2021 | **COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction** | **COMPLETER** | CVPR |  [![](https://img.shields.io/badge/-paper-blue)](http://pengxi.me/wp-content/uploads/2021/03/2021CVPR-completer.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/XLearning-SCU/2021-CVPR-Completer) |
| 2021 | **iCmSC: Incomplete Cross-modal Subspace Clustering** | **iCmSC** | TIP |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9259207) | [![](https://img.shields.io/badge/-code-red)](https://github.com/IMKBLE/iCmSC) |
| 2021 | **Generative Partial Multi-View Clustering With Adaptive Fusion and Cycle Consistency** | **GP-MVC** | TIP |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9318542/) | [![](https://img.shields.io/badge/-code-red)](https://github.com/IMKBLE/GP-MVC) |
| 2021 | **Clustering-Induced Adaptive Structure Enhancing Network for Incomplete Multi-View Data** | **CASEN** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/proceedings/2021/0445.pdf) | - |
| 2021 | **Structural Deep Incomplete Multi-view Clustering Network** | **SDIMC-net** | CIKM |  [![](https://img.shields.io/badge/-paper-blue)](https://dl.acm.org/doi/abs/10.1145/3459637.3482192) | - |
| 2021 | **Dual Alignment Self-Supervised Incomplete Multi-View Subspace Clustering Network** | **DASIMSC** | SPL |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/abstract/document/9573269/) | - |
| 2020 | **Deep Partial Multi-View Learning** | **DPML** | TPAMI |  [![](https://img.shields.io/badge/-paper-blue)](https://ieeexplore.ieee.org/document/9258396) | [![](https://img.shields.io/badge/-code-red)](http://cic.tju.edu.cn/faculty/zhangchangqing/code/DPML.zip) |
| 2020 | **CDIMC-net：Cognitive Deep Incomplete Multi-view Clustering Network(** | **CDIMC-net** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/proceedings/2020/447) | [![](https://img.shields.io/badge/-code-red)](https://github.com/DarrenZZhang/CDIMC-Net) |
| 2020 | **DIMC-net：Deep Incomplete Multi-view Clustering Network** | **DIMC-net** | ACM MM |  [![](https://img.shields.io/badge/-paper-blue)](https://dl.acm.org/doi/10.1145/3394171.3413807) | - |
| 2020 | **Deep Incomplete Multi-View Multiple Clusterings** | **DiMVMC** | ICDM |  [![](https://img.shields.io/badge/-paper-blue)](https://arxiv.org/pdf/2010.02024) | [![](https://img.shields.io/badge/-code-red)](http://www.sdu-idea.cn/codes.php?name=DiMVMC) |
| 2019 | **CPM-Nets: Cross Partial Multi-View Networks** | **CPM-Nets** | NeurIPS |  [![](https://img.shields.io/badge/-paper-blue)](https://papers.nips.cc/paper/2019/file/11b9842e0a271ff252c1903e7132cd68-Paper.pdf) | [![](https://img.shields.io/badge/-code-red)](https://github.com/hanmenghan/CPM_Nets) |
| 2019 | **Adversarial Incomplete Multi-view Clustering** | **AIMC** | IJCAI |  [![](https://img.shields.io/badge/-paper-blue)](https://www.ijcai.org/proceedings/2019/0546.pdf) | - |
| 2018 | **Partial Multi-View Clustering via Consistent GAN** | **PVC-GAN** | ICDM |  [![](https://img.shields.io/badge/-paper-blue)](https://drive.google.com/file/d/1RrVeq_FHkLSgltNd1bVfyaHhtIclV5ZG/view) | [![](https://img.shields.io/badge/-code-red)](https://github.com/IMKBLE/PVC-GAN) |

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

@inproceedings{wangevaluate,
  title={Evaluate then Cooperate: Shapley-based View Cooperation Enhancement for Multi-view Clustering},
  author={Wang, Fangdi and Jin, Jiaqi and Hu, Jingtao and Liu, Suyuan and Yang, Xihong and Wang, Siwei and Liu, Xinwang and Zhu, En},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}

@article{wang2022align,
  title={Align then fusion: Generalized large-scale multi-view clustering with anchor matching correspondences},
  author={Wang, Siwei and Liu, Xinwang and Liu, Suyuan and Jin, Jiaqi and Tu, Wenxuan and Zhu, Xinzhong and Zhu, En},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={5882--5895},
  year={2022}
}

@inproceedings{dong2023cross,
  title={Cross-view topology based consistent and complementary information for deep multi-view clustering},
  author={Dong, Zhibin and Wang, Siwei and Jin, Jiaqi and Liu, Xinwang and Zhu, En},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19440--19451},
  year={2023}
}

@inproceedings{yang2023dealmvc,
  title={Dealmvc: Dual contrastive calibration for multi-view clustering},
  author={Yang, Xihong and Jiaqi, Jin and Wang, Siwei and Liang, Ke and Liu, Yue and Wen, Yi and Liu, Suyuan and Zhou, Sihang and Liu, Xinwang and Zhu, En},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={337--346},
  year={2023}
}

@inproceedings{wang2024view,
  title={View Gap Matters: Cross-view Topology and Information Decoupling for Multi-view Clustering},
  author={Wang, Fangdi and Jin, Jiaqi and Dong, Zhibin and Yang, Xihong and Feng, Yu and Liu, Xinwang and Zhu, Xinzhong and Wang, Siwei and Liu, Tianrui and Zhu, En},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={8431--8440},
  year={2024}
}

@article{dong2024subgraph,
  title={Subgraph Propagation and Contrastive Calibration for Incomplete Multiview Data Clustering},
  author={Dong, Zhibin and Jin, Jiaqi and Xiao, Yuyang and Xiao, Bin and Wang, Siwei and Liu, Xinwang and Zhu, En},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}

@article{dong2023iterative,
  title={Iterative deep structural graph contrast clustering for multiview raw data},
  author={Dong, Zhibin and Jin, Jiaqi and Xiao, Yuyang and Wang, Siwei and Zhu, Xinzhong and Liu, Xinwang and Zhu, En},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
