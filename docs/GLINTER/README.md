Inter-protein (interfacial) contact prediction is very useful for in silico structural characterization of protein–protein interactions. Although deep learning has been applied to this problem, its accuracy is not as good as intra-protein contact prediction.

We propose a new deep learning method GLINTER (Graph Learning of INTER-protein contacts) for interfacial contact prediction of dimers, leveraging a rotational invariant representation of protein tertiary structures and a pretrained language model of multiple sequence alignments. Tested on the 13th and 14th CASP-CAPRI datasets, the average top L/10 precision achieved by GLINTER is 54% on the homodimers and 52% on all the dimers, much higher than 30% obtained by the latest deep learning method DeepHomo on the homodimers and 15% obtained by BIPSPI on all the dimers. Our experiments show that GLINTER-predicted contacts help improve selection of docking decoys.


> 蛋白质间（界面）接触预测对于蛋白质‑蛋白质相互作用的计算机模拟结构表征十分有用。尽管深度学习已被应用于该问题，但其精度不及蛋白质内部接触预测。
> 
> 我们提出了一种全新深度学习方法GLINTER（蛋白质间接触图学习），用于二聚体的界面接触预测。该方法利用蛋白质三级结构的旋转不变表征以及多序列比对预训练语言模型。在第13届和第14届CASP‑CAPRI数据集上开展测试，GLINTER取得的平均top L/10精度在同源二聚体上为54%，在全部二聚体上为52%，远高于最新深度学习方法DeepHomo在同源二聚体上的30%，以及BIPSPI在全部二聚体上的15%。实验表明，由GLINTER预测得到的接触有助于提升对接构象候选集的筛选效果。