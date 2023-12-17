# LIGAND<br><sup>Locus Inference and Generative Adversarial Network for gRNA Design</sup>

### Abstract

The advent of Clustered Regularly Interspaced Short Palindromic Repeats (CRISPR) technology, notably the CRISPR-Cas9 system—an RNA-guided DNA endonuclease-introduces an exciting era of precise gene editing. Now, a central problem becomes the design of guide RNA (gRNA), the sequence of RNA responsible for locating a bind location in the genome for the CRISPR-Cas9 protein. While existing tools can predict gRNA activity, only experimental or algorithmic methods are used to generate gRNA specific to DNA subsequences. In this study, we propose LIGAND, a model which leverages a generative adversarial network (GAN) and novel attention-based architectures to simultaneously address on- and off- target gRNA activity prediction and gRNA sequence generation. LIGAND’s generator produces a plurality of viable, highly precise, and effective gRNA sequences with a novel objective function consideration for off-site activity minimization, while the discriminator maintains state-of-the-art performance in gRNA activity prediction with any DNA and epigenomic prior. This dual functionality positions LIGAND as a versatile tool with applications spanning medicine and research.

### Results

LIGAND can sucessfully generate and discriminate biologically-validated activity for arbitrary DNA/epigenomic/gRNA pairings, including the consideration of offsite effects in both generation and discrimination. For particular gene knockout regions, we can design gRNA to specifically cleave in those regions while maintaining minimal off-site effects.

![](https://github.com/woody-hulse/LIGAND/assets/112116530/dc890317-76ed-4b16-9d35-9b39428d08c8)

*Average predicted activity over DNA region for validated gRNA*

![grna_region](https://github.com/woody-hulse/LIGAND/assets/112116530/628c1173-4578-4a8e-85dd-52d114c47fcd)

*Generation and top-5 evaluation of candidate gRNA for particular gene knockout regions*


---
