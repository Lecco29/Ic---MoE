# Evaluating a Deep Metric Model for Pattern Retrieval Based on Color and Texture Features

# Arlete Teresinha Beuren

*Department of Computer Science Federal University of Technology (UTFPR)*

Santa Helena (PR), Brazil [arletebeuren@utfpr.edu.br](mailto:arletebeuren@utfpr.edu.br)

# Vin´ıcius Cerqueira Ribeiro

*Department of Computer Science Federal University of Technology (UTFPR)*

Santa Helena (PR), Brazil [viniciuscerqueira@alunos.utfpr.edu.br](mailto:viniciuscerqueira@alunos.utfpr.edu.br)

# Leonardo Jose Reis Pinto

*Department of Computer Science Federal University of Technology (UTFPR)*

Santa Helena (PR), Brazil leonardopinto[@alunos.utfpr.edu.br](mailto:viniciuscerqueira@alunos.utfpr.edu.br)

# Alceu de Souza Britto Jr.

*Department of Computer Science Pontifical Catholic University of Parana´ (PUCPR)*  
Curitiba (PR), Brazil [alceu@ppgia.pucpr.br](mailto:alceu@ppgia.pucpr.br)

# Jose Saavedra

*dept. name of organization (of Aff.) Universidad de Los Andes*

City, Chile

email address or ORCID

# Jonathan xxxx

*Department of Computer Science Federal University of Technology (UTFPR)*

Santa Helena (PR), Brazil xxx[@alunos.utfpr.edu.br](mailto:viniciuscerqueira@alunos.utfpr.edu.br)

***Abstract*****—This study addresses the challenge of effectively extracting image content based on specific attributes: color, tex- ture, or their combination—an area of particular relevance to e- commerce applications. The deep metric learning model SimSiam is evaluated to understand the sensitivity of its layers to these attributes. Experiments were conducted to assess layer-specific representations and determine their effectiveness in capturing color, texture, and combined features. The study also explores the suitability of ResNet-50 and VGG-16 as backbone architectures for SimSiam. Results indicate that both models perform well, with VGG-16 achieving the best balance in accuracy across color, texture, and combined attributes. These findings demonstrate the potential of SimSiam for attribute-driven image retrieval tasks. *Index Terms*—Convolutional Neural Networks, Deep metrics,**  
**Attributes**

1. Introduction

Advances in convolutional networks have enabled increasingly viable computer vision solutions for industrial applications. One notable example is image retrieval in e-commerce, where searches can be conducted using text, images, or a combination of the key components of innovative systems. While text-based search remains the most common query method for search engines, its effectiveness relies heavily on detailed product descriptions. To enhance user experience, some search engines incorporate image-based search, offering significantly more precise results \[1\].

Image content-based search engines pose unique challenges and find applications beyond e-commerce, including medical image retrieval, geographic information systems, surveillance, and more \[12\]. The primary challenge lies in accurately extracting the relevant content of an image, particularly when the user’s interest is based on specific attributes such as color, texture, or their combination (Figure 1). In the context of e- commerce, this type of attribute-focused search is increasingly in demand. 

![][image1]

Fig. 1\. Six different color and texture-based patterns (adapted from \[2\]).

Color and texture feature extraction has been a fundamental field of study in Content-Based Image Retrieval (CBIR). Classic methods for color extraction include color histograms, which quantify the color distribution in an image, and color moments, which capture spatial information such as mean, variance, and skewness of the color distribution \[24\]. For texture analysis, techniques such as Gabor filters, which analyze the image at different scales and orientations, and Local Binary Pattern (LBP), which encodes local texture patterns robustly to variations in illumination, are widely used \[25\].

With the advent of deep learning, Convolutional Neural Networks (CNNs) have become the standard for feature extraction. Studies such as that of Balcian et al. \[2\] investigated how different layers of a ResNet-50 respond to color and texture attributes, concluding that initial layers are more sensitive to color, while deeper layers capture texture better. Other research proposes merging color and texture features extracted by CNNs to create a richer and more discriminative feature vector \[26\]. This work fits into this context, evaluating and comparing deep learning architectures, from classic CNNs to the most recent models based on Transformers and State Space Models, for the task of image retrieval based on color and texture attributes.

The dataset used in this study was created and made available by \[2\], leveraging images sourced from the Kaggle platform and other online repositories. It features a diverse collection of clothing items, including hats, dresses, shirts, socks, purses, and scarves. The dataset is organized into two distinct subsets. The first subset focuses on categorizing images by color, with the following classes: red, black, blue, green, yellow, gray, brown, pink, purple, and orange. Each class contains 100 images, resulting in a total of 1,000 images in the color dataset, as illustrated in Figure 2\.

![][image2]

Fig. 2\. Samples of images from each class in the color dataset.

The second dataset categorizes images based on their tex- ture, with the following classes: squared, striped, flowers, leopard, polka, basic, paisley, argyle, crowsfeet, and sequin. Similarly, each texture class contains 100 images, resulting in a total of 1000 images in the texture dataset. Based on this problem, the objective is to evaluate a deep metric model and the sensitivity of their layers to searching only for color and texture. Considering the proposed challenge, this study evaluates some models cited in the literature. The intention is to discover how different layers behave in relation to different attributes, colors and textures, carrying out different experiments to determine which hidden layers provide the best representation for each and their combination. That is, to evaluate which layers should be considered to represent each attribute and its combination and, finally, to analyze the behavior of the architectures to be evaluated. The selected architectures include ResNet-50 \[4\], VGG-16 \[7\], Mamba \[16\], iBOT \[17\] and LMFCN \[lmfcn\] Jonathan's architecture.

To guide the experiments, two main research questions are formulated: (RQ1) Which architecture performs best? evaluating a simpler architecture, such as VGG-16, a more complex model, such as ResNet-50 and more recent models in which it is possible to work with convolutional layers. (RQ2) Which layers within each model provide the best representation of color and texture attributes?

This study is organized as follows: Section 2 reviews related work, Section 3 details the proposed method, Section 4 discusses the experiments conducted, Section 5 experimental results and Section 6 presents the conclusions.

2. RELATED WORK

The use of convolutional networks to seek to represent the similarity of objects is widely used in the literature. Search engines use text searches as their main means of querying, but searches using image queries are advancing and bringing interesting results to this field of research, as presented by the authors in \[2\]. In addition to using images, they investigated ResNet-50 layers to determine the search for colors and textures. In their research, the authors conclude that the initial layers of the network are better for colors and the deeper layers for textures. In research, features are extracted with ResNet-50 pre-trained with the Imagenet dataset \[3\]. In feature extraction, the first convolutional layer is renamed as Block 1, and the residual blocks 2 to 5\. At the output of each block, a Global Average Pooling (GAP) layer was applied and the outputs of these layers were used as input to a k-nearest neighbor (KNN) classifier with 5-fold cross-validation.

In \[10\], the authors em- ployed neural networks to learn a set of features using object categories. In \[11\] the authors propose multi-source domain generalization techniques for cross-learning. The authors in \[8\] classify plant species by leaf image using the architecture of Siamese neural networks (SNN’s), combining features from intermediate convolutional layers to improve representations. According to the authors, the combination of characteristics from different layers presents a relevant performance gain and different layers bring different results. In the study by \[9\], the authors used a Siamese neural network to define the similarity index of the images of each class through an analysis of the entire leaf and parts of it. In \[13\], the authors present an efficient feature extraction method that is based on the concept of in-depth texture analysis. For this, wavelet packets and Eigen values from Gabor filters are used for image representation purposes and a supervised partial learning scheme that is based on K-nearest neighbors. Studies also propose the combination of color and texture features as a feature vector \[14\]. \[15\] proposed three image content features based on color, texture and color distribution, such as color co-occurrence matrix (CCM), the difference between pixels of the scan pattern (DBPSP) and color histogram for K-mean (CHKM) respectively.

Recent advances in deep learning have introduced new architectures that challenge the dominance of Transformers. The Mamba model, proposed by Gu and Dao \[16\], is a notable example. It is a State Space Model (SSM) that achieves linear-time complexity in sequence modeling, offering a significant computational advantage over the quadratic complexity of Transformers. Mamba’s selective SSMs allow it to filter and recall information based on content, a capability previously considered a key strength of attention mechanisms. This makes Mamba a promising alternative for various modalities, including language, audio, and genomics, where it has demonstrated state-of-the-art performance.

In the domain of self-supervised learning for vision, the iBOT (Image BERT Pre-Training with Online Tokenizer) model, introduced by Zhou et al. \[17\], has made significant contributions. iBOT utilizes a masked image modeling (MIM) approach, similar to BERT’s masked language modeling, but with a key innovation: an online tokenizer. This allows the model to learn a semantically meaningful visual tokenizer concurrently with the MIM objective, eliminating the need for a separate pre-training stage for the tokenizer. The integration of iBOT with convolutional layers allows for a powerful combination of local feature extraction, characteristic of CNNs, with the global context understanding of the Transformer-based iBOT architecture. This hybrid approach has shown strong performance on dense downstream tasks such as object detection and semantic segmentation. The Large Margin Fully Convolutional Network (LMFCN) was designed to train Fully Convolutional Networks (FCNs) whose outputs are subsequently used as input features for a large-margin classifier, specifically a Support Vector Machine (SVM). During training, the method updates the FCN weights to generate feature representations that maximize the inter-class margin, thereby improving the classification accuracy of the SVM.

3. APPLIED METHODOLOGY

This study evaluates different deep learning architectures to identify the most effective model for object retrieval based on color and texture. The models evaluated include two classic Convolutional Neural Networks (CNNs), ResNet-50 and VGG-16, and three architectures from recent studies, Mamba, iBOT, and LMFCN, which bring new paradigms to sequence and image modeling.

*A. ResNet-50*

The Residual Network 50 (ResNet-50), introduced by He et al. \[4\], is a 50-layer deep convolutional neural network. Its core innovation is the use of “residual blocks” with skip connections, which allow the model to learn residual functions. This architecture effectively addresses the vanishing gradient problem in very deep networks, enabling the training of models with hundreds or even thousands of layers while maintaining high performance. ResNet-50 is widely used as a backbone for various computer vision tasks due to its powerful feature extraction capabilities (Figure 3).

*![][image3]*

 Fig. 3\. Architecture of the ResNet-50 model. Adaptado de Mukherjee \[23\], baseado no trabalho original de He et al. \[4\].

### *B. VGG-16*

The VGG-16 model, proposed by Simonyan and Zisserman, is a convolutional neural network characterized by its simplicity and depth \[7\]. It is composed of 16 layers, primarily using small 3x3 convolutional filters stacked on top of each other. This stacking of small filters allows the model to have a large receptive field while maintaining a low number of parameters. Despite its relatively simple architecture, VGG-16 is known for its excellent performance in image classification and its features have proven to be highly transferable to other tasks (Figure 4).

![][image4]

Fig. 4\. Architecture of the VGG-16 model, illustrating its sequential structure. Source: Simonyan & Zisserman \[7\].

### 

### *C. Mamba*

Mamba, developed by Gu and Dao \[16\], represents a new class of sequence models based on Structured State Space Models (SSMs). Unlike Transformers, which have quadratic complexity with respect to sequence length, Mamba processes sequences in linear time. It employs a selective scan mechanism that allows the model to selectively propagate or forget information along a sequence, enabling content-based reasoning. This makes Mamba highly efficient for long sequences and a strong performer across different modalities, including vision, where it can be adapted as a backbone for feature extraction (Figure 4).

*![][image5]*

Fig. 4\. Mamba block diagram, illustrating the selective SSM mechanism. Source: Gu & Dao \[16\].

### *D. iBOT with Convolutional Layers*

The iBOT (Image BERT Pre-Training with Online Tokenizer) model, from Zhou et al. \[17\], is a self-supervised learning framework based on Vision Transformers (ViT). It uses a Masked Image Modeling (MIM) objective where parts of an image are masked, and the model must predict the masked content. A key feature of iBOT is its online tokenizer, which learns to create semantically rich visual tokens during the pre-training process. For this study, iBOT is combined with convolutional layers, creating a hybrid model. This approach leverages the strengths of CNNs in capturing local features and fine-grained details, while the Transformer component models the global context and relationships between image patches (Figure 5).

*![][image6]*

Fig. 5\. Overview of the iBOT framework, showing the r architecture with an online tokenizer. Source: Zhou et al. \[17\].

### *E. LMFCN*

![][image7]

Fig. xxx. Overview of the LMFCN. Source: de Matos et al. \[lmfcn\].

The LMFCN architecture is presented in Fig. xxx. It takes as input images from the training set T, extracts feature representations using an FCN, and uses these features to construct the kernel matrix K for an SVM classifier. The FCN may be any convolutional architecture capable of producing a latent representation of the data, such as a TCNN \[lmfcn\], ResNet, VGG, or Inception network. The matrix D is a distance matrix defined using the same metric as the kernel matrix K and is employed to measure distances between training instances and the SVM support vectors. Based on the SVM classification results, the distance matrix, and the support vectors, the FCN parameters are updated using a loss function that increases the penalty for (i) instances that are far from the nearest support vector of the same class, (ii) instances that are close to support vectors of the opposite class, and (iii) instances that are close to non-support instances of the opposite class. The sequence of operations described above defines a single training epoch. This method was designed primarily for training small and lightweight FCNs on limited-size datasets, particularly in texture-related classification tasks.

4. Backbone models evaluated

In this study, classical and modern models were selected. The VGG-16 is a classical CNN with 16 weighted layers organized into 5 convolutional blocks using 3×3 filters, each followed by max-pooling, with feature dimensions of 64, 128, 256, 512, and 512. The ResNet-50 introduced residual connections (skip connections) enabling deeper networks, consisting of 4 main residual blocks with dimensions of 256, 512, 1024, and 2048. Features are extracted from both CNNs using Global Average Pooling (GAP).

For modern architectures, the iBOT (Image BERT Pre-Training with Online Tokenizer) is a Vision Transformer pre-trained using self-supervised masked image modeling, with the ViT-S/16 variant having 12 transformer blocks producing 384-dimensional embeddings extracted via the CLS token. The VMamba adapts the Mamba architecture—a selective state space model—for visual tasks, achieving linear complexity unlike transformers; the VMamba-Tiny variant has 4 stages with dimensions of 96, 192, 384, and 768. Additionally, the LMFCN framework was evaluated using TCNN architectures and ResNet-18.

Both CNNs are cited in different works. For the study, the outputs of the initial five layers trained with ImageNet were evaluated with CNN. The choice of these layers is based on studies published in the literature, \[2\] and \[8\], which detected different results in each of these layers. Initial tests allowed us to identify which layers are best for color and texture for each of these CNNs. Features were extracted using Global Average Pooling on the output of each layer. The generated vectors are fed to the kNN (K-nearest neighbors) algorithm using 5-fold cross validation. Global Average Pooling is a function to reduce the spatial size of the representation to reduce the number of parameters and computation in the network. It is mainly used to reduce the dimensions of the feature map. After discovering the best color and texture layer for the models with the proposed CNNs, the best layers are also useful to compare with the normal application of CNNs. For clarity, we name layer 1 to layer 5 as CNNLayer1, CNNLayer2, CNNLayer3, CNNLayer4 and CNNLayer5, for each CNN.

The LMFCN was evaluated using two fully convolutional network (FCN) architectures: a texture convolutional neural network (TCNN) with three, four, five, and six layers, and a pretrained ResNet-18 with the final fully connected layers removed. For the ResNet-18 architecture, two configurations were examined: one comprising all convolutional blocks and another retaining only the first two blocks. The FCNs were trained using the LMFCN framework, and the classification accuracy at each FCN layer was assessed using the previously described k-nearest neighbors (kNN) evaluation method. These architectures were selected in accordance with the LMFCN design objective of effective integration with small and lightweight convolutional networks.

5. EXPERIMENTAL RESULTS

This section will present the database and experiments carried out to answer the proposed research questions.

1. *Experimental Protocol*

The dataset used in this research was the same used by the authors in \[2\], which consists of 10 classes for color, each containing 100 images, and 10 classes for texture, also with 100 images each, totaling 1000 images for the color set and 1000 images for the texture set. The base contains two sets of data using Kaggle, one grouped by color and the other by texture. The description of these datasets is given below:  
(a) Color dataset: Contains 100 images of clothing in each of the following colors: red, black, blue, green, yellow, gray, brown, pink, purple, and orange; (b)Texture dataset: Contains 100 clothing images of each of the following texture patterns: square, striped, flowers, leopard, polka, basic, paisley, argyle, crow’s feet, and sequins. The dataset is compiled from the various garments, ranging from socks to shirts.

The database was divided into 70% for training and 30% for testing.


B. Experimental Results

Table I presents the classification accuracy obtained for the hidden layers of VGG-16 and ResNet-50. Features were extracted using Global Average Pooling (GAP) and evaluated with k-NN using 5-fold cross-validation. The results reveal a clear relationship between layer depth and attribute sensitivity. In VGG-16, Layer 1 achieved the highest color accuracy (95.13%), with performance decreasing progressively in deeper layers until reaching 55.07% at Layer 5. Conversely, texture accuracy improved with depth, peaking at Layer 4 (94.73%). ResNet-50 exhibited similar behavior: Layer 2 obtained the best color result (95.87%), while Layer 3 was most effective for texture (93.27%). The accuracy drop in the final layers of both networks, particularly for color classification, suggests that low-level features captured in initial layers are more relevant for distinguishing chromatic attributes.

TABLE I - Accuracy of hidden layers of VGG-16 and ResNet-50 for color and texture classification.

| Layer | Color | Standard Deviation | Texture | Standard Deviation | Dim |
| :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-16 Layer 1 | **0.9513** | 0.0081 | 0.7073 | 0.0225 | 64 |
| VGG-16 Layer 2 | 0.9027 | 0.0127 | 0.8660 | 0.0157 | 128 |
| VGG-16 Layer 3 | 0.8547 | 0.0233 | 0.9320 | 0.0148 | 256 |
| VGG-16 Layer 4 | 0.7180 | 0.0255 | **0.9473** | 0.0074 | 512 |
| VGG-16 Layer 5 | 0.5507 | 0.0264 | 0.8940 | 0.0164 | 512 |
| ResNet-50 Layer 1 | 0.9547 | 0.0078 | 0.8340 | 0.0136 | 256 |
| ResNet-50 Layer 2 | **0.9587** | 0.0050 | 0.9067 | 0.0165 | 512 |
| ResNet-50 Layer 3 | 0.8827 | 0.0112 | **0.9327** | 0.0025 | 1024 |
| ResNet-50 Layer 4 | 0.5920 | 0.0311 | 0.8633 | 0.0092 | 2048 |

The evaluation was extended to modern architectures to verify whether this layer-wise pattern generalizes beyond traditional CNNs. Table II shows the results for iBOT and VMamba. For iBOT, features were extracted from each transformer block via the CLS token, while for VMamba, features were obtained from the output of each processing stage. The iBOT model demonstrated a particularly pronounced gradient: color accuracy decreased from 94.53% at Block 0 to 62.80% at Block 11, while texture accuracy increased from 52.20% to 97.20% at Block 9 before slightly decreasing in the final blocks. This 97.20% texture accuracy represents the highest value among all evaluated architectures, indicating that self-supervised Vision Transformers capture especially discriminative representations for texture patterns. VMamba showed a more compressed behavior due to its four-stage architecture, with Stage 1 achieving the best color accuracy (94.41%) and Stages 2-3 achieving the best texture accuracy (95.76%).

TABLE II - Accuracy of iBOT blocks and VMamba stages for color and texture classification.

| Layer | Color | Standard Deviation | Texture | Standard Deviation | Dim |
| :---: | :---: | :---: | :---: | :---: | :---: |
| iBOT Block 0 | 0.9453 | 0.0113 | 0.5220 | 0.0269 | 384 |
| iBOT Block 1 | 0.9540 | 0.0127 | 0.6987 | 0.0228 | 384 |
| iBOT Block 2 | **0.9587** | 0.0105 | 0.7993 | 0.0261 | 384 |
| iBOT Block 3 | 0.9520 | 0.0083 | 0.8787 | 0.0144 | 384 |
| iBOT Block 4 | 0.9213 | 0.0027 | 0.9053 | 0.0133 | 384 |
| iBOT Block 5 | 0.8873 | 0.0077 | 0.9100 | 0.0202 | 384 |
| iBOT Block 6 | 0.8493 | 0.0124 | 0.9220 | 0.0185 | 384 |
| iBOT Block 7 | 0.8407 | 0.0118 | 0.9500 | 0.0099 | 384 |
| iBOT Block 8 | 0.7600 | 0.0076 | 0.9687 | 0.0054 | 384 |
| iBOT Block 9 | 0.6993 | 0.0197 | **0.9720** | 0.0054 | 384 |
| iBOT Block 10 | 0.6613 | 0.0328 | 0.9627 | 0.0053 | 384 |
| iBOT Block 11 | 0.6280 | 0.0173 | 0.9600 | 0.0087 | 384 |
| VMamba Stage 1 | **0.9441** | 0.0052 | 0.9386 | 0.0107 | 192 |
| VMamba Stage 2 | 0.9341 | 0.0047 | **0.9576** | 0.0084 | 384 |
| VMamba Stage 3 | 0.7834 | 0.0375 | 0.9576 | 0.0075 | 768 |
| VMamba Stage 4 | 0.5414 | 0.0239 | 0.8652 | 0.0257 | 768 |

Table III consolidates the best results obtained by each architecture. For color classification, ResNet-50 and iBOT tied with 95.87% accuracy, followed closely by VGG-16 (95.13%) and VMamba (94.41%). For texture classification, iBOT stood out with 97.20%, surpassing VMamba (95.76%), VGG-16 (94.73%), and ResNet-50 (93.27%). These results indicate that while classical CNNs remain competitive for color recognition, modern architectures—particularly self-supervised Vision Transformers—offer advantages for texture-based tasks.

TABLE III - Best accuracy achieved by each backbone architecture.

| Backbone | Best Color Layer | Color | Standard Deviation | Best Texture Layer | Texture | Standard Deviation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-16 | Layer 1 | 0.9513 | 0.0081 | Layer 4 | 0.9473 | 0.0074 |
| ResNet-50 | Layer 2 | 0.9587 | 0.0050 | Layer 3 | 0.9327 | 0.0025 |
| iBOT | Block 2 | 0.9587 | 0.0105 | Block 9 | **0.9720** | 0.0054 |
| VMamba | Stage 1 | 0.9441 | 0.0052 | Stage 2 | 0.9576 | 0.0084 |

Figure 7 illustrates sample images from the color and texture datasets, along with examples of misclassified images. Analysis of the errors reveals that most misclassifications occur between visually similar classes. In the color dataset, confusions frequently involve adjacent hues in the color spectrum, such as gray versus black or pink versus red, which can appear similar under certain lighting conditions. In the texture dataset, errors arise between patterns with similar structural characteristics at different scales, such as polka dots confused with leopard spots, or sequined fabrics misclassified as leopard print due to their spotted appearance.

![Color Classification Examples](visualizations/ibot_color_amostras.png)

*Fig. 7a. Sample images from the color dataset showing different color classes.*

![Texture Classification Examples](visualizations/ibot_texture_amostras.png)

*Fig. 7b. Sample images from the texture dataset showing different texture patterns.*

![Incorrect Classifications](visualizations/ibot_texture_incorretos.png)

*Fig. 7c. Examples of misclassified texture images (predicted class above, true class below).*

C. Discussion

The experimental results presented in this study provide evidence that the layer-wise sensitivity pattern previously observed in classical CNNs extends to modern architectures based on Vision Transformers and State Space Models. Across all four evaluated backbones, early layers consistently captured features more suitable for color classification, while deeper layers proved more effective for texture recognition. This pattern held regardless of the architectural paradigm—whether convolutional (VGG-16, ResNet-50), attention-based (iBOT), or state-space-based (VMamba).

The performance differences between architectures offer insights into their representational characteristics. iBOT's exceptional texture accuracy (97.20%) suggests that the self-supervised masked image modeling objective encourages the learning of rich structural representations in deeper layers. The relatively uniform feature dimensionality across iBOT blocks (384 dimensions) contrasts with the increasing dimensions in CNNs and VMamba, yet iBOT achieved superior texture discrimination. This indicates that the quality of learned representations, rather than their dimensionality, is the determining factor for texture classification.

The practical implications of these findings are significant for content-based image retrieval systems. Rather than using features from the final layer of a pretrained network—as is common practice—our results suggest that feature extraction should be layer-specific based on the target attribute. For applications requiring color-based retrieval, features from early layers should be preferred, while texture-based retrieval benefits from deeper layer representations. Furthermore, the choice of backbone architecture should consider the primary attribute of interest: classical CNNs offer a good balance for color tasks, while iBOT provides substantial advantages for texture-focused applications.

6. CONCLUSIONS

This study presents a comprehensive evaluation of feature extraction from hidden layers across multiple deep learning architectures for color and texture classification in fashion images. We evaluated classical CNNs (VGG-16 and ResNet-50) and modern architectures (iBOT and VMamba).

The experiments confirm that the layer-wise sensitivity pattern observed in previous studies extends to modern Vision Transformers and State Space Models: early layers are more effective for color classification, while deeper layers excel at texture recognition. Among the evaluated architectures, iBOT achieved the highest texture accuracy (97.20% at Block 9), while ResNet-50 and iBOT tied for best color accuracy (95.87%).

These findings have practical implications for content-based image retrieval systems, suggesting that feature extraction should be layer-specific based on the target attribute. Future work will investigate combining features from optimal layers across different architectures to potentially improve classification performance.

Acknowledgment

This work is supported by the Graduate Program in Informatics at the Pontifical Catholic University of Paraná and partially supported by the Brazilian Council for Scientific and Technological Development (CNPq).

References

[1] S. R. Dubey, "A decade survey of content based image retrieval using deep learning," IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 5, pp. 2687-2704, 2021.

[2] A. Baloian, N. Murrugarra-Llerena, and J. M. Saavedra, "Scalable visual attribute extraction through hidden layers of a residual convnet," arXiv preprint arXiv:2104.00161, 2021.

[3] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in Proc. IEEE CVPR, 2009, pp. 248-255.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE CVPR, 2016, pp. 770-778.

[5] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.

[6] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," arXiv preprint arXiv:2312.00752, 2023.

[7] J. Zhou et al., "iBOT: Image BERT pre-training with online tokenizer," arXiv preprint arXiv:2111.07832, 2021.

[8] M. Moresco et al., "Combining multi-layer features for plant species classification in a Siamese network," in Proc. IEEE SMC, 2022, pp. 2446-2451.
