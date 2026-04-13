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

***Abstract***—This study investigates how color and texture information are distributed across the depth of different neural network architectures for visual classification tasks. We evaluate classical Convolutional Neural Networks (VGG-16 and ResNet-50) alongside modern architectures based on Vision Transformers (iBOT) and State Space Models (VMamba), as well as the LMFCN framework. Through systematic layer-wise analysis using k-NN classification, we identify the optimal layers for color and texture feature extraction in each architecture. Our experiments reveal a consistent pattern across all architectures: early layers are more effective for color classification, while deeper layers excel at texture recognition. Furthermore, we investigate whether early-deep feature fusion improves classification performance compared to single-layer representations. Results indicate that iBOT achieves the highest texture accuracy (97.20%), while ResNet-50 and iBOT tie for best color accuracy (95.87%). Notably, simple concatenation fusion does not improve over specialized single-layer representations, suggesting that attribute-specific feature extraction is more effective than naive multi-layer fusion.

*Index Terms*—Convolutional Neural Networks, Vision Transformers, State Space Models, Feature Extraction, Color Classification, Texture Classification

1. Introduction

Advances in convolutional networks have enabled increasingly viable computer vision solutions for industrial applications. One notable example is image retrieval in e-commerce, where searches can be conducted using text, images, or a combination of the key components of innovative systems. While text-based search remains the most common query method for search engines, its effectiveness relies heavily on detailed product descriptions. To enhance user experience, some search engines incorporate image-based search, offering significantly more precise results \[1\].

Image content-based search engines pose unique challenges and find applications beyond e-commerce, including medical image retrieval, geographic information systems, surveillance, and more \[12\]. The primary challenge lies in accurately extracting the relevant content of an image, particularly when the user’s interest is based on specific attributes such as color, texture, or their combination (Figure 1). In the context of e- commerce, this type of attribute-focused search is increasingly in demand. 

![][image1]

Fig. 1\. Six different color and texture-based patterns (adapted from \[2\]).

Color and texture feature extraction has been a fundamental field of study in Content-Based Image Retrieval (CBIR). Classic methods for color extraction include color histograms, which quantify the color distribution in an image, and color moments, which capture spatial information such as mean, variance, and skewness of the color distribution \[24\]. For texture analysis, techniques such as Gabor filters, which analyze the image at different scales and orientations, and Local Binary Pattern (LBP), which encodes local texture patterns robustly to variations in illumination, are widely used \[25\].

With the advent of deep learning, Convolutional Neural Networks (CNNs) have become the standard for feature extraction. Studies such as that of Baloian et al. \[2\] investigated how different layers of a ResNet-50 respond to color and texture attributes, concluding that initial layers are more sensitive to color, while deeper layers capture texture better. Other research proposes merging color and texture features extracted by CNNs to create a richer and more discriminative feature vector \[26\]. This work fits into this context, evaluating and comparing deep learning architectures, from classic CNNs to the most recent models based on Transformers and State Space Models, for the task of image retrieval based on color and texture attributes.

The dataset used in this study was created and made available by \[2\], leveraging images sourced from the Kaggle platform and other online repositories. It features a diverse collection of clothing items, including hats, dresses, shirts, socks, purses, and scarves. The dataset is organized into two distinct subsets. The first subset focuses on categorizing images by color, with the following classes: red, black, blue, green, yellow, gray, brown, pink, purple, and orange. Each class contains 100 images, resulting in a total of 1,000 images in the color dataset, as illustrated in Figure 2\.

![][image2]

Fig. 2\. Samples of images from each class in the color dataset.

The second dataset categorizes images based on their tex- ture, with the following classes: squared, striped, flowers, leopard, polka, basic, paisley, argyle, crowsfeet, and sequin. Similarly, each texture class contains 100 images, resulting in a total of 1000 images in the texture dataset. Based on this problem, the objective is to evaluate a deep metric model and the sensitivity of their layers to searching only for color and texture. Considering the proposed challenge, this study evaluates some models cited in the literature. The intention is to discover how different layers behave in relation to different attributes, colors and textures, carrying out different experiments to determine which hidden layers provide the best representation for each and their combination. That is, to evaluate which layers should be considered to represent each attribute and its combination and, finally, to analyze the behavior of the architectures to be evaluated. The selected architectures include ResNet-50 \[4\], VGG-16 \[7\], VMamba \[16\], iBOT \[17\], and LMFCN \[28\].

The main objectives of this study are to investigate how color and texture information distribute across the depth of different neural network architectures, identify which layers yield the best classification performance for each attribute, verify whether the last-layer representation is optimal or if intermediate layers provide superior performance, evaluate whether early-deep feature fusion improves classification compared to single-layer representations, and determine if layer-wise patterns are consistent across CNN, Transformer, and Mamba architectures.

This study is organized as follows: Section 2 reviews related work, Section 3 details the proposed method, Section 4 discusses the experiments conducted, Section 5 experimental results and Section 6 presents the conclusions.

2. RELATED WORK

The use of convolutional networks to seek to represent the similarity of objects is widely used in the literature. Search engines use text searches as their main means of querying, but searches using image queries are advancing and bringing interesting results to this field of research, as presented by the authors in \[2\]. In addition to using images, they investigated ResNet-50 layers to determine the search for colors and textures. In their research, the authors conclude that the initial layers of the network are better for colors and the deeper layers for textures. In research, features are extracted with ResNet-50 pre-trained with the Imagenet dataset \[3\]. In feature extraction, the first convolutional layer is renamed as Block 1, and the residual blocks 2 to 5\. At the output of each block, a Global Average Pooling (GAP) layer was applied and the outputs of these layers were used as input to a k-nearest neighbor (KNN) classifier with 5-fold cross-validation.

In \[10\], the authors em- ployed neural networks to learn a set of features using object categories. In \[11\] the authors propose multi-source domain generalization techniques for cross-learning. The authors in \[8\] classify plant species by leaf image using the architecture of Siamese neural networks (SNN’s), combining features from intermediate convolutional layers to improve representations. According to the authors, the combination of characteristics from different layers presents a relevant performance gain and different layers bring different results. In the study by \[9\], the authors used a Siamese neural network to define the similarity index of the images of each class through an analysis of the entire leaf and parts of it. In \[13\], the authors present an efficient feature extraction method that is based on the concept of in-depth texture analysis. For this, wavelet packets and Eigen values from Gabor filters are used for image representation purposes and a supervised partial learning scheme that is based on K-nearest neighbors. Studies also propose the combination of color and texture features as a feature vector \[14\]. \[15\] proposed three image content features based on color, texture and color distribution, such as color co-occurrence matrix (CCM), the difference between pixels of the scan pattern (DBPSP) and color histogram for K-mean (CHKM) respectively.

Multi-layer feature fusion has been explored as a strategy to combine representations from different depths of neural networks. The intuition is that early layers capture low-level features (edges, colors) while deeper layers encode high-level semantic information, and combining them could yield richer representations \[26\]. Common fusion strategies include early fusion (concatenating features before classification), late fusion (combining classifier outputs), and learned fusion using attention mechanisms \[27\]. However, the effectiveness of such approaches for attribute-specific tasks like color and texture classification remains underexplored, particularly across different architectural paradigms.

Recent advances in deep learning have introduced new architectures that challenge the dominance of Transformers. The Mamba model, proposed by Gu and Dao \[16\], is a notable example. It is a State Space Model (SSM) that achieves linear-time complexity in sequence modeling, offering a significant computational advantage over the quadratic complexity of Transformers. Mamba’s selective SSMs allow it to filter and recall information based on content, a capability previously considered a key strength of attention mechanisms. This makes Mamba a promising alternative for various modalities, including language, audio, and genomics, where it has demonstrated state-of-the-art performance.

In the domain of self-supervised learning for vision, the iBOT (Image BERT Pre-Training with Online Tokenizer) model, introduced by Zhou et al. \[17\], has made significant contributions. iBOT utilizes a masked image modeling (MIM) approach, similar to BERT’s masked language modeling, but with a key innovation: an online tokenizer. This allows the model to learn a semantically meaningful visual tokenizer concurrently with the MIM objective, eliminating the need for a separate pre-training stage for the tokenizer. The integration of iBOT with convolutional layers allows for a powerful combination of local feature extraction, characteristic of CNNs, with the global context understanding of the Transformer-based iBOT architecture. This hybrid approach has shown strong performance on dense downstream tasks such as object detection and semantic segmentation. The Large Margin Fully Convolutional Network (LMFCN) was designed to train Fully Convolutional Networks (FCNs) whose outputs are subsequently used as input features for a large-margin classifier, specifically a Support Vector Machine (SVM). During training, the method updates the FCN weights to generate feature representations that maximize the inter-class margin, thereby improving the classification accuracy of the SVM.

3. APPLIED METHODOLOGY

This study evaluates different deep learning architectures to identify the most effective model for object retrieval based on color and texture. The models evaluated include two classic Convolutional Neural Networks (CNNs), ResNet-50 and VGG-16, and three architectures from recent studies, Mamba, iBOT, and LMFCN, which bring new paradigms to sequence and image modeling.

*A. ResNet-50*

The Residual Network 50 (ResNet-50), introduced by He et al. \[4\], is a 50-layer deep convolutional neural network. Its core innovation is the use of “residual blocks” with skip connections, which allow the model to learn residual functions. This architecture effectively addresses the vanishing gradient problem in very deep networks, enabling the training of models with hundreds or even thousands of layers while maintaining high performance. ResNet-50 is widely used as a backbone for various computer vision tasks due to its powerful feature extraction capabilities (Figure 3).

*![][image3]*

 Fig. 3\. Architecture of the ResNet-50 model. Adapted from Mukherjee \[23\], based on the original work of He et al. \[4\].

### *B. VGG-16*

The VGG-16 model, proposed by Simonyan and Zisserman, is a convolutional neural network characterized by its simplicity and depth \[7\]. It is composed of 16 layers, primarily using small 3x3 convolutional filters stacked on top of each other. This stacking of small filters allows the model to have a large receptive field while maintaining a low number of parameters. Despite its relatively simple architecture, VGG-16 is known for its excellent performance in image classification and its features have proven to be highly transferable to other tasks (Figure 4).

![][image4]

Fig. 4\. Architecture of the VGG-16 model, illustrating its sequential structure. Source: Simonyan & Zisserman \[7\].

### 

### *C. Mamba*

Mamba, developed by Gu and Dao \[16\], represents a new class of sequence models based on Structured State Space Models (SSMs). Unlike Transformers, which have quadratic complexity with respect to sequence length, Mamba processes sequences in linear time. It employs a selective scan mechanism that allows the model to selectively propagate or forget information along a sequence, enabling content-based reasoning. This makes Mamba highly efficient for long sequences and a strong performer across different modalities, including vision, where it can be adapted as a backbone for feature extraction (Figure 5).

*![][image5]*

Fig. 5\. Mamba block diagram, illustrating the selective SSM mechanism. Source: Gu & Dao \[16\].

### *D. iBOT with Convolutional Layers*

The iBOT (Image BERT Pre-Training with Online Tokenizer) model, from Zhou et al. \[17\], is a self-supervised learning framework based on Vision Transformers (ViT). It uses a Masked Image Modeling (MIM) objective where parts of an image are masked, and the model must predict the masked content. A key feature of iBOT is its online tokenizer, which learns to create semantically rich visual tokens during the pre-training process. For this study, iBOT is combined with convolutional layers, creating a hybrid model. This approach leverages the strengths of CNNs in capturing local features and fine-grained details, while the Transformer component models the global context and relationships between image patches (Figure 6).

*![][image6]*

Fig. 6\. Overview of the iBOT framework, showing the architecture with an online tokenizer. Source: Zhou et al. \[17\].

### *E. LMFCN*

![][image7]

Fig. 7\. Overview of the LMFCN. Source: de Matos et al. \[28\].

The LMFCN architecture is presented in Fig. 7. It takes as input images from the training set T, extracts feature representations using an FCN, and uses these features to construct the kernel matrix K for an SVM classifier. The FCN may be any convolutional architecture capable of producing a latent representation of the data, such as a TCNN \[28\], ResNet, VGG, or Inception network. The matrix D is a distance matrix defined using the same metric as the kernel matrix K and is employed to measure distances between training instances and the SVM support vectors. Based on the SVM classification results, the distance matrix, and the support vectors, the FCN parameters are updated using a loss function that increases the penalty for (i) instances that are far from the nearest support vector of the same class, (ii) instances that are close to support vectors of the opposite class, and (iii) instances that are close to non-support instances of the opposite class. The sequence of operations described above defines a single training epoch. This method was designed primarily for training small and lightweight FCNs on limited-size datasets, particularly in texture-related classification tasks.

4. Backbone models evaluated

In this study, classical and modern models were selected. The VGG-16 is a classical CNN with 16 weighted layers organized into 5 convolutional blocks using 3×3 filters, each followed by max-pooling, with feature dimensions of 64, 128, 256, 512, and 512\. The ResNet-50 introduced residual connections (skip connections) enabling deeper networks, consisting of 4 main residual blocks with dimensions of 256, 512, 1024, and 2048\. Features are extracted from both CNNs using Global Average Pooling (GAP).

For modern architectures, the iBOT (Image BERT Pre-Training with Online Tokenizer) is a Vision Transformer pre-trained using self-supervised masked image modeling, with the ViT-S/16 variant having 12 transformer blocks producing 384-dimensional embeddings extracted via the CLS token. The VMamba adapts the Mamba architecture—a selective state space model—for visual tasks, achieving linear complexity unlike transformers; the VMamba-Tiny variant has 4 stages with dimensions of 96, 192, 384, and 768\. Additionally, the LMFCN framework was evaluated using TCNN architectures and ResNet-18.

Both CNNs are cited in different works. For the study, the outputs of the initial five layers trained with ImageNet were evaluated with CNN. The choice of these layers is based on studies published in the literature, \\\[2\\\] and \\\[8\\\], which detected different results in each of these layers. Initial tests allowed us to identify which layers are best for color and texture for each of these CNNs. Features were extracted using Global Average Pooling on the output of each layer. The generated vectors are fed to the kNN (K-nearest neighbors) algorithm using 5-fold cross validation. Global Average Pooling is a function to reduce the spatial size of the representation to reduce the number of parameters and computation in the network. It is mainly used to reduce the dimensions of the feature map. After discovering the best color and texture layer for the models with the proposed CNNs, the best layers are also useful to compare with the normal application of CNNs. For clarity, we name layer 1 to layer 5 as CNNLayer1, CNNLayer2, CNNLayer3, CNNLayer4 and CNNLayer5, for each CNN.

The LMFCN was evaluated using two fully convolutional network (FCN) architectures: a texture convolutional neural network (TCNN) with three, four, five, and six layers, and a pretrained ResNet-18 with the final fully connected layers removed. For the ResNet-18 architecture, two configurations were examined: one comprising all convolutional blocks and another retaining only the first two blocks. The FCNs were trained using the LMFCN framework, and the classification accuracy at each FCN layer was assessed using the previously described k-nearest neighbors (kNN) evaluation method. These architectures were selected in accordance with the LMFCN design objective of effective integration with small and lightweight convolutional networks.

5. EXPERIMENTAL RESULTS

This section presents the experimental protocol and results obtained to address the proposed objectives.

1. *Experimental Protocol*

The dataset used in this research was the same used by the authors in \[2\], which consists of 10 classes for color, each containing 100 images, and 10 classes for texture, also with 100 images each, totaling 1000 images for the color set and 1000 images for the texture set. The base contains two sets of data using Kaggle, one grouped by color and the other by texture. The description of these datasets is given below:  
(a) Color dataset: Contains 100 images of clothing in each of the following colors: red, black, blue, green, yellow, gray, brown, pink, purple, and orange; (b)Texture dataset: Contains 100 clothing images of each of the following texture patterns: square, striped, flowers, leopard, polka, basic, paisley, argyle, crow’s feet, and sequins. The dataset is compiled from the various garments, ranging from socks to shirts.

The database was divided into 70% for training and 30% for testing.

B. Results with LMFCN

Table XXX presents the results obtained on the texture dataset, indicating that increasing the depth of the TCNN leads to improved performance, with accuracy consistently rising as the FCN depth increases. In contrast, the ResNet-18 configuration including all four convolutional blocks yielded inferior kNN performance on texture data. The results obtained on the color dataset indicate that, in comparison to the texture dataset, higher classification accuracy is achieved from the early layers of the network. In the ResNet-18 configuration with four convolutional blocks, a degradation in accuracy was observed at the third layer. The LMFCN with its SVM classifier obtained the best performance for the texture dataset for the ResNet-18 with two blocks, achieving 96,87% of accuracy and for the color dataset with the TCNN with three layers, with accuracy of 97,73%.

Table xxx \- Accuracy of the hidden layers of a ResNet-18 and an TCNN using LMFCN for training and finetuning

| Method | Color (%) | Texture (%) |
| :---- | :---- | :---- |
| TCNN3 Layer 1 | 0,9333 | 0,4873 |
| TCNN3 Layer 2 | 0,9513 | 0,5513 |
| TCNN3 Layer 3 | 0,9594 | 0,5827 |
| TCNN4 Layer 1 | 0,9267 | 0,4867 |
| TCNN4 Layer 2 | 0,9434 | 0,5473 |
| TCNN4 Layer 3 | 0,9420 | 0,5973 |
| TCNN4 Layer 4 | 0,9480 | 0,6040 |
| TCNN5 Layer 1 | 0,9326 | 0,4867 |
| TCNN5 Layer 2 | 0,9487 | 0,5533 |
| TCNN5 Layer 3 | 0,9433 | 0,6080 |
| TCNN5 Layer 4 | 0,9467 | 0,6227 |
| TCNN5 Layer 5 | 0,9540 | 0,6293 |
| TCNN6 Layer 1 | 0,9287 | 0,4933 |
| TCNN6 Layer 2 | 0,9467 | 0,5647 |
| TCNN6 Layer 3 | 0,9473 | 0,6087 |
| TCNN6 Layer 4 | 0,9453 | 0,6347 |
| TCNN6 Layer 5 | 0,9433 | 0,6327 |
| TCNN6 Layer 6 | 0,9453 | 0,6434 |
| ResNet18 (2 blocks) Layer 1 | 0,9500 | 0,6273 |
| ResNet18 (2 blocks) Layer 2 | 0,9613 | 0,8487 |
| ResNet18 (2 blocks) Layer 3 | 0,9193 | 0,9100 |
| ResNet18 (4 blocks) Layer 1 | 0,9453 | 0,6367 |
| ResNet18 (4 blocks) Layer 2 | 0,9593 | 0,8413 |
| ResNet18 (4 blocks) Layer 3 | 0,9107 | 0,9120 |
| ResNet18 (4 blocks) Layer 4 | 0,8953 | 0,9247 |
| ResNet18 (4 blocks) Layer 5 | 0,7520 | 0,8674 |

B. Experimental Results

Table I presents the classification accuracy obtained for the hidden layers of VGG-16 and ResNet-50. Features were extracted using Global Average Pooling (GAP) and evaluated with k-NN using 5-fold cross-validation. The results reveal a clear relationship between layer depth and attribute sensitivity. In VGG-16, Layer 1 achieved the highest color accuracy (95.13%), with performance decreasing progressively in deeper layers until reaching 55.07% at Layer 5\. Conversely, texture accuracy improved with depth, peaking at Layer 4 (94.73%). ResNet-50 exhibited similar behavior: Layer 2 obtained the best color result (95.87%), while Layer 3 was most effective for texture (93.27%). The accuracy drop in the final layers of both networks, particularly for color classification, suggests that low-level features captured in initial layers are more relevant for distinguishing chromatic attributes.

TABLE I \- Accuracy of hidden layers of VGG-16 and ResNet-50 for color and texture classification.

| Layer | Color | Std\_Color | Texture | Std\_Texture | Dim |
| ----- | :---: | :---: | :---: | :---: | :---: |
| VGG-16 Layer 1 | 0.9513 | 0.0081 | 0.7073 | 0.0225 | 64 |
| VGG-16 Layer 2 | 0.9027 | 0.0127 | 0.866 | 0.0157 | 128 |
| VGG-16 Layer 3 | 0.8547 | 0.0233 | 0.932 | 0.0148 | 256 |
| VGG-16 Layer 4 | 0.718 | 0.0255 | 0.9473 | 0.0074 | 512 |
| VGG-16 Layer 5 | 0.5507 | 0.0264 | 0.894 | 0.0164 | 512 |
| ResNet-50 Layer 1 | 0.9547 | 0.0078 | 0.834 | 0.0136 | 256 |
| ResNet-50 Layer 2 | 0.9587 | 0.005 | 0.9067 | 0.0165 | 512 |
| ResNet-50 Layer 3 | 0.8827 | 0.0112 | 0.9327 | 0.0025 | 1024 |
| ResNet-50 Layer 4 | 0.592 | 0.0311 | 0.8633 | 0.0092 | 2048 |

The evaluation was extended to modern architectures to verify whether this layer-wise pattern generalizes beyond traditional CNNs. Table II shows the results for iBOT and VMamba. For iBOT, features were extracted from each transformer block via the CLS token, while for VMamba, features were obtained from the output of each processing stage. The iBOT model demonstrated a particularly pronounced gradient: color accuracy decreased from 94.53% at Block 0 to 62.80% at Block 11, while texture accuracy increased from 52.20% to 97.20% at Block 9 before slightly decreasing in the final blocks. This 97.20% texture accuracy represents the highest value among all evaluated architectures, indicating that self-supervised Vision Transformers capture especially discriminative representations for texture patterns. VMamba showed a more compressed behavior due to its four-stage architecture, with Stage 1 achieving the best color accuracy (94.41%) and Stages 2-3 achieving the best texture accuracy (95.76%).

TABLE II \- Accuracy of iBOT blocks and VMamba stages for color and texture classification.

| Layer | Color | Std\_Color | Texture | Std\_Texture | Dim |
| ----- | :---: | :---: | :---: | :---: | :---: |
| iBOT Block 0 | 0.9453 | 0.0113 | 0.522 | 0.0269 | 384 |
| iBOT Block 1 | 0.954 | 0.0127 | 0.6987 | 0.0228 | 384 |
| iBOT Block 2 | 0.9587 | 0.0105 | 0.7993 | 0.0261 | 384 |
| iBOT Block 3 | 0.952 | 0.0083 | 0.8787 | 0.0144 | 384 |
| iBOT Block 4 | 0.9213 | 0.0027 | 0.9053 | 0.0133 | 384 |
| iBOT Block 5 | 0.8873 | 0.0077 | 0.91 | 0.0202 | 384 |
| iBOT Block 6 | 0.8493 | 0.0124 | 0.922 | 0.0185 | 384 |
| iBOT Block 7 | 0.8407 | 0.0118 | 0.95 | 0.0099 | 384 |
| iBOT Block 8 | 0.76 | 0.0076 | 0.9687 | 0.0054 | 384 |
| iBOT Block 9 | 0.6993 | 0.0197 | 0.972 | 0.0054 | 384 |
| iBOT Block 10 | 0.6613 | 0.0328 | 0.9627 | 0.0053 | 384 |
| iBOT Block 11 | 0.628 | 0.0173 | 0.96 | 0.0087 | 384 |
| VMamba Stage 1 | 0.9441 | 0.0052 | 0.9386 | 0.0107 | 192 |
| VMamba Stage 2 | 0.9341 | 0.0047 | 0.9576 | 0.0084 | 384 |
| VMamba Stage 3 | 0.7834 | 0.0375 | 0.9576 | 0.0075 | 768 |
| VMamba Stage 4 | 0.5414 | 0.0239 | 0.8652 | 0.0257 | 768 |

Table III consolidates the best results obtained by each architecture. For color classification, ResNet-50 and iBOT tied with 95.87% accuracy, followed closely by VGG-16 (95.13%) and VMamba (94.41%). For texture classification, iBOT stood out with 97.20%, surpassing VMamba (95.76%), VGG-16 (94.73%), and ResNet-50 (93.27%). These results indicate that while classical CNNs remain competitive for color recognition, modern architectures—particularly self-supervised Vision Transformers—offer advantages for texture-based tasks.

TABLE III \- Best accuracy achieved by each backbone architecture.

| Backbone | Best\_Color\_Layer | Color | Std\_Color | Best\_Texture\_Layer | Texture | Std\_Texture |
| ----- | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-16 | Layer 1 | 0.9513 | 0.0081 | Layer 4 | 0.9473 | 0.0074 |
| ResNet-50 | Layer 2 | 0.9587 | 0.005 | Layer 3 | 0.9327 | 0.0025 |
| iBOT | Block 2 | 0.9587 | 0.0105 | Block 9 | 0.972 | 0.0054 |
| VMamba | Stage 1 | 0.9441 | 0.0052 | Stage 2 | 0.9576 | 0.0084 |

Figures 8 and 9 illustrate correctly classified images from the texture and color datasets using iBOT. Analysis of classification errors reveals that most misclassifications occur between visually similar classes. In the color dataset, confusions frequently involve adjacent hues in the color spectrum, such as gray versus black or pink versus red, which can appear similar under certain lighting conditions. In the texture dataset, errors arise between patterns with similar structural characteristics at different scales, such as polka dots confused with leopard spots, or sequined fabrics misclassified as leopard print due to their spotted appearance.

![][image8]

Fig. 8\. Correctly classified images from texture dataset using iBOT.

![][image9]
Fig. 9\. Correctly classified images from color dataset using iBOT.

C. Experiment 2: Early-Deep Feature Fusion

This experiment evaluates whether concatenating features from the best color layer (E) and best texture layer (D) improves classification performance. For each backbone, we performed fusion as z = concat(E, D) and evaluated with k-NN (k=5, 5-fold CV).

TABLE IV \- Fusion configuration for each backbone.

| Backbone | E (Color) | Dim E | D (Texture) | Dim D | Dim z |
| ----- | :---: | :---: | :---: | :---: | :---: |
| VGG-16 | Layer 1 | 64 | Layer 4 | 512 | 576 |
| ResNet-50 | Layer 2 | 512 | Layer 3 | 1024 | 1536 |
| iBOT | Block 2 | 384 | Block 9 | 384 | 768 |
| VMamba | Stage 1 | 96 | Stage 2 | 192 | 288 |

TABLE V \- Fusion results for color classification.

| Backbone | E (color) | D (texture) | Z (fusion) | Δ |
| ----- | :---: | :---: | :---: | :---: |
| VGG-16 | **95.13%** | 71.80% | 86.13% | -9.00% |
| ResNet-50 | **95.87%** | 88.27% | 94.13% | -1.73% |
| iBOT | **96.93%** | 81.33% | 85.13% | -11.80% |
| VMamba | **94.80%** | 93.93% | 94.13% | -0.67% |

TABLE VI \- Fusion results for texture classification.

| Backbone | E (color) | D (texture) | Z (fusion) | Δ |
| ----- | :---: | :---: | :---: | :---: |
| VGG-16 | 70.73% | **94.73%** | 94.73% | 0.00% |
| ResNet-50 | 90.67% | 93.27% | **93.53%** | +0.27% |
| iBOT | 78.33% | **87.80%** | 88.07% | +0.27% |
| VMamba | 88.40% | **94.33%** | 93.40% | -0.93% |

The fusion results show that simple concatenation generally does not improve classification performance. For color classification, the fusion consistently degraded accuracy compared to using the specialized color layer alone (E), with iBOT showing the largest degradation (-11.80%). For texture classification, the results were mixed, with minimal improvements for ResNet-50 and iBOT (+0.27%) but degradation for VMamba (-0.93%). These results indicate that specialized single-layer representations are more effective than naive multi-layer fusion.

2. *DISCUSSION*

The experimental results presented in this study provide evidence that the layer-wise sensitivity pattern previously observed in classical CNNs extends to modern architectures based on Vision Transformers and State Space Models. Our experiments reveal a clear pattern regarding how color and texture information distribute across network depth: color information concentrates in early/shallow layers, while texture information concentrates in mid-to-deep layers. Across all four evaluated backbones, early layers consistently captured features more suitable for color classification, while deeper layers proved more effective for texture recognition. This pattern held regardless of the architectural paradigm—whether convolutional (VGG-16, ResNet-50), attention-based (iBOT), or state-space-based (VMamba)—suggesting a fundamental principle of hierarchical feature learning that transcends specific architectural choices.

Regarding the optimal layers for each attribute, the results show that for color classification, early layers perform best: Layer 1-2 in CNNs, Block 2 in iBOT, and Stage 1 in VMamba. For texture classification, deeper layers are more effective: Layer 3-4 in CNNs, Block 9 in iBOT, and Stage 2-3 in VMamba. Notably, the last layer is not optimal for either color or texture classification. All architectures show performance degradation at the final layer, particularly for color classification. This finding challenges the common practice of using only final-layer features for image retrieval.

The performance differences between architectures offer insights into their representational characteristics. iBOT's exceptional texture accuracy (97.20%) suggests that the self-supervised masked image modeling objective encourages the learning of rich structural representations in deeper layers. The relatively uniform feature dimensionality across iBOT blocks (384 dimensions) contrasts with the increasing dimensions in CNNs and VMamba, yet iBOT achieved superior texture discrimination. This indicates that the quality of learned representations, rather than their dimensionality, is the determining factor for texture classification. VMamba's compressed four-stage architecture shows balanced performance for both attributes, with Stage 1 being effective for color (94.41%) and Stage 2 for texture (95.76%). Its linear complexity makes it an attractive option for applications requiring efficiency.

The early-deep fusion experiment revealed that simple concatenation does not improve classification performance. For color classification, fusion consistently degraded accuracy compared to using the specialized color layer alone, with iBOT showing the largest degradation (-11.80%). For texture classification, the results were marginal at best (+0.27% for ResNet-50 and iBOT). These results indicate that specialized single-layer representations are more effective than naive multi-layer fusion, and more sophisticated fusion mechanisms (such as attention-based or learned weighting approaches) may be necessary to benefit from multi-layer features.

The practical implications of these findings are significant for content-based image retrieval systems. Rather than using features from the final layer of a pretrained network—as is common practice—our results suggest that feature extraction should be layer-specific based on the target attribute. For applications requiring color-based retrieval, features from early layers should be preferred, while texture-based retrieval benefits from deeper layer representations. Furthermore, the choice of backbone architecture should consider the primary attribute of interest: classical CNNs offer a good balance for color tasks, while iBOT provides substantial advantages for texture-focused applications.

6. CONCLUSIONS

This study presents a comprehensive layer-wise analysis of feature extraction across multiple deep learning architectures for color and texture classification in fashion images. We evaluated classical CNNs (VGG-16 and ResNet-50), modern architectures (iBOT and VMamba), and the LMFCN framework.

The experiments confirm that the layer-wise sensitivity pattern observed in previous studies extends to modern Vision Transformers and State Space Models: early layers are more effective for color classification, while deeper layers excel at texture recognition. This pattern is consistent across all architectural paradigms.

Key findings include:
- iBOT achieved the highest texture accuracy (97.20% at Block 9)
- ResNet-50 and iBOT tied for best color accuracy (95.87%)
- The last layer is NOT optimal for either attribute—all architectures show degradation at final layers
- Simple early-deep fusion (concatenation) does not improve over single-layer representations
- The layer-wise pattern is consistent across CNN, Transformer, and Mamba architectures

These findings have practical implications for content-based image retrieval systems, suggesting that feature extraction should be attribute-specific rather than relying on final-layer representations. Future work will investigate more sophisticated fusion mechanisms, such as attention-based multi-layer aggregation, to potentially combine the strengths of different layer representations.

Acknowledgment

This work is supported by the Graduate Program in Informatics at the Pontifical Catholic University of Paraná and partially supported by the Brazilian Council for Scientific and Technological Development (CNPQ) linked to the Ministry of Science, Technology and Innovation, to enhance research in Brazil.

References

1. Dubey, S. R. (2021). A decade survey of content based image retrieval using deep learning. IEEE Transactions on Circuits and Systems for Video Technology, 32(5), 2687-2704.

2. Baloian, A., Murrugarra-Llerena, N., Saavedra, J. M. (2021). Scalable visual attribute extraction through hidden layers of a residual convnet. arXiv preprint arXiv:2104.00161.

3. Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., Fei-Fei, L. (2009,

June). Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition (pp. 248- 255). Ieee.

4. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

5. Bromley, J., Guyon, I., LeCun, Y., Sa¨ckinger, E., Shah, R. (1993). Signature verification using a” siamese” time delay neural network. Advances in neural information processing systems, 6\.

6. Chen, X., He, K. (2021). Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 15750-15758).

7. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

8. Moresco, M., Britto, A. D. S., Costa, Y. M., Senger, L. J., Hochuli,

A. G. (2022, October). Combining muti-layer features for plant species classification in a Siamese network. In 2022 IEEE International Confer- ence on Systems, Man, and Cybernetics (SMC) (pp. 2446-2451). IEEE.

9. Arau´jo, V. M., Britto Jr, A. S., Oliveira, L. S., Koerich, A. L. (2022). Two-view fine-grained classification of plant species. Neurocomputing, 467, 427-441.

10. Liang, K., Chang, H., Shan, S., Chen, X. (2015). A unified multi- plicative framework for attribute learning. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2506-2514).

11. Gan, C., Yang, T., Gong, B. (2016). Learning attributes equals multi- source domain generalization. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 87-97).

12. Youssef, S. M. (2012). ICTEDCT-CBIR: Integrating curvelet transform with enhanced dominant colors extraction and texture analysis for effi- cient content-based image retrieval. Computers Electrical Engineering, 38(5), 1358-1376.

13. Irtaza, A., Jaffar, M. A., Aleisa, E., Choi, T. S. (2014). Embedding neu- ral networks for semantic association in content based image retrieval. Multimedia tools and applications, 72, 1911-1931.

14. Atlam, H. F., Attiya, G., El-Fishawy, N. (2017). Integration of color and texture features in CBIR system. Int. J. Comput. Appl, 164(3), 23-29.

15. Lin, C. H., Chen, R. T., Chan, Y. K. (2009). A smart content-based image retrieval system based on color and texture feature. Image and vision Computing, 27(6), 658-665.

16. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752. Retrieved from https://arxiv.org/abs/2312.00752

17. Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., & Kong, T. (2021). iBOT: Image BERT pre-training with online tokenizer. arXiv preprint arXiv:2111.07832. Retrieved from https://arxiv.org/abs/2111.07832

18. Liang, K., Chang, H., Shan, S., Chen, X. (2015). A unified multi- plicative framework for attribute learning. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2506-2514).

19. Gan, C., Yang, T., Gong, B. (2016). Learning attributes equals multi- source domain generalization. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 87-97).

20. Youssef, S. M. (2012). ICTEDCT-CBIR: Integrating curvelet transform with enhanced dominant colors extraction and texture analysis for effi- cient content-based image retrieval. Computers Electrical Engineering, 38(5), 1358-1376.

21. Irtaza, A., Jaffar, M. A., Aleisa, E., Choi, T. S. (2014). Embedding neu- ral networks for semantic association in content based image retrieval. Multimedia tools and applications, 72, 1911-1931.

22. Atlam, H. F., Attiya, G., El-Fishawy, N. (2017). Integration of color and texture features in CBIR system. Int. J. Comput. Appl, 164(3), 23-29.

23. Mukherjee, D., Mondal, R., Singh, P. K., Sarkar, R., & Bhattacharjee, D. (2020). EnsembleNet: A hybrid approach for vehicle detection and estimation of traffic density based on faster R-CNN and YOLO models. Neural Computing and Applications, 32(15), 14207-14228.

\[24\] Yue, J., Li, Z., Liu, L., & Fu, Z. (2011). Content-based image retrieval using color and texture. In 2011 Sixth International Conference on Image and Graphics (pp. 833-837). IEEE.

\[25\] Prakasa, E. (2016). Texture Feature Extraction by Using Local Binary Pattern. *INKOM Journal*, *1*(1), 1-6.

\[26\] Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 3431-3440).

\[27\] Guo, Y., Liu, Y., Georgiou, T., & Lew, M. S. (2018). A review of semantic segmentation using deep neural networks. International Journal of Multimedia Information Retrieval, 7(2), 87-93.

\[26\] Naveen, K., V, S., & Kumar, A. (2025). AI powered multi feature fusion framework for retrieving content based medical images. *Scientific Reports*, *15*(1), 1-14.

\[28\] de Matos, J., de Oliveira, L. E. S., Junior, A. D. S. B., & Koerich, A. L.  (2023). Large-margin representation learning for texture classification. Pattern Recognition Letters, 170, 39-47.

24. 









