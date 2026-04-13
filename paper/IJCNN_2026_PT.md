# Avaliação de um Modelo de Métrica Profunda para Recuperação de Padrões Baseada em Características de Cor e Textura

## Arlete Teresinha Beuren
*Departamento de Ciência da Computação, Universidade Tecnológica Federal do Paraná (UTFPR)*
Santa Helena (PR), Brasil - arletebeuren@utfpr.edu.br

## Vinícius Cerqueira Ribeiro
*Departamento de Ciência da Computação, Universidade Tecnológica Federal do Paraná (UTFPR)*
Santa Helena (PR), Brasil - viniciuscerqueira@alunos.utfpr.edu.br

## Leonardo Jose Reis Pinto
*Departamento de Ciência da Computação, Universidade Tecnológica Federal do Paraná (UTFPR)*
Santa Helena (PR), Brasil - leonardopinto@alunos.utfpr.edu.br

## Alceu de Souza Britto Jr.
*Departamento de Ciência da Computação, Pontifícia Universidade Católica do Paraná (PUCPR)*
Curitiba (PR), Brasil - alceu@ppgia.pucpr.br

## Jose Saavedra
*Universidad de Los Andes*
Santiago, Chile

## Jonathan xxxx
*Departamento de Ciência da Computação, Universidade Tecnológica Federal do Paraná (UTFPR)*
Santa Helena (PR), Brasil

---

**Resumo** — Este estudo aborda o desafio de extrair efetivamente o conteúdo de imagens com base em atributos específicos: cor, textura ou sua combinação — uma área de particular relevância para aplicações de e-commerce. São avaliados modelos de aprendizado profundo para compreender a sensibilidade de suas camadas a esses atributos. Experimentos foram conduzidos para avaliar representações específicas por camada e determinar sua eficácia na captura de características de cor, textura e combinadas. O estudo também explora a adequação de ResNet-50, VGG-16, iBOT, VMamba e LMFCN como arquiteturas backbone. Os resultados indicam que as camadas iniciais são mais eficazes para classificação de cor, enquanto camadas mais profundas se destacam no reconhecimento de textura. O iBOT alcançou a maior acurácia em textura (97,20%), enquanto ResNet-50 e iBOT empataram na melhor acurácia de cor (95,87%). Adicionalmente, investigou-se se a fusão de características early-deep melhora o desempenho, concluindo que a concatenação simples não supera representações especializadas de camada única.

*Palavras-chave* — Redes Neurais Convolucionais, Métricas Profundas, Atributos, Vision Transformers, Modelos de Espaço de Estados

---

## 1. INTRODUÇÃO

Os avanços em redes convolucionais possibilitaram soluções de visão computacional cada vez mais viáveis para aplicações industriais. Um exemplo notável é a recuperação de imagens em e-commerce, onde buscas podem ser realizadas usando texto, imagens ou uma combinação dos principais componentes de sistemas inovadores. Embora a busca baseada em texto permaneça como o método de consulta mais comum para mecanismos de busca, sua eficácia depende fortemente de descrições detalhadas dos produtos. Para melhorar a experiência do usuário, alguns mecanismos de busca incorporam pesquisa baseada em imagem, oferecendo resultados significativamente mais precisos [1].

Mecanismos de busca baseados em conteúdo de imagem apresentam desafios únicos e encontram aplicações além do e-commerce, incluindo recuperação de imagens médicas, sistemas de informação geográfica, vigilância e outros [12]. O principal desafio reside em extrair com precisão o conteúdo relevante de uma imagem, particularmente quando o interesse do usuário é baseado em atributos específicos como cor, textura ou sua combinação (Figura 1). No contexto do e-commerce, este tipo de busca focada em atributos está cada vez mais em demanda.

*Fig. 1. Seis diferentes padrões baseados em cor e textura (adaptado de [2]).*

A extração de características de cor e textura tem sido um campo fundamental de estudo em Recuperação de Imagens Baseada em Conteúdo (CBIR). Métodos clássicos para extração de cor incluem histogramas de cor, que quantificam a distribuição de cores em uma imagem, e momentos de cor, que capturam informações espaciais como média, variância e assimetria da distribuição de cores [24]. Para análise de textura, técnicas como filtros de Gabor, que analisam a imagem em diferentes escalas e orientações, e Local Binary Pattern (LBP), que codifica padrões locais de textura de forma robusta a variações de iluminação, são amplamente utilizadas [25].

Com o advento do deep learning, as Redes Neurais Convolucionais (CNNs) tornaram-se o padrão para extração de características. Estudos como o de Baloian et al. [2] investigaram como diferentes camadas de uma ResNet-50 respondem a atributos de cor e textura, concluindo que camadas iniciais são mais sensíveis à cor, enquanto camadas mais profundas capturam melhor a textura. Outras pesquisas propõem a fusão de características de cor e textura extraídas por CNNs para criar um vetor de características mais rico e discriminativo [26]. Este trabalho se insere neste contexto, avaliando e comparando arquiteturas de deep learning, desde CNNs clássicas até os modelos mais recentes baseados em Transformers e Modelos de Espaço de Estados, para a tarefa de recuperação de imagens baseada em atributos de cor e textura.

O conjunto de dados utilizado neste estudo foi criado e disponibilizado por [2], aproveitando imagens provenientes da plataforma Kaggle e outros repositórios online. Ele apresenta uma coleção diversificada de itens de vestuário, incluindo chapéus, vestidos, camisas, meias, bolsas e cachecóis. O conjunto de dados está organizado em dois subconjuntos distintos. O primeiro subconjunto foca na categorização de imagens por cor, com as seguintes classes: vermelho, preto, azul, verde, amarelo, cinza, marrom, rosa, roxo e laranja. Cada classe contém 100 imagens, resultando em um total de 1.000 imagens no conjunto de dados de cor, conforme ilustrado na Figura 2.

*Fig. 2. Amostras de imagens de cada classe no conjunto de dados de cor.*

O segundo conjunto de dados categoriza imagens com base em sua textura, com as seguintes classes: quadriculado, listrado, flores, leopardo, bolinhas, básico, paisley, argyle, pé de galinha e lantejoulas. Da mesma forma, cada classe de textura contém 100 imagens, resultando em um total de 1.000 imagens no conjunto de dados de textura. Com base neste problema, o objetivo é avaliar um modelo de métrica profunda e a sensibilidade de suas camadas para busca apenas por cor e textura. Considerando o desafio proposto, este estudo avalia alguns modelos citados na literatura. A intenção é descobrir como diferentes camadas se comportam em relação a diferentes atributos, cores e texturas, realizando diferentes experimentos para determinar quais camadas ocultas fornecem a melhor representação para cada um e sua combinação. Ou seja, avaliar quais camadas devem ser consideradas para representar cada atributo e sua combinação e, finalmente, analisar o comportamento das arquiteturas a serem avaliadas. As arquiteturas selecionadas incluem ResNet-50 [4], VGG-16 [7], VMamba [16], iBOT [17] e LMFCN [28].

Os principais objetivos deste estudo são: (1) identificar qual arquitetura apresenta melhor desempenho, avaliando uma arquitetura mais simples como VGG-16, um modelo mais complexo como ResNet-50 e modelos mais recentes; e (2) determinar quais camadas dentro de cada modelo fornecem a melhor representação de atributos de cor e textura. Adicionalmente, investiga-se se a fusão de características de camadas especializadas melhora o desempenho de classificação.

Este estudo está organizado da seguinte forma: a Seção 2 revisa trabalhos relacionados, a Seção 3 detalha o método proposto, a Seção 4 discute os experimentos conduzidos, a Seção 5 apresenta os resultados experimentais e a Seção 6 apresenta as conclusões.

## 2. TRABALHOS RELACIONADOS

O uso de redes convolucionais para buscar representar a similaridade de objetos é amplamente utilizado na literatura. Mecanismos de busca usam pesquisas de texto como seu principal meio de consulta, mas pesquisas usando consultas de imagem estão avançando e trazendo resultados interessantes para este campo de pesquisa, como apresentado pelos autores em [2]. Além de usar imagens, eles investigaram as camadas da ResNet-50 para determinar a busca por cores e texturas. Em sua pesquisa, os autores concluem que as camadas iniciais da rede são melhores para cores e as camadas mais profundas para texturas. Na pesquisa, características são extraídas com ResNet-50 pré-treinada com o conjunto de dados Imagenet [3]. Na extração de características, a primeira camada convolucional é renomeada como Bloco 1, e os blocos residuais 2 a 5. Na saída de cada bloco, uma camada de Global Average Pooling (GAP) foi aplicada e as saídas dessas camadas foram usadas como entrada para um classificador k-nearest neighbor (KNN) com validação cruzada de 5 folds.

Em [10], os autores empregaram redes neurais para aprender um conjunto de características usando categorias de objetos. Em [11] os autores propõem técnicas de generalização de domínio multi-fonte para aprendizado cruzado. Os autores em [8] classificam espécies de plantas por imagem de folha usando a arquitetura de redes neurais siamesas (SNN's), combinando características de camadas convolucionais intermediárias para melhorar representações. De acordo com os autores, a combinação de características de diferentes camadas apresenta um ganho de desempenho relevante e diferentes camadas trazem resultados diferentes. No estudo de [9], os autores usaram uma rede neural siamesa para definir o índice de similaridade das imagens de cada classe através de uma análise da folha inteira e partes dela. Em [13], os autores apresentam um método eficiente de extração de características baseado no conceito de análise de textura em profundidade. Para isso, pacotes wavelet e autovalores de filtros de Gabor são usados para fins de representação de imagem e um esquema de aprendizado parcial supervisionado baseado em K-nearest neighbors. Estudos também propõem a combinação de características de cor e textura como vetor de características [14]. [15] propôs três características de conteúdo de imagem baseadas em cor, textura e distribuição de cor, como matriz de co-ocorrência de cor (CCM), diferença entre pixels do padrão de varredura (DBPSP) e histograma de cor para K-mean (CHKM) respectivamente.

A fusão de características de múltiplas camadas tem sido explorada como estratégia para combinar representações de diferentes profundidades de redes neurais. A intuição é que camadas iniciais capturam características de baixo nível (bordas, cores) enquanto camadas mais profundas codificam informações semânticas de alto nível, e combiná-las poderia produzir representações mais ricas [26]. Estratégias comuns de fusão incluem fusão precoce (concatenando características antes da classificação), fusão tardia (combinando saídas de classificadores) e fusão aprendida usando mecanismos de atenção [27]. No entanto, a eficácia de tais abordagens para tarefas específicas de atributos como classificação de cor e textura permanece pouco explorada.

Avanços recentes em deep learning introduziram novas arquiteturas que desafiam a dominância dos Transformers. O modelo Mamba, proposto por Gu e Dao [16], é um exemplo notável. É um Modelo de Espaço de Estados (SSM) que alcança complexidade de tempo linear na modelagem de sequências, oferecendo uma vantagem computacional significativa sobre a complexidade quadrática dos Transformers. Os SSMs seletivos do Mamba permitem filtrar e recuperar informações com base no conteúdo, uma capacidade anteriormente considerada uma força chave dos mecanismos de atenção. Isso torna o Mamba uma alternativa promissora para várias modalidades, incluindo linguagem, áudio e genômica, onde demonstrou desempenho estado-da-arte.

No domínio do aprendizado auto-supervisionado para visão, o modelo iBOT (Image BERT Pre-Training with Online Tokenizer), introduzido por Zhou et al. [17], fez contribuições significativas. O iBOT utiliza uma abordagem de modelagem de imagem mascarada (MIM), semelhante à modelagem de linguagem mascarada do BERT, mas com uma inovação chave: um tokenizador online. Isso permite que o modelo aprenda um tokenizador visual semanticamente significativo simultaneamente com o objetivo MIM, eliminando a necessidade de um estágio de pré-treinamento separado para o tokenizador. A integração do iBOT com camadas convolucionais permite uma combinação poderosa de extração de características locais, característica das CNNs, com a compreensão de contexto global da arquitetura iBOT baseada em Transformer. Esta abordagem híbrida mostrou forte desempenho em tarefas downstream densas como detecção de objetos e segmentação semântica. A Large Margin Fully Convolutional Network (LMFCN) foi projetada para treinar Redes Totalmente Convolucionais (FCNs) cujas saídas são subsequentemente usadas como características de entrada para um classificador de grande margem, especificamente uma Máquina de Vetores de Suporte (SVM). Durante o treinamento, o método atualiza os pesos da FCN para gerar representações de características que maximizam a margem entre classes, melhorando assim a precisão de classificação do SVM.

## 3. METODOLOGIA APLICADA

Este estudo avalia diferentes arquiteturas de deep learning para identificar o modelo mais eficaz para recuperação de objetos baseada em cor e textura. Os modelos avaliados incluem duas Redes Neurais Convolucionais (CNNs) clássicas, ResNet-50 e VGG-16, e três arquiteturas de estudos recentes, VMamba, iBOT e LMFCN, que trazem novos paradigmas para modelagem de sequências e imagens.

### A. ResNet-50

A Residual Network 50 (ResNet-50), introduzida por He et al. [4], é uma rede neural convolucional profunda de 50 camadas. Sua inovação central é o uso de "blocos residuais" com conexões de atalho (skip connections), que permitem ao modelo aprender funções residuais. Esta arquitetura aborda efetivamente o problema do gradiente desvanecente em redes muito profundas, permitindo o treinamento de modelos com centenas ou até milhares de camadas mantendo alto desempenho. A ResNet-50 é amplamente usada como backbone para várias tarefas de visão computacional devido às suas poderosas capacidades de extração de características (Figura 3).

*Fig. 3. Arquitetura do modelo ResNet-50. Adaptado de Mukherjee [23], baseado no trabalho original de He et al. [4].*

### B. VGG-16

O modelo VGG-16, proposto por Simonyan e Zisserman, é uma rede neural convolucional caracterizada por sua simplicidade e profundidade [7]. É composto por 16 camadas, utilizando principalmente pequenos filtros convolucionais 3x3 empilhados uns sobre os outros. Este empilhamento de pequenos filtros permite ao modelo ter um grande campo receptivo mantendo um baixo número de parâmetros. Apesar de sua arquitetura relativamente simples, a VGG-16 é conhecida por seu excelente desempenho em classificação de imagens e suas características provaram ser altamente transferíveis para outras tarefas (Figura 4).

*Fig. 4. Arquitetura do modelo VGG-16, ilustrando sua estrutura sequencial. Fonte: Simonyan & Zisserman [7].*

### C. VMamba

O VMamba, desenvolvido por Gu e Dao [16], representa uma nova classe de modelos de sequência baseados em Modelos de Espaço de Estados Estruturados (SSMs). Diferente dos Transformers, que têm complexidade quadrática em relação ao comprimento da sequência, o Mamba processa sequências em tempo linear. Ele emprega um mecanismo de varredura seletiva que permite ao modelo propagar ou esquecer seletivamente informações ao longo de uma sequência, permitindo raciocínio baseado em conteúdo. Isso torna o Mamba altamente eficiente para sequências longas e um forte executor em diferentes modalidades, incluindo visão, onde pode ser adaptado como backbone para extração de características (Figura 5).

*Fig. 5. Diagrama de blocos do Mamba, ilustrando o mecanismo SSM seletivo. Fonte: Gu & Dao [16].*

### D. iBOT com Camadas Convolucionais

O modelo iBOT (Image BERT Pre-Training with Online Tokenizer), de Zhou et al. [17], é um framework de aprendizado auto-supervisionado baseado em Vision Transformers (ViT). Ele usa um objetivo de Modelagem de Imagem Mascarada (MIM) onde partes de uma imagem são mascaradas, e o modelo deve prever o conteúdo mascarado. Uma característica chave do iBOT é seu tokenizador online, que aprende a criar tokens visuais semanticamente ricos durante o processo de pré-treinamento. Para este estudo, o iBOT é combinado com camadas convolucionais, criando um modelo híbrido. Esta abordagem aproveita as forças das CNNs na captura de características locais e detalhes finos, enquanto o componente Transformer modela o contexto global e relacionamentos entre patches de imagem (Figura 6).

*Fig. 6. Visão geral do framework iBOT, mostrando a arquitetura com um tokenizador online. Fonte: Zhou et al. [17].*

### E. LMFCN

*Fig. 7. Visão geral da LMFCN. Fonte: de Matos et al. [28].*

A arquitetura LMFCN é apresentada na Fig. 7. Ela recebe como entrada imagens do conjunto de treinamento T, extrai representações de características usando uma FCN, e usa essas características para construir a matriz de kernel K para um classificador SVM. A FCN pode ser qualquer arquitetura convolucional capaz de produzir uma representação latente dos dados, como uma TCNN [28], ResNet, VGG ou rede Inception. A matriz D é uma matriz de distância definida usando a mesma métrica que a matriz de kernel K e é empregada para medir distâncias entre instâncias de treinamento e os vetores de suporte do SVM. Com base nos resultados de classificação do SVM, a matriz de distância e os vetores de suporte, os parâmetros da FCN são atualizados usando uma função de perda que aumenta a penalidade para (i) instâncias que estão longe do vetor de suporte mais próximo da mesma classe, (ii) instâncias que estão próximas de vetores de suporte da classe oposta, e (iii) instâncias que estão próximas de instâncias não-suporte da classe oposta. A sequência de operações descrita acima define uma única época de treinamento. Este método foi projetado principalmente para treinar FCNs pequenas e leves em conjuntos de dados de tamanho limitado, particularmente em tarefas de classificação relacionadas a textura.

## 4. MODELOS BACKBONE AVALIADOS

Neste estudo, modelos clássicos e modernos foram selecionados. A VGG-16 é uma CNN clássica com 16 camadas ponderadas organizadas em 5 blocos convolucionais usando filtros 3×3, cada um seguido por max-pooling, com dimensões de características de 64, 128, 256, 512 e 512. A ResNet-50 introduziu conexões residuais (skip connections) permitindo redes mais profundas, consistindo em 4 blocos residuais principais com dimensões de 256, 512, 1024 e 2048. Características são extraídas de ambas as CNNs usando Global Average Pooling (GAP).

Para arquiteturas modernas, o iBOT (Image BERT Pre-Training with Online Tokenizer) é um Vision Transformer pré-treinado usando modelagem de imagem mascarada auto-supervisionada, com a variante ViT-S/16 tendo 12 blocos transformer produzindo embeddings de 384 dimensões extraídos via token CLS. O VMamba adapta a arquitetura Mamba — um modelo seletivo de espaço de estados — para tarefas visuais, alcançando complexidade linear diferentemente dos transformers; a variante VMamba-Tiny tem 4 estágios com dimensões de 96, 192, 384 e 768. Adicionalmente, o framework LMFCN foi avaliado usando arquiteturas TCNN e ResNet-18.

Ambas as CNNs são citadas em diferentes trabalhos. Para o estudo, as saídas das cinco camadas iniciais treinadas com ImageNet foram avaliadas com CNN. A escolha dessas camadas é baseada em estudos publicados na literatura, [2] e [8], que detectaram resultados diferentes em cada uma dessas camadas. Testes iniciais permitiram identificar quais camadas são melhores para cor e textura para cada uma dessas CNNs. Características foram extraídas usando Global Average Pooling na saída de cada camada. Os vetores gerados são alimentados ao algoritmo kNN (K-nearest neighbors) usando validação cruzada de 5 folds. Global Average Pooling é uma função para reduzir o tamanho espacial da representação para reduzir o número de parâmetros e computação na rede. É principalmente usado para reduzir as dimensões do mapa de características. Após descobrir a melhor camada de cor e textura para os modelos com as CNNs propostas, as melhores camadas também são úteis para comparar com a aplicação normal de CNNs. Para clareza, nomeamos camada 1 a camada 5 como CNNLayer1, CNNLayer2, CNNLayer3, CNNLayer4 e CNNLayer5, para cada CNN.

O LMFCN foi avaliado usando duas arquiteturas de rede totalmente convolucional (FCN): uma rede neural convolucional de textura (TCNN) com três, quatro, cinco e seis camadas, e uma ResNet-18 pré-treinada com as camadas totalmente conectadas finais removidas. Para a arquitetura ResNet-18, duas configurações foram examinadas: uma compreendendo todos os blocos convolucionais e outra retendo apenas os dois primeiros blocos. As FCNs foram treinadas usando o framework LMFCN, e a precisão de classificação em cada camada FCN foi avaliada usando o método de avaliação k-nearest neighbors (kNN) descrito anteriormente. Estas arquiteturas foram selecionadas de acordo com o objetivo de design do LMFCN de integração eficaz com redes convolucionais pequenas e leves.

## 5. RESULTADOS EXPERIMENTAIS

Esta seção apresentará o banco de dados e experimentos realizados para responder aos objetivos propostos.

### A. Protocolo Experimental

O conjunto de dados usado nesta pesquisa foi o mesmo usado pelos autores em [2], que consiste em 10 classes para cor, cada uma contendo 100 imagens, e 10 classes para textura, também com 100 imagens cada, totalizando 1000 imagens para o conjunto de cor e 1000 imagens para o conjunto de textura. A base contém dois conjuntos de dados usando Kaggle, um agrupado por cor e outro por textura. A descrição desses conjuntos de dados é dada abaixo:

(a) Conjunto de dados de cor: Contém 100 imagens de roupas em cada uma das seguintes cores: vermelho, preto, azul, verde, amarelo, cinza, marrom, rosa, roxo e laranja;

(b) Conjunto de dados de textura: Contém 100 imagens de roupas de cada um dos seguintes padrões de textura: quadrado, listrado, flores, leopardo, bolinhas, básico, paisley, argyle, pé de galinha e lantejoulas.

O conjunto de dados é compilado de várias peças de vestuário, variando de meias a camisas. O banco de dados foi dividido em 70% para treinamento e 30% para teste.

### B. Resultados com LMFCN

A Tabela I apresenta os resultados obtidos no conjunto de dados de textura, indicando que aumentar a profundidade da TCNN leva a um melhor desempenho, com a precisão aumentando consistentemente à medida que a profundidade da FCN aumenta. Em contraste, a configuração ResNet-18 incluindo todos os quatro blocos convolucionais produziu desempenho kNN inferior em dados de textura. Os resultados obtidos no conjunto de dados de cor indicam que, em comparação com o conjunto de dados de textura, maior precisão de classificação é alcançada das camadas iniciais da rede. Na configuração ResNet-18 com quatro blocos convolucionais, uma degradação na precisão foi observada na terceira camada. O LMFCN com seu classificador SVM obteve o melhor desempenho para o conjunto de dados de textura para a ResNet-18 com dois blocos, alcançando 96,87% de precisão e para o conjunto de dados de cor com a TCNN com três camadas, com precisão de 97,73%.

**TABELA I** - Acurácia das camadas ocultas de uma ResNet-18 e uma TCNN usando LMFCN para treinamento e ajuste fino

| Método | Cor (%) | Textura (%) |
|--------|:-------:|:-----------:|
| TCNN3 Camada 1 | 0,9333 | 0,4873 |
| TCNN3 Camada 2 | 0,9513 | 0,5513 |
| TCNN3 Camada 3 | 0,9594 | 0,5827 |
| TCNN4 Camada 1 | 0,9267 | 0,4867 |
| TCNN4 Camada 2 | 0,9434 | 0,5473 |
| TCNN4 Camada 3 | 0,9420 | 0,5973 |
| TCNN4 Camada 4 | 0,9480 | 0,6040 |
| TCNN5 Camada 1 | 0,9326 | 0,4867 |
| TCNN5 Camada 2 | 0,9487 | 0,5533 |
| TCNN5 Camada 3 | 0,9433 | 0,6080 |
| TCNN5 Camada 4 | 0,9467 | 0,6227 |
| TCNN5 Camada 5 | 0,9540 | 0,6293 |
| TCNN6 Camada 1 | 0,9287 | 0,4933 |
| TCNN6 Camada 2 | 0,9467 | 0,5647 |
| TCNN6 Camada 3 | 0,9473 | 0,6087 |
| TCNN6 Camada 4 | 0,9453 | 0,6347 |
| TCNN6 Camada 5 | 0,9433 | 0,6327 |
| TCNN6 Camada 6 | 0,9453 | 0,6434 |
| ResNet18 (2 blocos) Camada 1 | 0,9500 | 0,6273 |
| ResNet18 (2 blocos) Camada 2 | 0,9613 | 0,8487 |
| ResNet18 (2 blocos) Camada 3 | 0,9193 | 0,9100 |
| ResNet18 (4 blocos) Camada 1 | 0,9453 | 0,6367 |
| ResNet18 (4 blocos) Camada 2 | 0,9593 | 0,8413 |
| ResNet18 (4 blocos) Camada 3 | 0,9107 | 0,9120 |
| ResNet18 (4 blocos) Camada 4 | 0,8953 | 0,9247 |
| ResNet18 (4 blocos) Camada 5 | 0,7520 | 0,8674 |

### C. Resultados Experimentais - CNNs Clássicas

A Tabela II apresenta a acurácia de classificação obtida para as camadas ocultas da VGG-16 e ResNet-50. Características foram extraídas usando Global Average Pooling (GAP) e avaliadas com k-NN usando validação cruzada de 5 folds. Os resultados revelam uma relação clara entre profundidade da camada e sensibilidade ao atributo. Na VGG-16, a Camada 1 alcançou a maior acurácia de cor (95,13%), com o desempenho diminuindo progressivamente em camadas mais profundas até alcançar 55,07% na Camada 5. Por outro lado, a acurácia de textura melhorou com a profundidade, atingindo o pico na Camada 4 (94,73%). A ResNet-50 exibiu comportamento similar: a Camada 2 obteve o melhor resultado de cor (95,87%), enquanto a Camada 3 foi mais eficaz para textura (93,27%). A queda de acurácia nas camadas finais de ambas as redes, particularmente para classificação de cor, sugere que características de baixo nível capturadas em camadas iniciais são mais relevantes para distinguir atributos cromáticos.

**TABELA II** - Acurácia das camadas ocultas da VGG-16 e ResNet-50 para classificação de cor e textura

| Camada | Cor | Std_Cor | Textura | Std_Textura | Dim |
|--------|:---:|:-------:|:-------:|:-----------:|:---:|
| VGG-16 Camada 1 | 0.9513 | 0.0081 | 0.7073 | 0.0225 | 64 |
| VGG-16 Camada 2 | 0.9027 | 0.0127 | 0.8660 | 0.0157 | 128 |
| VGG-16 Camada 3 | 0.8547 | 0.0233 | 0.9320 | 0.0148 | 256 |
| VGG-16 Camada 4 | 0.7180 | 0.0255 | 0.9473 | 0.0074 | 512 |
| VGG-16 Camada 5 | 0.5507 | 0.0264 | 0.8940 | 0.0164 | 512 |
| ResNet-50 Camada 1 | 0.9547 | 0.0078 | 0.8340 | 0.0136 | 256 |
| ResNet-50 Camada 2 | 0.9587 | 0.0050 | 0.9067 | 0.0165 | 512 |
| ResNet-50 Camada 3 | 0.8827 | 0.0112 | 0.9327 | 0.0025 | 1024 |
| ResNet-50 Camada 4 | 0.5920 | 0.0311 | 0.8633 | 0.0092 | 2048 |

### D. Resultados - Arquiteturas Modernas

A avaliação foi estendida a arquiteturas modernas para verificar se este padrão por camada se generaliza além das CNNs tradicionais. A Tabela III mostra os resultados para iBOT e VMamba. Para iBOT, características foram extraídas de cada bloco transformer via token CLS, enquanto para VMamba, características foram obtidas da saída de cada estágio de processamento. O modelo iBOT demonstrou um gradiente particularmente pronunciado: a acurácia de cor diminuiu de 94,53% no Bloco 0 para 62,80% no Bloco 11, enquanto a acurácia de textura aumentou de 52,20% para 97,20% no Bloco 9 antes de diminuir ligeiramente nos blocos finais. Esta acurácia de 97,20% em textura representa o maior valor entre todas as arquiteturas avaliadas, indicando que Vision Transformers auto-supervisionados capturam representações especialmente discriminativas para padrões de textura. O VMamba mostrou um comportamento mais comprimido devido à sua arquitetura de quatro estágios, com o Estágio 1 alcançando a melhor acurácia de cor (94,41%) e os Estágios 2-3 alcançando a melhor acurácia de textura (95,76%).

**TABELA III** - Acurácia dos blocos iBOT e estágios VMamba para classificação de cor e textura

| Camada | Cor | Std_Cor | Textura | Std_Textura | Dim |
|--------|:---:|:-------:|:-------:|:-----------:|:---:|
| iBOT Bloco 0 | 0.9453 | 0.0113 | 0.5220 | 0.0269 | 384 |
| iBOT Bloco 1 | 0.9540 | 0.0127 | 0.6987 | 0.0228 | 384 |
| iBOT Bloco 2 | 0.9587 | 0.0105 | 0.7993 | 0.0261 | 384 |
| iBOT Bloco 3 | 0.9520 | 0.0083 | 0.8787 | 0.0144 | 384 |
| iBOT Bloco 4 | 0.9213 | 0.0027 | 0.9053 | 0.0133 | 384 |
| iBOT Bloco 5 | 0.8873 | 0.0077 | 0.9100 | 0.0202 | 384 |
| iBOT Bloco 6 | 0.8493 | 0.0124 | 0.9220 | 0.0185 | 384 |
| iBOT Bloco 7 | 0.8407 | 0.0118 | 0.9500 | 0.0099 | 384 |
| iBOT Bloco 8 | 0.7600 | 0.0076 | 0.9687 | 0.0054 | 384 |
| iBOT Bloco 9 | 0.6993 | 0.0197 | 0.9720 | 0.0054 | 384 |
| iBOT Bloco 10 | 0.6613 | 0.0328 | 0.9627 | 0.0053 | 384 |
| iBOT Bloco 11 | 0.6280 | 0.0173 | 0.9600 | 0.0087 | 384 |
| VMamba Estágio 1 | 0.9441 | 0.0052 | 0.9386 | 0.0107 | 192 |
| VMamba Estágio 2 | 0.9341 | 0.0047 | 0.9576 | 0.0084 | 384 |
| VMamba Estágio 3 | 0.7834 | 0.0375 | 0.9576 | 0.0075 | 768 |
| VMamba Estágio 4 | 0.5414 | 0.0239 | 0.8652 | 0.0257 | 768 |

A Figura 10 ilustra o comportamento de cada arquitetura em relação à profundidade da camada, mostrando claramente o padrão inverso entre cor e textura. Em todas as arquiteturas, observa-se que a acurácia de cor (linha azul) tende a diminuir com a profundidade, enquanto a acurácia de textura (linha vermelha) aumenta até atingir um pico em camadas intermediárias ou profundas.

*Fig. 10. Performance × Profundidade: Classificação de Cor e Textura para VGG-16, ResNet-50, iBOT e VMamba. Os marcadores indicam as melhores camadas para cada atributo (E=cor, D=textura).*

![Performance × Profundidade](curvas_desempenho_profundidade.png)

### E. Resumo dos Melhores Resultados

A Tabela IV consolida os melhores resultados obtidos por cada arquitetura. Para classificação de cor, ResNet-50 e iBOT empataram com 95,87% de acurácia, seguidos de perto por VGG-16 (95,13%) e VMamba (94,41%). Para classificação de textura, o iBOT se destacou com 97,20%, superando VMamba (95,76%), VGG-16 (94,73%) e ResNet-50 (93,27%). Estes resultados indicam que enquanto CNNs clássicas permanecem competitivas para reconhecimento de cor, arquiteturas modernas — particularmente Vision Transformers auto-supervisionados — oferecem vantagens para tarefas baseadas em textura.

**TABELA IV** - Melhor acurácia alcançada por cada arquitetura backbone

| Backbone | Melhor_Camada_Cor | Cor | Std_Cor | Melhor_Camada_Textura | Textura | Std_Textura |
|----------|:-----------------:|:---:|:-------:|:---------------------:|:-------:|:-----------:|
| VGG-16 | Camada 1 | 0.9513 | 0.0081 | Camada 4 | 0.9473 | 0.0074 |
| ResNet-50 | Camada 2 | 0.9587 | 0.0050 | Camada 3 | 0.9327 | 0.0025 |
| iBOT | Bloco 2 | 0.9587 | 0.0105 | Bloco 9 | 0.9720 | 0.0054 |
| VMamba | Estágio 1 | 0.9441 | 0.0052 | Estágio 2 | 0.9576 | 0.0084 |

A Figura 11 apresenta uma comparação direta entre todas as arquiteturas, com a profundidade normalizada (0=camada inicial, 1=camada profunda) para permitir comparação entre modelos com diferentes números de camadas. Observa-se que todas as arquiteturas seguem o mesmo padrão geral, com o iBOT apresentando o gradiente mais pronunciado.

*Fig. 11. Comparação entre Arquiteturas: Acurácia vs Profundidade Normalizada para classificação de cor (esquerda) e textura (direita).*

![Comparação entre Arquiteturas](curvas_comparacao_arquiteturas.png)

As Figuras 8 e 9 ilustram imagens corretamente classificadas dos conjuntos de dados de textura e cor usando iBOT. A análise dos erros revela que a maioria das classificações incorretas ocorre entre classes visualmente similares. No conjunto de dados de cor, confusões frequentemente envolvem matizes adjacentes no espectro de cores, como cinza versus preto ou rosa versus vermelho, que podem parecer similares sob certas condições de iluminação. No conjunto de dados de textura, erros surgem entre padrões com características estruturais similares em diferentes escalas, como bolinhas confundidas com manchas de leopardo, ou tecidos com lantejoulas classificados incorretamente como estampa de leopardo devido à sua aparência pontilhada.

*Fig. 8. Imagens corretamente classificadas do conjunto de dados de textura usando iBOT.*

*Fig. 9. Imagens corretamente classificadas do conjunto de dados de cor usando iBOT.*

### F. Experimento 2: Fusão de Características Early-Deep

Este experimento avalia se concatenar características da melhor camada de cor (E) e da melhor camada de textura (D) melhora o desempenho de classificação. Para cada backbone, realizamos fusão como z = concat(E, D) e avaliamos com k-NN (k=5, validação cruzada 5-fold).

**TABELA V** - Configuração de fusão para cada backbone

| Backbone | E (Cor) | Dim E | D (Textura) | Dim D | Dim z |
|----------|:-------:|:-----:|:-----------:|:-----:|:-----:|
| VGG-16 | Camada 1 | 64 | Camada 4 | 512 | 576 |
| ResNet-50 | Camada 2 | 512 | Camada 3 | 1024 | 1536 |
| iBOT | Bloco 2 | 384 | Bloco 9 | 384 | 768 |
| VMamba | Estágio 1 | 96 | Estágio 2 | 192 | 288 |

**TABELA VI** - Resultados de fusão para classificação de cor

| Backbone | E (cor) | D (textura) | Z (fusão) | Δ |
|----------|:-------:|:-----------:|:---------:|:-:|
| VGG-16 | **95,13%** | 71,80% | 86,13% | -9,00% |
| ResNet-50 | **95,87%** | 88,27% | 94,13% | -1,73% |
| iBOT | **96,93%** | 81,33% | 85,13% | -11,80% |
| VMamba | **94,80%** | 93,93% | 94,13% | -0,67% |

**TABELA VII** - Resultados de fusão para classificação de textura

| Backbone | E (cor) | D (textura) | Z (fusão) | Δ |
|----------|:-------:|:-----------:|:---------:|:-:|
| VGG-16 | 70,73% | **94,73%** | 94,73% | 0,00% |
| ResNet-50 | 90,67% | 93,27% | **93,53%** | +0,27% |
| iBOT | 78,33% | **87,80%** | 88,07% | +0,27% |
| VMamba | 88,40% | **94,33%** | 93,40% | -0,93% |

A Figura 12 visualiza os resultados de fusão, comparando a acurácia das características da camada de cor (E), camada de textura (D) e a fusão concatenada (Z) para cada tarefa e arquitetura.

*Fig. 12. Experimento 2: Fusão Early-Deep. Comparação entre características da camada de cor (E), camada de textura (D) e fusão concatenada (Z) para classificação de cor e textura em cada arquitetura.*

![Fusão Early-Deep: Comparação E vs D vs Z](exp2_fusao_comparacao.png)

A Figura 13 apresenta o impacto da fusão em relação à melhor camada individual, mostrando claramente que a fusão por concatenação raramente melhora o desempenho e frequentemente o degrada, especialmente para classificação de cor.

*Fig. 13. Impacto da Fusão: Melhoria (ou degradação) da fusão Z em relação à melhor camada individual (E ou D) para cada tarefa.*

![Impacto da Fusão](exp2_fusao_improvement.png)

Os resultados de fusão mostram que a concatenação simples geralmente não melhora o desempenho de classificação. Para classificação de cor, a fusão consistentemente degradou a acurácia em comparação com o uso da camada de cor especializada sozinha (E), com o iBOT mostrando a maior degradação (-11,80%). Para classificação de textura, os resultados foram mistos, com melhorias mínimas para ResNet-50 e iBOT (+0,27%) mas degradação para VMamba (-0,93%). Estes resultados indicam que representações especializadas de camada única são mais eficazes do que fusão ingênua de múltiplas camadas.

### G. Fusão de Características Early-Deep com LMFCN

Para avaliar se o padrão observado nas arquiteturas anteriores se mantém no framework LMFCN, também foram conduzidos experimentos de fusão concatenando características das melhores camadas para cor e textura. A Tabela VIII apresenta a configuração de fusão utilizada para cada arquitetura LMFCN.

**TABELA VIII** - Configuração de fusão para cada arquitetura LMFCN

| Arquitetura | Camadas Fusionadas | Dim z |
|-------------|:------------------:|:-----:|
| TCNN3 | Camada 1 + Camada 2 | 1920 |
| TCNN4 | Camada 2 + Camada 4 | 1920 |
| TCNN5 | Camada 2 + Camada 5 | 1920 |
| TCNN6 | Camada 6 + Camada 3 | 1920 |
| ResNet-18 (2 blocos) | Camada 1 + Camada 2 | 1920 |
| ResNet-18 (4 blocos) | Camada 3 + Camada 1 | 3200 |

**TABELA IX** - Resultados de fusão LMFCN para classificação de textura

| Arquitetura | Melhor Camada (%) | Fusão (%) | Std | Δ |
|-------------|:-----------------:|:---------:|:---:|:-:|
| TCNN3 | **58,27** | 56,07 | 0,0225 | -2,20% |
| TCNN4 | **60,40** | 58,07 | 0,0240 | -2,33% |
| TCNN5 | **62,93** | 58,47 | 0,0201 | -4,46% |
| TCNN6 | **64,34** | 62,87 | 0,0339 | -1,47% |
| ResNet-18 (2 blocos) | **91,00** | 89,40 | 0,0169 | -1,60% |
| ResNet-18 (4 blocos) | **92,47** | 91,93 | 0,0086 | -0,54% |

**TABELA X** - Resultados de fusão LMFCN para classificação de cor

| Arquitetura | Melhor Camada (%) | Fusão (%) | Std | Δ |
|-------------|:-----------------:|:---------:|:---:|:-:|
| TCNN3 | **95,94** | 95,27 | 0,0064 | -0,67% |
| TCNN4 | **94,80** | 93,20 | 0,0061 | -1,60% |
| TCNN5 | **95,40** | 95,07 | 0,0076 | -0,33% |
| TCNN6 | **94,73** | 95,53 | 0,0038 | +0,80% |
| ResNet-18 (2 blocos) | **96,13** | 96,33 | 0,0063 | +0,20% |
| ResNet-18 (4 blocos) | **95,93** | 94,80 | 0,0087 | -1,13% |

Os resultados de fusão com LMFCN confirmam o padrão observado nas outras arquiteturas. Para classificação de textura, a fusão consistentemente degradou a acurácia em comparação com a melhor camada individual, com quedas variando de -0,54% (ResNet-18 com 4 blocos) a -4,46% (TCNN5). Para classificação de cor, os resultados foram mistos: enquanto a maioria das configurações apresentou degradação, TCNN6 (+0,80%) e ResNet-18 com 2 blocos (+0,20%) mostraram pequenas melhorias. Notavelmente, o classificador SVM do LMFCN obteve os melhores resultados absolutos para cor (97,73% com TCNN3) e textura (96,87% com ResNet-18 de 2 blocos), superando a fusão por concatenação em ambos os casos.

## 6. DISCUSSÃO

Os resultados experimentais apresentados neste estudo fornecem evidências de que o padrão de sensibilidade por camada observado anteriormente em CNNs clássicas se estende a arquiteturas modernas baseadas em Vision Transformers e Modelos de Espaço de Estados. Em todos os quatro backbones avaliados, camadas iniciais consistentemente capturaram características mais adequadas para classificação de cor, enquanto camadas mais profundas provaram ser mais eficazes para reconhecimento de textura. Este padrão se manteve independentemente do paradigma arquitetural — seja convolucional (VGG-16, ResNet-50), baseado em atenção (iBOT) ou baseado em espaço de estados (VMamba).

As diferenças de desempenho entre arquiteturas oferecem insights sobre suas características representacionais. A excepcional acurácia de textura do iBOT (97,20%) sugere que o objetivo de modelagem de imagem mascarada auto-supervisionada encoraja o aprendizado de ricas representações estruturais em camadas mais profundas. A dimensionalidade de características relativamente uniforme entre os blocos iBOT (384 dimensões) contrasta com as dimensões crescentes em CNNs e VMamba, no entanto o iBOT alcançou discriminação de textura superior. Isso indica que a qualidade das representações aprendidas, ao invés de sua dimensionalidade, é o fator determinante para classificação de textura.

O experimento de fusão early-deep revelou que a concatenação simples não melhora o desempenho de classificação. Para classificação de cor, a fusão consistentemente degradou a acurácia comparada ao uso da camada de cor especializada sozinha, com o iBOT mostrando a maior degradação (-11,80%). Para classificação de textura, os resultados foram marginais na melhor das hipóteses (+0,27% para ResNet-50 e iBOT). Estes resultados indicam que representações especializadas de camada única são mais eficazes do que fusão ingênua de múltiplas camadas, e mecanismos de fusão mais sofisticados (como abordagens baseadas em atenção ou ponderação aprendida) podem ser necessários para se beneficiar de características de múltiplas camadas.

As implicações práticas dessas descobertas são significativas para sistemas de recuperação de imagens baseados em conteúdo. Ao invés de usar características da camada final de uma rede pré-treinada — como é prática comum — nossos resultados sugerem que a extração de características deve ser específica por camada com base no atributo alvo. Para aplicações que requerem recuperação baseada em cor, características de camadas iniciais devem ser preferidas, enquanto recuperação baseada em textura se beneficia de representações de camadas mais profundas. Além disso, a escolha da arquitetura backbone deve considerar o atributo primário de interesse: CNNs clássicas oferecem um bom equilíbrio para tarefas de cor, enquanto o iBOT fornece vantagens substanciais para aplicações focadas em textura.

## 7. CONCLUSÕES

Este estudo apresenta uma avaliação abrangente da extração de características de camadas ocultas em múltiplas arquiteturas de deep learning para classificação de cor e textura em imagens de moda. Avaliamos CNNs clássicas (VGG-16 e ResNet-50), arquiteturas modernas (iBOT e VMamba) e o framework LMFCN.

Os experimentos confirmam que o padrão de sensibilidade por camada observado em estudos anteriores se estende aos Vision Transformers e Modelos de Espaço de Estados modernos: camadas iniciais são mais eficazes para classificação de cor, enquanto camadas mais profundas se destacam no reconhecimento de textura. Entre as arquiteturas avaliadas, o iBOT alcançou a maior acurácia de textura (97,20% no Bloco 9), enquanto ResNet-50 e iBOT empataram na melhor acurácia de cor (95,87%).

Adicionalmente, a fusão simples por concatenação de características das melhores camadas de cor e textura não melhora o desempenho em relação às representações especializadas de camada única, sugerindo que mecanismos de fusão mais sofisticados são necessários.

Estas descobertas têm implicações práticas para sistemas de recuperação de imagens baseados em conteúdo, sugerindo que a extração de características deve ser específica por atributo, ao invés de depender de representações da camada final. Trabalhos futuros investigarão a combinação de características de camadas ótimas entre diferentes arquiteturas e mecanismos de fusão baseados em atenção para potencialmente melhorar o desempenho de classificação.

## AGRADECIMENTOS

Este trabalho é apoiado pelo Programa de Pós-Graduação em Informática da Pontifícia Universidade Católica do Paraná e parcialmente apoiado pelo Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) vinculado ao Ministério da Ciência, Tecnologia e Inovação, para aprimorar a pesquisa no Brasil.

## REFERÊNCIAS

1. Dubey, S. R. (2021). A decade survey of content based image retrieval using deep learning. IEEE Transactions on Circuits and Systems for Video Technology, 32(5), 2687-2704.

2. Baloian, A., Murrugarra-Llerena, N., Saavedra, J. M. (2021). Scalable visual attribute extraction through hidden layers of a residual convnet. arXiv preprint arXiv:2104.00161.

3. Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition (pp. 248-255).

4. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

5. Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., Shah, R. (1993). Signature verification using a siamese time delay neural network. Advances in neural information processing systems, 6.

6. Chen, X., He, K. (2021). Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 15750-15758).

7. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

8. Moresco, M., Britto, A. D. S., Costa, Y. M., Senger, L. J., Hochuli, A. G. (2022). Combining multi-layer features for plant species classification in a Siamese network. In 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC) (pp. 2446-2451). IEEE.

9. Araújo, V. M., Britto Jr, A. S., Oliveira, L. S., Koerich, A. L. (2022). Two-view fine-grained classification of plant species. Neurocomputing, 467, 427-441.

10. Liang, K., Chang, H., Shan, S., Chen, X. (2015). A unified multiplicative framework for attribute learning. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2506-2514).

11. Gan, C., Yang, T., Gong, B. (2016). Learning attributes equals multi-source domain generalization. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 87-97).

12. Youssef, S. M. (2012). ICTEDCT-CBIR: Integrating curvelet transform with enhanced dominant colors extraction and texture analysis for efficient content-based image retrieval. Computers & Electrical Engineering, 38(5), 1358-1376.

13. Irtaza, A., Jaffar, M. A., Aleisa, E., Choi, T. S. (2014). Embedding neural networks for semantic association in content based image retrieval. Multimedia tools and applications, 72, 1911-1931.

14. Atlam, H. F., Attiya, G., El-Fishawy, N. (2017). Integration of color and texture features in CBIR system. Int. J. Comput. Appl, 164(3), 23-29.

15. Lin, C. H., Chen, R. T., Chan, Y. K. (2009). A smart content-based image retrieval system based on color and texture feature. Image and vision Computing, 27(6), 658-665.

16. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

17. Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., & Kong, T. (2021). iBOT: Image BERT pre-training with online tokenizer. arXiv preprint arXiv:2111.07832.

23. Mukherjee, D., Mondal, R., Singh, P. K., Sarkar, R., & Bhattacharjee, D. (2020). EnsembleNet: A hybrid approach for vehicle detection. Neural Computing and Applications, 32(15), 14207-14228.

24. Yue, J., Li, Z., Liu, L., & Fu, Z. (2011). Content-based image retrieval using color and texture. In 2011 Sixth International Conference on Image and Graphics (pp. 833-837). IEEE.

25. Prakasa, E. (2016). Texture Feature Extraction by Using Local Binary Pattern. INKOM Journal, 1(1), 1-6.

26. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE CVPR (pp. 3431-3440).

27. Guo, Y., Liu, Y., Georgiou, T., & Lew, M. S. (2018). A review of semantic segmentation using deep neural networks. International Journal of Multimedia Information Retrieval, 7(2), 87-93.

28. de Matos, J., de Oliveira, L. E. S., Junior, A. D. S. B., & Koerich, A. L. (2023). Large-margin representation learning for texture classification. Pattern Recognition Letters, 170, 39-47.
