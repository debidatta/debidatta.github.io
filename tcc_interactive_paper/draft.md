## Abstract

We introduce a self-supervised representation learning method based on the task of
    temporal alignment between videos. The method trains a network using temporal cycle-consistency (TCC), a differentiable cycle-consistency loss that can be used to find correspondences across time in multiple videos. The resulting per-frame embeddings can be used to align videos by simply matching frames using nearest-neighbors in the learned embedding space.

To evaluate the power of the embeddings, we densely label the <i>Pouring</i> and <i>Penn Action</i> video datasets for action phases. We show that (i) the learned embeddings enable few-shot classification of these action phases, significantly reducing the supervised training requirements; and (ii) TCC is complementary to other methods of self-supervised learning in videos, such as Shuffle and Learn and Time-Contrastive Networks. The embeddings are also used for a number of applications based on alignment (dense temporal correspondence) between video pairs, including transfer of metadata of synchronized modalities between videos (sounds, temporal semantic labels), synchronized playback of multiple videos, and anomaly detection.

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/top.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: TCC embeddings are useful  for temporally fine-grained tasks. In the above video, we retrieve nearest neighbors in the embedding space to frames in the reference video. In spite of many variations, TCC maps semantically similar frames to nearby points in the embedding space.
</figcaption>
</div>

______

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/teaser.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 125%;"></video>
<figcaption>
Figure 2. We present a self-supervised representation learning technique called temporal cycle consistency (TCC) learning. It is inspired by the temporal video alignment problem, which refers to the task of finding correspondences across multiple videos despite many factors of variation. The learned representations are useful for fine-grained temporal understanding in videos. Additionally, we can now align multiple videos by simply finding nearest-neighbor frames in the embedding space.
</figcaption>
</div>

## Introduction

The world presents us with abundant examples of sequential processes. A plant growing from a seedling to a tree, the daily routine of getting up, going to work and coming back home, or a person pouring themselves a glass of water -- are all examples of events that happen in a particular order. Videos capturing such processes not only contain information about the causal nature of these events, but also provide us with a valuable signal -- the possibility of temporal <i>correspondences</i> lurking across multiple instances of the same process. 
For example, during pouring, one could be reaching for a teapot, a bottle of wine, or a glass of water to pour from. Key moments such as the first touch to the container or the container being lifted from the ground are common to all pouring sequences. These correspondences, which exist in spite of many varying factors like visual changes in viewpoint, scale, container style, the speed of the event, etc., could serve as the link between raw video sequences and high-level temporal abstractions (e.g. phases of actions). In this work we present evidence that suggests the very act of <i>looking for correspondences</i> in sequential data enables the learning of rich and useful representations, particularly suited for fine-grained temporal understanding of videos. 


Temporal reasoning in videos, understanding multiple stages of a process and causal relations between them, is a relatively less studied problem compared to recognizing action categories <dt-cite key="carreira2017quo,soomro2012ucf101"></dt-cite>. Learning representations that can differentiate between states of objects as an action proceeds is critical for perceiving and acting in the world. It would be desirable for a robot tasked with learning to pour drinks to understand each intermediate state of the world as it proceeds with performing the task. Although videos are a rich source of sequential data essential to understanding such state changes, their true potential remains largely untapped. One hindrance in the fine-grained temporal understanding of videos can be an excessive dependence on pure supervised learning methods that require per-frame annotations. It is not only difficult to get every frame labeled in a video because of the manual effort involved, but also it is not entirely clear what are the exhaustive set of labels that need to be collected for fine-grained understanding of videos. Alternatively, we explore self-supervised learning of correspondences between videos across time. We show that the emerging features have strong temporal reasoning capacity, which is demonstrated through tasks such as action phase classification and tracking the progress of an action.   

When frame-by-frame alignment (i.e. supervision) is available, learning correspondences reduces to learning a common embedding space from pairs of aligned frames (e.g. CCA<dt-cite key="anderson1958introduction,andrew2013deep"></dt-cite> and ranking loss<dt-cite key="Sermanet2017TCN"></dt-cite>). However, for most of the real world sequences such frame-by-frame alignment does not exist naturally. One option would be to artificially obtain aligned sequences by recording the same event through multiple cameras<dt-cite key="Sermanet2017TCN,sigurdsson2018actor,revaud2013event"></dt-cite>. Such data collection methods might find it difficult to capture all the variations present naturally in videos in the wild. On the other hand, our self-supervised objective does not need explicit correspondences to align different sequences. It can align significant variations within an action category (e.g. pouring liquids, or baseball pitch). Interestingly, the embeddings that emerge from learning the alignment prove to be useful for fine-grained temporal understanding of videos. More specifically, we learn an embedding space that maximizes one-to-one mappings (i.e. cycle-consistent points) across pairs of video sequences within an action category. In order to do that, we introduce two differentiable versions of cycle consistency computation which can be optimized by conventional gradient-based optimization methods. Further details of the method will be explained in section <a href="#cycle_consistent_representation_learning">Cycle Consistent Representation Learning</a>.

The main contribution of this paper is a new self-supervised training method, referred to as temporal cycle consistency (TCC) learning, that learns representations by aligning video sequences of the same action. We compare TCC representations against features from existing self-supervised video representation methods <dt-cite key="Sermanet2017TCN,misra2016shuffle"></dt-cite> and supervised learning, for the tasks of action phase classification and continuous progress tracking of an action. Our approach provides significant performance boosts when there is a lack of labeled data. We also collect per-frame annotations of Penn Action<dt-cite key="zhang2013actemes"></dt-cite> and Pouring<dt-cite key="Sermanet2017TCN"></dt-cite> datasets that we will release publicly to facilitate evaluation of fine-grained video understanding tasks.

## Related Work

**Cycle consistency**. Validating good matches by cycling between two or more samples is a commonly used technique in computer vision. It has been applied successfully for tasks like co-segmentation<dt-cite key="wang2014unsupervised,wang2013image"></dt-cite>, structure from motion <dt-cite key="zach2010disambiguating,wilson2013network"></dt-cite>, and image matching<dt-cite key="zhou2015multi,zhou2016learning,zhou2015flowweb"></dt-cite>.  
For instance, FlowWeb<dt-cite key="zhou2015flowweb"></dt-cite> optimizes globally-consistent
dense correspondences using the cycle consistent flow fields between all pairs of images in a collection, whereas Zhou et al.<dt-cite key="zhou2015multi"></dt-cite> approaches a similar task by formulating it as a low-rank matrix recovery problem and solves it through fast alternating minimization. These methods learn robust dense correspondences on top of fixed feature representations (e.g. SIFT, deep features, etc.) by enforcing cycle consistency and/or spatial constraints between the images. Our method differs from these approaches in that TCC is a self-supervised representation learning method which learns embedding spaces that are optimized to give good correspondences. Furthermore we address a temporal correspondence problem rather than a spatial one.
Zhou et al.<dt-cite key="zhou2016learning"></dt-cite> learn to align multiple images using the supervision from 3D guided cycle-consistency by leveraging the initial correspondences that are available between multiple renderings of a 3D model, whereas we don't assume any given correspondences. Another way of using cyclic relations is to directly learn bi-directional transformation functions between multiple spaces such as CycleGANs<dt-cite key="zhu2017unpaired"></dt-cite> for learning image transformations, and CyCADA<dt-cite key="hoffman2017cycada"></dt-cite> for domain adaptation. Unlike these approaches we don't have multiple domains, and we can't learn transformation functions between all pairs of sequences. Instead we learn a joint embedding space in which the Euclidean distance defines the mapping across the frames of multiple sequences.  Similar to us, Aytar et al.<dt-cite key="aytar2018playing"></dt-cite> applies cycle-consistency between temporal sequences, however they use it as a validation tool for hyper-parameter optimization of learned representations for the end goal of imitation learning. Unlike our approach, their cycle-consistency measure is non-differentiable and hence can't be directly used for representation learning. 

**Video alignment**. When we have synchronization information (e.g. multiple cameras recording the same event) then learning a mapping between multiple video sequences can be accomplished by using existing methods such as Canonical Correlation Analysis (CCA)<dt-cite key="anderson1958introduction,andrew2013deep"></dt-cite>, ranking<dt-cite key="Sermanet2017TCN"></dt-cite> or match-classification<dt-cite key="arandjelovic2017look"></dt-cite> objectives. For instance TCN<dt-cite key="Sermanet2017TCN"></dt-cite> and circulant temporal encoding<dt-cite key="revaud2013event"></dt-cite> align multiple views of the same event, whereas Sigurdsson et al.<dt-cite key="sigurdsson2018actor"></dt-cite> learns to align first and third person videos. Although we have a similar objective, these methods are not suitable for our task as we cannot assume any given correspondences between different videos.

**Action localization and parsing**. As action recognition is quite popular in the computer vision community, many studies
<dt-cite key="wang2016temporal,sigurdsson2017asynchronous,zhao2017temporal,girdhar2017actionvlad,yeung2018every"></dt-cite> explore efficient deep architectures for action recognition and localization in videos.  Past work has also explored parsing of fine-grained actions in videos <dt-cite key="pirsiavash2014parsing,lan2015action,lea2016segmental"></dt-cite> while some others 
<dt-cite key="shechtman2007matching,del2015articulated,sener2015unsupervised,sener2018unsupervised"></dt-cite> discover sub-activities without explicit supervision of temporal boundaries. <dt-cite key="heidarivincheh2018action"></dt-cite> learns a supervised regression model with voting to predict the completion of an action, and <dt-cite key="Alayrac16unsupervised"></dt-cite> discovers key events in an unsuperivsed manner using a weak association between videos and text instructions. 
However all these methods heavily rely on existing deep image <dt-cite key="he2016deep,simonyan2014very"></dt-cite> or spatio-temporal<dt-cite key="wang2013action"></dt-cite> features, whereas we learn our representation from scratch using raw video sequences.    

**Soft nearest neighbours**. The differentiable or soft formulation for nearest-neighbors is a commonly known method <dt-cite key="goldberger2005neighbourhood"></dt-cite>. This formulation has recently found application in metric learning for few-shot learning<dt-cite key="snell2017prototypical,movshovitz2017no,rocco2018neighbourhood"></dt-cite>. We also make use of soft nearest neighbor formulation as a component in our differentiable cycle-consistency computation.

**Self-supervised representations.** There has been significant progress in learning from images and videos without requiring class or temporal segmentation labels. Instead of labels, self-supervised learning methods use signals such as temporal order<dt-cite key="misra2016shuffle,fernando2017self"></dt-cite>, consistency across viewpoints and/or temporal neighbors<dt-cite key="Sermanet2017TCN"></dt-cite>, classifying arbitrary temporal segments<dt-cite key="hyvarinen2016unsupervised"></dt-cite>, temporal distance classification within or across modalities<dt-cite key="aytar2018playing"></dt-cite>, spatial permutation of patches<dt-cite key="doersch2015unsupervised,anoop33deeppermnet"></dt-cite>, visual similarity<dt-cite key="sanakoyeu2018deep"></dt-cite> or a combination of such signals<dt-cite key="doersch2017multi"></dt-cite>.
While most of these approaches optimize each sample independently, TCC jointly optimizes over two sequences at a time, potentially capturing more variations in the embedding space. Additionally, we show that TCC yields best results when combined with some of the unsupervised losses above.

## Cycle Consistent Representation Learning
<div id="cycle_consistent_representation_learning"></div>

<div class="figure" id="cycle">
<video class="b-lazy" data-src="assets/mp4/teaser_small.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 80%;"></video>
<figcaption>
Figure 3: <strong>Cycle-consistent representation learning.</strong> We show two example video sequences encoded in an example embedding space. If we use nearest neighbors for matching, one point (shown in black) is <i>cycling back to itself</i> while another one (shown in red) is not. Our target is to learn an embedding space where maximum number of points can cycle back to themselves. We achieve it by minimizing the cycle consistency error (shown in red dotted line) for each point in every pair of sequences.
</figcaption>
</div>

<div class="figure" id="soft_cycle_consistency">
<img src="assets/fig/method.png" style="margin: 0; width: 125%;"/>
</div>
<div class="figure">
<video class="b-lazy" data-src="assets/mp4/method.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 125%;"></video>
</figcaption>
</div>

Figure 4: <strong>Temporal cycle consistency</strong>. The embedding sequences $U$ and $V$ are obtained by encoding video sequences $S$ and $T$ with the encoder network $\phi$, respectively. For the selected point $u_i$ in $U$, soft nearest neighbor computation and cycling back to $U$ again is demonstrated visually. Finally the normalized distance between the index $i$ and cycling back distribution $N(\mu,\sigma^2)$ (which is fitted to $\beta$) is minimized.

The core contribution of this work is a self-supervised approach to learn an embedding space where two similar video sequences can be aligned temporally. More specifically, we intend to maximize the number of points that can be mapped one-to-one between two sequences by using the minimum distance in the learned embedding space. We can achieve such an objective by maximizing the number of cycle-consistent frames between two sequences (see <a href="#cycle">Figure 3</a>). However, cycle-consistency computation is typically not a differentiable procedure. In order to facilitate learning such an embedding space using back-propagation, we introduce two differentiable versions of the <i>cycle-consistency loss</i>, which we describe in detail below.

Given any frame $s_i$ in a sequence $S=\{s_1,s_2,...,s_N\}$, the embedding is computed as $u_i = \phi(s_i;\theta)$, where $\phi$ is the neural network encoder parameterized by $\theta$. For the following sections, assume we are given two video sequences $S$ and $T$, with lengths $N$ and $M$, respectively. Their embeddings are computed as $U=\{u_1,u_2,...,u_N\}$ and $V=\{v_1,v_2,...,v_M\}$ such that $u_i = \phi(s_i;\theta)$ and $v_i = \phi(t_i;\theta)$. 


### Cycle-consistency
<div id="cycle_consistency"></div>

In order to check if a point $u_i \in U$ is cycle consistent, we first determine its nearest neighbor, $v_j = \arg \min_{v \in V} ||u_i-v||$. We then repeat the process to find the nearest neighbor of $v_j$ in $U$, i.e. $u_k = \arg \min_{u \in U} ||v_j-u||$. The point $u_i$ is <i>cycle-consistent</i> if and only if $i=k$, in other words if the point $u_i$ cycles back to itself.
<a href="#cycle">Figure 3</a> provides positive and negative examples of cycle consistent points in an embedding space.
We can learn a good embedding space by maximizing the number of cycle-consistent points for any pair of sequences. However that would require a differentiable version of cycle-consistency measure, two of which we introduce below.

### Cycle-back Classification

We first compute the soft nearest neighbor $\tilde{v}$ of $u_i$ in $V$, then figure out the nearest neighbor of $\tilde{v}$ back in $U$. We consider each frame in the first sequence $U$ to be a separate class and our task of checking for cycle-consistency reduces to classification of the nearest neighbor correctly. The logits are calculated using the distances between $\tilde{v}$ and any $u_k \in U$, and the ground truth label $y$ are all zeros except for the $i^{th}$ index which is set to 1.  

For the selected point $u_i$, we use the softmax function to define its soft nearest neighbor $\tilde{v}$ as:

$\tilde{v} = \sum_j^M \alpha_j v_j, \quad where \quad \alpha_j = \frac{e^{-||u_i-v_j||^2}}{\sum_k^M e^{-||u_i-v_k||^2}}$ &nbsp; &nbsp; (1)

and $\alpha$ is the the similarity distribution which signifies the proximity between $u_i$ and each $v_j \in V$. And then we solve the $N$ class (i.e.\ number of frames in $U$) classification problem where the logits are $x_k = -||\tilde{v} - u_k||^2$ and the predicted labels are $\hat{y} = softmax(x)$. Finally we optimize the cross-entropy loss as follows:

$L_{cbc} = -\sum_j^N y_j \log(\hat{y}_j)$ &nbsp; &nbsp; (2)

### Cycle-back Regression

Although cycle-back classification defines a differentiable cycle-consistency loss function, it has no notion of how close or far in time the point to which we cycled back is. We want to penalize the model less if we are able to cycle back to closer neighbors as opposed to the other frames that are farther away in time. In order to incorporate temporal proximity in our loss, we introduce cycle-back regression. A visual description of the entire process is shown in <a href="#soft_cycle_consistency">Figure 4</a>. Similar to the previous method first we compute the soft nearest neighbor $\tilde{v}$ of $u_i$ in $V$. Then we compute the similarity vector $\beta$ that defines the proximity between $\tilde{v}$ and each $u_k \in U$ as:

$\beta_k = \frac{e^{-||\tilde{v}-u_k||^2}}{\sum_j^N e^{-||\tilde{v} - u_j||^2}}$ &nbsp; &nbsp; (3)

Note that $\beta$ is a discrete distribution of similarities over time and we expect it to show a peaky behavior around the $i^{th}$ index in time. Therefore, we impose a Gaussian prior on $\beta$ by minimizing the normalized squared distance $\frac{|i-\mu|^2}{\sigma^2}$ as our objective. We enforce $\beta$ to be more peaky around $i$ by applying additional variance regularization. We define our final objective as:

<a name="eq:4"></a>
$L_{cbr} = \frac{|i-\mu|^2}{\sigma^2} + \lambda \log(\sigma)$ &nbsp; &nbsp; (4)

where $\mu = \sum_{k}^N \beta_k * k$ and $\sigma^2 = \sum_{k}^N \beta_k * (k-\mu)^2$, and $\lambda$ is the regularization weight. Note that we minimize the log of variance as using just the variance is more prone to numerical instabilities. All these formulations are differentiable and can conveniently be optimized with conventional back-propagation. 

### Implementation details
<div id="implementation_details"></div>

**Training Procedure**. Our self-supervised representation is learned by minimizing the cycle-consistency loss for all the pair of sequences in the training set. Given a sequence pair, their frames are embedded using the encoder network and we optimize cycle consistency losses for randomly selected frames within each sequence until convergence. We used Tensorflow<dt-cite key="abadi2016tensorflow"></dt-cite> for all our experiments.

**Encoding Network**.  All the frames in a given video sequence are resized to $224 \times 224$. When using ImageNet pretrained features, we use ResNet-50<dt-cite key="he2016deep"></dt-cite> architecture to extract features from the output of <i>Conv4c</i> layer. The size of the extracted convolutional features are $14 \times 14 \times 1024$. Because of the size of the datasets, when training from scratch we use a smaller model along the lines of VGG-M<dt-cite key="chatfield2014return"></dt-cite>. This network takes input at the same resolution as ResNet-50 but is only 7 layers deep. The convolutional features produced by this base network are of the size $14 \times 14 \times 512$. These features are provided as input to our embedder network (presented in <a href="#tab:archiecture">Table 1</a>). We stack the features of any given frame and its $k$ context frames along the dimension of time. This is followed by 3D convolutions for aggregating temporal information. We reduce the dimensionality by using 3D max-pooling followed by two fully connected layers. Finally, we use a linear projection to get a 128-dimensional embedding for each frame. More details of the architecture are presented in the supplementary material.

<div class="figure" id="tab:archiecture">
<img src="assets/fig/table1.png" style="margin: 0; width: 80%;"/>
<figcaption>
Table 1: Architecture of the embedding network.
</figcaption>
</div>

## Datasets and Evaluation

<div class="figure" id="annotation">
<img src="assets/fig/annotation.png" style="margin: 0; width: 120%;"/>
<figcaption>
Figure 5: <strong>Example labels</strong> for the actions `Baseball Pitch' (top row) and `Pouring' (bottom row).
The key events are shown in boxes below the frame (e.g. `Hand touches bottle'),
and each frame in between two key events has a phase label (e.g. `Lifting bottle').
</figcaption>
</div>

We validate the usefulness of our representation learning technique on
two datasets: (i) <i>Pouring</i><dt-cite key="Sermanet2017TCN"></dt-cite>; and (ii)
<i>Penn Action</i><dt-cite key="zhang2013actemes"></dt-cite>. These datasets both contain
videos of humans performing actions, 
and provide us with collections of videos where dense alignment can be
performed. 
While <i>Pouring</i> focuses
more on the objects being interacted with, <i>Penn Action</i>
focuses on humans doing sports or exercise. 

**Annotations.** For evaluation purposes, we add two types of labels to the video frames of 
these datasets: key events and phases. 
Densely labeling each frame in a video is a difficult and
time-consuming task. Labelling only <i>key events</i> both reduces the number of frames
that need to be annotated, and also reduces 
the ambiguity of the task (and thus the
disagreement between annotators). For example, annotators agree more
about the frame when the golf club hits the ball (a key event) than when  the
golf club is at a certain angle. The <i>phase</i> is the period between two key events, and all frames in the
period have the same phase label. It is similar to tasks proposed in<dt-cite key="kuehne2014language,bojanowski2014weakly,damen2018scaling"></dt-cite>. Examples of key events and phases are shown in 
<a href="#annotation">Figure 5</a>, 
and <a href="#tab:dataset">Table 2</a> gives the complete list for all the actions we consider.

We use all the real videos from the <i>Pouring</i> dataset, and all but two action categories
in <i>Penn Action</i>. We do not use
<i>Strumming guitar</i> and <i>Jumping rope</i> because
it is difficult to define unambiguous key events for these. We 
use the train/val splits of the original
datasets<dt-cite key="Sermanet2017TCN,zhang2013actemes"></dt-cite>. 
We will publicly release these new annotations.

<div class="figure" id="tab:dataset">
<img src="assets/fig/table2.png" style="margin: 0; width: 125%;"/>
<figcaption>
Table 2: List of all key events in each dataset. Note that each action has a <i>Start</i> event and <i>End</i> event in addition to the key events above.
</figcaption>
</div>

### Evaluation
<div id="metrics"></div>

We use three evaluation measures computed on the validation set. These metrics evaluate the model on fine-grained temporal understanding of a given action.
Note, the networks are first trained on the training set and then frozen. SVM classifiers and linear regressors are trained on the features from the networks, with no additional fine-tuning of the networks. 
For all measures a higher score implies a better model. 

<strong>1. Phase classification accuracy:</strong> is the per frame phase classification accuracy.
This is implemented by training a SVM classifier on  the phase labels for each frame of the
training data.

<strong>2. Phase progression:</strong>
 measures how well the <i>progress</i> of a process or action is captured by the embeddings. We first define
an approximate measure of progress through a phase
as the difference in time-stamps between any
given frame and each key event. This is normalized by the number of
frames present in that video. Similar definitions can be found in recent literature <dt-cite key="ma2016learning,becattini2017done,heidarivincheh2018action"></dt-cite>.
We use a linear regressor on the features to predict the phase progression values. It is computed as the  the average $R$-squared measure (coefficient of
determination)<dt-cite key="wiki:rsquared"></dt-cite>, given by:

$R^2 = 1- \frac{\sum_{i=1}^n (y_i - \hat{y_i})^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$

where $y_i$ is the ground truth event progress value, $\bar{y}$ is the
mean of all $y_i$ and $\hat{y_i}$ is the prediction made by the linear
regression model. The maximum value of this measure is $1$.

<strong>3. Kendall's Tau <dt-cite key="wiki:kendallstau"></dt-cite>:</strong>
is a  statistical measure that can determine how 
well-aligned two sequences are in time. Unlike the above two 
measures it does not require  additional labels for evaluation. 
Kendall's Tau is calculated over every pair of frames in a pair of videos by 
sampling  a pair of frames ($u_i, u_j$) in the first video (which has $n$
frames) and retrieving the corresponding nearest frames in the second
video, ($v_p$, $v_q$). This quadruplet of frame indices $(i,j,p,q)$ is
said to be <i>concordant</i> if $i < j$ and $p < q$ or $i > j$ and
$p > q$. Otherwise it is said to be <i>discordant</i>. Kendall's Tau
is defined over all pairs of frames in the first video as:

$\tau = \frac{(\text{no. of concordant pairs} - \text{no. of discordant
pairs})}{\frac{n(n-1)}{2}}$

We refer the reader to <dt-cite key="wiki:kendallstau"></dt-cite> to check out the
complete definition. The reported metric is the average Kendall's Tau
over all pairs of videos in the validation set. It is a measure of how
well the learned representations generalize to aligning unseen
sequences if we used nearest neighbour matching for aligning a pair of
videos. A value of 1 implies the videos are perfectly aligned while a
value of -1 implies the videos are aligned in the reverse order. One
drawback of Kendall's tau is that it assumes there are no repetitive frames in a video. This might not be the case if an action is
being done slowly or if there is periodic motion. For the datasets we
consider, this drawback is not a problem.

## Experiments

### Baselines
<div id="methods"></div>

We compare our representations with existing self-supervised video representation learning methods. For completeness, we briefly describe the baselines below but recommend referring to the original papers for more details. 

**Shuffle and Learn (SaL)<dt-cite key="misra2016shuffle"></dt-cite>.** We randomly sample triplets of frames in the manner suggested by <dt-cite key="misra2016shuffle"></dt-cite>. We train a small classifier to predict if the frames are in order or shuffled. The labels for training this classifier are derived from the indices of the triplet we sampled. This loss encourages the representations to encode information about the order in which an action should be performed.

**Time-Constrastive Networks (TCN)<dt-cite key="Sermanet2017TCN"></dt-cite>.** We sample $n$ frames from the sequence and use these as anchors (as defined in the metric learning literature). For each anchor, we sample positives within a fixed time window. This gives us n-pairs of anchors and positives. We use the n-pairs loss<dt-cite key="sohn2016improved"></dt-cite> to learn our embedding space. For any particular pair, the n-pairs loss considers all the other pairs as negatives. This loss encourages representations to be disentangled in time while still adhering to metric constraints.


**Combined Losses.** In addition to these baselines, we can combine our cycle consistency loss with both SaL and TCN to get two more training methods: TCC+SaL and TCC+TCN. We learn the embedding by computing both losses and adding them in a weighted manner to get the total loss, based on which the gradients are calculated. The weights are selected by performing a search over 3 values $0.25, 0.5, 0.75$. All baselines share the same video encoder architecture, as described in section <a href="#implementation_details">Implementation Details</a>.

### Ablation of Different Cycle Consistency Losses
We ran an experiment on the Pouring dataset to see how the different losses compare against each other. We also report metrics on the Mean Squared Error (MSE) version of the cycle-back regression loss (<a href="#eq:4">Equation 4</a>) which is formulated by only minimizing $|i-\mu|^2$, ignoring the variance of predictions altogether. We present the results in <a href="#tab:pouring_loss_ablation">Table 3</a> and observe that the variance aware cycle-back regression loss outperforms both of the other losses in all metrics. We name this version of cycle-consistency as the final temporal cycle consistency (TCC) method, and use this version for the rest of the experiments.

<div class="figure" id="tab:pouring_loss_ablation">
<img src="assets/fig/table3.png" style="margin: 0; width: 70%;"/>
<figcaption>
Table 3: Ablation of different cycle consistency losses.
</figcaption>
</div>

### Action Phase Classification



**Self-supervised Learning from Scratch.** We perform experiments to compare different self-supervised methods for learning visual representations from scratch. This is a challenging setting as we learn the entire encoder from scratch without labels.
We use a smaller encoder model (i.e.\ VGG-M<dt-cite key="chatfield2014return"></dt-cite>) as the training samples are limited.
We report the results on the <i>Pouring</i> and <i>Penn Action</i> datasets in <a href="#tab:scratch_results">Table 4</a>. On both datasets, TCC features outperform the features learned by SaL and TCN. This might be attributed to the fact that TCC learns features across multiple videos during training itself. SaL and TCN losses operate on frames from a single video only but TCC considers frames from multiple videos while calculating the cycle-consistency loss. We can also compare these results with the supervised learning setting (first row in each section), in which we train the encoder using the labels of the phase classification task. For both datasets, TCC can be used for learning features from scratch and brings about significant performance boosts over plain supervised learning when there is limited labeled data.

<div class="figure" id="tab:scratch_results">
<img src="assets/fig/table4.png" style="margin: 0; width: 70%;"/>
<figcaption>
Table 4: Phase classification results when training VGG-M from scratch.
</figcaption>
</div>

<br>

<div class="figure" id="tab:finetuning_results">
<img src="assets/fig/table5.png" style="margin: 0; width: 70%;"/>
<figcaption>
Table 5: Phase classification results when fine-tuning ImageNet pre-trained ResNet-50.
</figcaption>
</div>

**Self-supervised Fine-tuning.** Features from networks trained for the task of image classification on the ImageNet dataset have been used for many other vision tasks. They are also useful because initializing from weights of pre-trained networks leads to faster convergence. We train all the representation learning methods mentioned in Section <a href="#metrics">Evaluation</a> and report the results on the <i>Pouring</i> and <i>Penn Action</i> datasets in <a href="#tab:finetuning_results">Table 5</a>. Here the encoder model is a ResNet-50<dt-cite key="he2016deep"></dt-cite> pre-trained on ImageNet dataset. We observe that existing self-supervised approaches like SaL and TCN learn features useful for fine-grained video tasks. TCC features achieve competitive performance with the other methods on the <i>Penn Action</i> dataset while outperforming them on the <i>Pouring</i> dataset. Interestingly, the best performance is achieved by combining the cycle-consistency loss with TCN (row 8 in each section). The boost in performance when combining losses might be because training with multiples losses reduces over-fitting to cues using which the model can  minimize <i>a</i> particular loss. We can also look at the first row of their respective sections to compare with supervised learning features obtained by training on the downstream task itself. We observe that the self-supervised fine-tuning gives significant performance boosts in the low-labeled data regime (columns 1 and 2).


**Self-supervised Few Shot Learning.** We also test the usefulness of our learned representations in the few-shot scenario: we have many training videos but <i>per-frame labels</i> are only available for a few of them. In this experiment, we use the same set-up as the fine-tuning experiment described above. The embeddings are learned using either a self-supervised loss or vanilla supervised learning. To learn the self-supervised features, we use the entire training set of videos. We compare these features against the supervised learning baseline where we train the model on the videos for which labels are available. Note that one labeled video means hundreds of labeled frames. In particular, we want to see how the performance on the phase classification task is affected by increasing the number of labeled videos. We present the results in <a href="#fig:fewshot">Figure 6</a>. 
We observe significant performance boost using self-supervised methods as opposed to just using supervised learning on the labeled videos. We present results from <i>Golf Swing</i> and <i>Tennis Serve</i> classes above. With only one labeled video, TCC and TCC+TCN achieve the performance that supervised learning achieves with about 50 densely labeled videos. This suggests that there is a lot of untapped signal present in the raw videos which can be harvested using self-supervision.

<div class="figure" id="fig:fewshot">
<img src="assets/fig/fewshot.png" style="margin: 0; width: 100%;"/>
<figcaption>
Figure 6: <strong>Few shot action phase classification.</strong> TCC features provide significant performance boosts when there is a dearth of labeled videos.
</figcaption>
</div>

<br>

<div class="figure" id="tab:all_regression_results">
<img src="assets/fig/table6.png" style="margin: 0; width: 80%;"/>
<figcaption>
Table 6: Phase Progression and Kendall's Tau results. SL: Supervised Learning.
</figcaption>
</div>

### Phase Progression and Kendall's Tau
We evaluate the encodings for the remaining tasks described in Section <a href="#metrics">Evaluation</a>. These tasks measure the effectiveness of representations at a more fine-grained level than phase classification. We report the results of these experiments in <a href="#tab:all_regression_results">Table 6</a>. We observe that when training from scratch TCC features perform better on both phase progression and Kendall's Tau for both the datasets. Additionally, we note that Kendall's Tau (which measures alignment between sequences using nearest neighbors matching) is significantly higher when we learn features using the combined losses. TCC + TCN outperforms both supervised learning and self-supervised learning methods significantly for both the datasets for fine-grained tasks.

## Applications

<div class="figure" id="fig:retrieval">
<img src="assets/fig/retrieval.png" style="margin: 0; width: 100%;"/>
<figcaption>
Figure 7: Nearest neighbors in the embedding space can be used for fine-grained retrieval.
</figcaption>
</div>

<br>

<div class="figure" id="fig:anomaly">
<img src="assets/fig/anomaly.png" style="margin: 0; width: 100%;"/>
<figcaption>
Figure 8: <strong>Example of anomaly detection in a video</strong>. Distance from typical action trajectories spikes up during anomalous activity.
</figcaption>
</div>

**Cross-modal transfer in Videos.** We are able to align a dataset of related videos without supervision. The alignment across videos enables transfer of annotations or other modalities from one video to another. For example, we can use this technique to transfer text annotations to an entire dataset of related videos by only labeling one video. One can also transfer other modalities associated with time like sound. We can <i>hallucinate</i> the sound of pouring liquids from one video to another purely on the basis of visual representations. We copy over the sound from the retrieved nearest neighbors and stitch the sounds together by simply concatenating the retrieved sounds. No other post-processing step is used. The results are in the supplementary material.

**Fine-grained retrieval in Videos.** We can use the nearest neighbours for fine-grained retrieval in a set of videos. In <a href="#fig:retrieval">Figure 7</a>, we show that we can retrieve frames when the glass is half full (Row 1) or when the hand has just placed the container back after pouring (Row 2). Note that in all retrieved examples, the liquid has already been transferred to the target container. For the <i>Baseball Pitch</i> class, the learned representations can even differentiate between the frames when the leg was up before the ball was pitched (Row 3) and after the ball was pitched (Row 4). 

**Anomaly detection.** Since we have well-behaved nearest neighbors in the TCC embedding space, we can use the distance from an <i>ideal</i> trajectory in this space to detect anomalous activities in videos. If a video's trajectory in the embedding space deviates too much from the ideal trajectory, we can mark those frames as anomalous. We present an example of a video of a person attempting to bench-press in <a href="#fig:anomaly">Figure 8</a>. In the beginning the distance of the nearest neighbor is quite low. But as the video progresses, we observe a sudden spike in this distance (around the $20^{th}$ frame) where the person's activity is very different from the ideal bench-press trajectory.

**Synchronous Playback.** Using the learned alignments, we can transfer the pace of a video to other videos of the same action. We include examples of different videos playing synchronously in the supplementary material.

## Conclusion

In this paper, we present a self-supervised learning approach that is able to learn features useful for temporally fine-grained tasks. In multiple experiments, we find self-supervised features lead to significant performance boosts when there is a lack of labeled data. With only one labeled video, TCC achieves similar performance to supervised learning models trained with about 50 videos. Additionally, TCC is more than a proxy task for representation learning. It serves as a general-purpose temporal alignment method that works without labels and benefits any task (like annotation transfer) which relies on the alignment itself.
