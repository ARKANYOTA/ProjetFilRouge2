<!-- page 1 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                   1

      Gradient-Based Learning Applied to Document
                      Recognition
                         Yann LeCun, L
                                      eon Bottou, Yoshua Bengio, and Patrick Haner

  Abstract |                                                                                I. Introduction
   Multilayer Neural Networks trained with the backpropa-
gation algorithm constitute the best example of a successful              Over the last several years, machine learning techniques,
Gradient-Based Learning technique. Given an appropriate                particularly when applied to neural networks, have played
network architecture, Gradient-Based Learning algorithms               an increasingly important role in the design of pattern
can be used to synthesize a complex decision surface that can
classify high-dimensional patterns such as handwritten char-           recognition systems. In fact, it could be argued that the
acters, with minimal preprocessing. This paper reviews var-            availability of learning techniques has been a crucial fac-
ious methods applied to handwritten character recognition              tor in the recent success of pattern recognition applica-
and compares them on a standard handwritten digit recog-
nition task. Convolutional Neural Networks, that are specif-           tions such as continuous speech recognition and handwrit-
ically designed to deal with the variability of 2D shapes, are         ing recognition.
shown to outperform all other techniques.                                 The main message of this paper is that better pattern
   Real-life document recognition systems are composed
of multiple modules including eld extraction, segmenta-               recognition systems can be built by relying more on auto-
tion, recognition, and language modeling. A new learning               matic learning, and less on hand-designed heuristics. This
paradigm, called Graph Transformer Networks (GTN), al-                 is made possible by recent progress in machine learning
lows such multi-module systems to be trained globally using            and computer technology. Using character recognition as
Gradient-Based methods so as to minimize an overall per-
formance measure.                                                      a case study, we show that hand-crafted feature extrac-
   Two systems for on-line handwriting recognition are de-             tion can be advantageously replaced by carefully designed
scribed. Experiments demonstrate the advantage of global               learning machines that operate directly on pixel images.
training, and the exibility of Graph Transformer Networks.             Using document understanding as a case study, we show
   A Graph Transformer Network for reading bank check is
also described. It uses Convolutional Neural Network char-             that the traditional way of building recognition systems by
acter recognizers combined with global training techniques             manually integrating individually designed modules can be
to provides record accuracy on business and personal checks.           replaced by a unied and well-principled design paradigm,
It is deployed commercially and reads several million checks           called Graph Transformer Networks, that allows training
per day.
   Keywords | Neural Networks, OCR, Document Recogni-                  all the modules to optimize a global performance criterion.
tion, Machine Learning, Gradient-Based Learning, Convo-                   Since the early days of pattern recognition it has been
lutional Neural Networks, Graph Transformer Networks, Fi-              known that the variability and richness of natural data,
nite State Transducers.                                                be it speech, glyphs, or other types of patterns, make it
                                                                       almost impossible to build an accurate recognition system
                        Nomenclature                                   entirely by hand. Consequently, most pattern recognition
  GT Graph transformer.                                                systems are built using a combination of automatic learn-
  GTN Graph transformer network.                                       ing techniques and hand-crafted algorithms. The usual
  HMM Hidden Markov model.                                             method of recognizing individual patterns consists in divid-
  HOS Heuristic oversegmentation.                                      ing the system into two main modules shown in gure 1.
  K-NN K-nearest neighbor.                                             The rst module, called the feature extractor, transforms
  NN Neural network.                                                   the input patterns so that they can be represented by low-
  OCR Optical character recognition.                                   dimensional vectors or short strings of symbols that (a) can
  PCA Principal component analysis.                                    be easily matched or compared, and (b) are relatively in-
  RBF Radial basis function.                                           variant with respect to transformations and distortions of
  RS-SVM Reduced-set support vector method.                            the input patterns that do not change their nature. The
  SDNN Space displacement neural network.                              feature extractor contains most of the prior knowledge and
  SVM Support vector method.                                           is rather specic to the task. It is also the focus of most of
  TDNN Time delay neural network.                                      the design e ort, because it is often entirely hand-crafted.
  V-SVM Virtual support vector method.                                 The classier, on the other hand, is often general-purpose
                                                                       and trainable. One of the main problems with this ap-
  The authors are with the Speech and Image Pro-
                                                                       proach is that the recognition accuracy is largely deter-
cessing Services Research Laboratory,              AT&T Labs-          mined by the ability of the designer to come up with an
Research, 100 Schulz Drive Red Bank, NJ 07701. E-mail:                 appropriate set of features. This turns out to be a daunt-
fyann,leonb,yoshua,ha nerg@research.att.com.       Yoshua Bengio       ing task which, unfortunately, must be redone for each new
is also with the Departement d'Informatique et de Recherche            problem. A large amount of the pattern recognition liter-
Operationelle, Universite de Montreal, C.P. 6128 Succ. Centre-Ville,
2920 Chemin de la Tour, Montreal, Quebec, Canada H3C 3J7.              ature is devoted to describing and comparing the relative

<!-- page 2 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                 2

                           Class scores                              manipulate directed graphs. This leads to the concept of
                                                                     trainable Graph Transformer Network (GTN) also intro-
                 TRAINABLE CLASSIFIER MODULE                         duced in Section IV. Section V describes the now clas-
                                                                     sical method of heuristic over-segmentation for recogniz-
                          Feature vector
                                                                     ing words or other character strings. Discriminative and
                                                                     non-discriminative gradient-based techniques for training
                                                                     a recognizer at the word level without requiring manual
                 FEATURE EXTRACTION MODULE
                                                                     segmentation and labeling are presented in Section VI. Sec-
                                                                     tion VII presents the promising Space-Displacement Neu-
                            Raw input                                ral Network approach that eliminates the need for seg-
Fig. 1. Traditional pattern recognition is performed with two mod-
                                                                     mentation heuristics by scanning a recognizer at all pos-
    ules: a xed feature extractor, and a trainable classier.       sible locations on the input. In section VIII, it is shown
                                                                     that trainable Graph Transformer Networks can be for-
                                                                     mulated as multiple generalized transductions, based on a
merits of di erent feature sets for particular tasks.                general graph composition algorithm. The connections be-
   Historically, the need for appropriate feature extractors         tween GTNs and Hidden Markov Models, commonly used
was due to the fact that the learning techniques used by             in speech recognition is also treated. Section IX describes
the classiers were limited to low-dimensional spaces with           a globally trained GTN system for recognizing handwrit-
easily separable classes 1]. A combination of three factors          ing entered in a pen computer. This problem is known as
have changed this vision over the last decade. First, the            \on-line" handwriting recognition, since the machine must
availability of low-cost machines with fast arithmetic units         produce immediate feedback as the user writes. The core of
allows to rely more on brute-force \numerical" methods               the system is a Convolutional Neural Network. The results
than on algorithmic renements. Second, the availability             clearly demonstrate the advantages of training a recognizer
of large databases for problems with a large market and              at the word level, rather than training it on pre-segmented,
wide interest, such as handwriting recognition, has enabled          hand-labeled, isolated characters. Section X describes a
designers to rely more on real data and less on hand-crafted         complete GTN-based system for reading handwritten and
feature extraction to build recognition systems. The third           machine-printed bank checks. The core of the system is the
and very important factor is the availability of powerful ma-        Convolutional Neural Network called LeNet-5 described in
chine learning techniques that can handle high-dimensional           Section II. This system is in commercial use in the NCR
inputs and can generate intricate decision functions when            Corporation line of check recognition systems for the bank-
fed with these large data sets. It can be argued that the            ing industry. It is reading millions of checks per month in
recent progress in the accuracy of speech and handwriting            several banks across the United States.
recognition systems can be attributed in large part to an
increased reliance on learning techniques and large training         A. Learning from Data
data sets. As evidence to this fact, a large proportion of              There are several approaches to automatic machine
modern commercial OCR systems use some form of multi-                learning, but one of the most successful approaches, pop-
layer Neural Network trained with back-propagation.                  ularized in recent years by the neural network community,
   In this study, we consider the tasks of handwritten char-         can be called \numerical" or gradient-based learning. The
acter recognition (Sections I and II) and compare the per-           learning machine computes a function Y p = F (Z p  W )
formance of several learning techniques on a benchmark               where Z p is the p-th input pattern, and W represents the
data set for handwritten digit recognition (Section III).            collection of adjustable parameters in the system. In a
While more automatic learning is benecial, no learning              pattern recognition setting, the output Y p may be inter-
technique can succeed without a minimal amount of prior              preted as the recognized class label of pattern Z p , or as
knowledge about the task. In the case of multi-layer neu-            scores or probabilities associated with each class. A loss
ral networks, a good way to incorporate knowledge is to              function E p = D(Dp  F (W Z p )), measures the discrep-
tailor its architecture to the task. Convolutional Neu-              ancy between Dp , the \correct" or desired output for pat-
ral Networks 2] introduced in Section II are an exam-                tern Z p , and the output produced by the system. The
ple of specialized neural network architectures which in-            average loss function Etrain (W ) is the average of the er-
corporate knowledge about the invariances of 2D shapes               rors E p over a set of labeled examples called the training
by using local connection patterns, and by imposing con-             set f(Z  D ) ::::(Z P  DP )g. In the simplest setting, the
                                                                            1    1

straints on the weights. A comparison of several methods             learning problem consists in nding the value of W that
for isolated handwritten digit recognition is presented in           minimizes Etrain (W ). In practice, the performance of the
section III. To go from the recognition of individual char-          system on a training set is of little interest. The more rel-
acters to the recognition of words and sentences in docu-            evant measure is the error rate of the system in the eld,
ments, the idea of combining multiple modules trained to             where it would be used in practice. This performance is
reduce the overall error is introduced in Section IV. Rec-           estimated by measuring the accuracy on a set of samples
ognizing variable-length objects such as handwritten words           disjoint from the training set, called the test set. Much
using multi-module systems is best done if the modules               theoretical and experimental work 3], 4], 5] has shown

<!-- page 3 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                            3

that the gap between the expected error rate on the test         Hessian matrix as in Newton or Quasi-Newton methods.
set Etest and the error rate on the training set Etrain de-      The Conjugate Gradient method 8] can also be used.
creases with the number of training samples approximately        However, Appendix B shows that despite many claims
as                                                               to the contrary in the literature, the usefulness of these
                 Etest ; Etrain = k(h=P )                (1)    second-order methods to large learning machines is very
                                                                 limited.
where P is the number of training samples, h is a measure of        A popular minimization procedure is the stochastic gra-
\e ective capacity" or complexity of the machine 6], 7],        dient algorithm, also called the on-line update. It consists
is a number between 0:5 and 1:0, and k is a constant. This       in updating the parameter vector using a noisy, or approx-
gap always decreases when the number of training samples         imated, version of the average gradient. In the most com-
increases. Furthermore, as the capacity h increases, Etrain      mon instance of it, W is updated on the basis of a single
decreases. Therefore, when increasing the capacity h, there      sample:
is a trade-o between the decrease of Etrain and the in-                                              pk
crease of the gap, with an optimal value of the capacity h                       Wk = Wk; ;  @E@W(W )
                                                                                            1                             (3)
that achieves the lowest generalization error Etest . Most
learning algorithms attempt to minimize Etrain as well as        With this procedure the parameter vector uctuates
some estimate of the gap. A formal version of this is called     around an average trajectory, but usually converges consid-
structural risk minimization 6], 7], and is based on den-       erably faster than regular gradient descent and second or-
ing a sequence of learning machines of increasing capacity,      der methods on large training sets with redundant samples
corresponding to a sequence of subsets of the parameter          (such as those encountered in speech or character recogni-
space such that each subset is a superset of the previous        tion). The reasons for this are explained in Appendix B.
subset. In practical terms, Structural Risk Minimization         The properties of such algorithms applied to learning have
is implemented by minimizing Etrain + H (W ), where the         been studied theoretically since the 1960's 9], 10], 11],
function H (W ) is called a regularization function, and  is    but practical successes for non-trivial tasks did not occur
a constant. H (W ) is chosen such that it takes large val-       until the mid eighties.
ues on parameters W that belong to high-capacity subsets         C. Gradient Back-Propagation
of the parameter space. Minimizing H (W ) in e ect lim-
its the capacity of the accessible subset of the parameter          Gradient-Based Learning procedures have been used
space, thereby controlling the tradeo between minimiz-           since the late 1950's, but they were mostly limited to lin-
ing the training error and minimizing the expected gap           ear systems 1]. The surprising usefulness of such sim-
between the training error and test error.                       ple gradient descent techniques for complex machine learn-
                                                                 ing tasks was not widely realized until the following three
B. Gradient-Based Learning                                       events occurred. The rst event was the realization that,
   The general problem of minimizing a function with re-         despite early warnings to the contrary 12], the presence
spect to a set of parameters is at the root of many issues in    of local minima in the loss function does not seem to
computer science. Gradient-Based Learning draws on the           be a major problem in practice. This became apparent
fact that it is generally much easier to minimize a reason-      when it was noticed that local minima did not seem to
ably smooth, continuous function than a discrete (combi-         be a major impediment to the success of early non-linear
natorial) function. The loss function can be minimized by        gradient-based Learning techniques such as Boltzmann ma-
estimating the impact of small variations of the parame-         chines 13], 14]. The second event was the popularization
ter values on the loss function. This is measured by the         by Rumelhart, Hinton and Williams 15] and others of a
gradient of the loss function with respect to the param-         simple and ecient procedure, the back-propagation al-
eters. Ecient learning algorithms can be devised when           gorithm, to compute the gradient in a non-linear system
the gradient vector can be computed analytically (as op-         composed of several layers of processing. The third event
posed to numerically through perturbations). This is the         was the demonstration that the back-propagation proce-
basis of numerous gradient-based learning algorithms with        dure applied to multi-layer neural networks with sigmoidal
continuous-valued parameters. In the procedures described        units can solve complicated learning tasks. The basic idea
in this article, the set of parameters W is a real-valued vec-   of back-propagation is that gradients can be computed e-
tor, with respect to which E (W ) is continuous, as well as      ciently by propagation from the output to the input. This
di erentiable almost everywhere. The simplest minimiza-          idea was described in the control theory literature of the
tion procedure in such a setting is the gradient descent         early sixties 16], but its application to machine learning
algorithm where W is iteratively adjusted as follows:            was not generally realized then. Interestingly, the early
                                                                 derivations of back-propagation in the context of neural
                                                                 network learning did not use gradients, but \virtual tar-
                  Wk = Wk; ;  @E@W
                                  (W ) :                  (2)    gets" for units in intermediate layers 17], 18], or minimal
                                                                 disturbance arguments 19]. The Lagrange formalism used
In the simplest case,  is a scalar constant. More sophisti-     in the control theory literature provides perhaps the best
cated procedures use variable , or substitute it for a diag-    rigorous method for deriving back-propagation 20], and for
onal matrix, or substitute it for an estimate of the inverse     deriving generalizations of back-propagation to recurrent

<!-- page 4 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                4

networks 21], and networks of heterogeneous modules 22].          ferentiable, and therefore lends itself to the use of Gradient-
A simple derivation for generic multi-layer systems is given      Based Learning methods. Section V introduces the use of
in Section I-E.                                                   directed acyclic graphs whose arcs carry numerical infor-
   The fact that local minima do not seem to be a problem         mation as a way to represent the alternative hypotheses,
for multi-layer neural networks is somewhat of a theoretical      and introduces the idea of GTN.
mystery. It is conjectured that if the network is oversized          The second solution described in Section VII is to elim-
for the task (as is usually the case in practice), the presence   inate segmentation altogether. The idea is to sweep the
of \extra dimensions" in parameter space reduces the risk         recognizer over every possible location on the input image,
of unattainable regions. Back-propagation is by far the           and to rely on the \character spotting" property of the rec-
most widely used neural-network learning algorithm, and           ognizer, i.e. its ability to correctly recognize a well-centered
probably the most widely used learning algorithm of any           character in its input eld, even in the presence of other
form.                                                             characters besides it, while rejecting images containing no
                                                                  centered characters 26], 27]. The sequence of recognizer
D. Learning in Real Handwriting Recognition Systems               outputs obtained by sweeping the recognizer over the in-
   Isolated handwritten character recognition has been ex-        put is then fed to a Graph Transformer Network that takes
tensively studied in the literature (see 23], 24] for reviews),   linguistic constraints into account and nally extracts the
and was one of the early successful applications of neural        most likely interpretation. This GTN is somewhat similar
networks 25]. Comparative experiments on recognition of           to Hidden Markov Models (HMM), which makes the ap-
individual handwritten digits are reported in Section III.        proach reminiscent of the classical speech recognition 28],
They show that neural networks trained with Gradient-              29]. While this technique would be quite expensive in
Based Learning perform better than all other methods              the general case, the use of Convolutional Neural Networks
tested here on the same data. The best neural networks,           makes it particularly attractive because it allows signicant
called Convolutional Networks, are designed to learn to           savings in computational cost.
extract relevant features directly from pixel images (see
Section II).                                                      E. Globally Trainable Systems
   One of the most dicult problems in handwriting recog-            As stated earlier, most practical pattern recognition sys-
nition, however, is not only to recognize individual charac-      tems are composed of multiple modules. For example, a
ters, but also to separate out characters from their neigh-       document recognition system is composed of a eld locator,
bors within the word or sentence, a process known as seg-         which extracts regions of interest, a eld segmenter, which
mentation. The technique for doing this that has become           cuts the input image into images of candidate characters, a
the \standard" is called Heuristic Over-Segmentation. It          recognizer, which classies and scores each candidate char-
consists in generating a large number of potential cuts           acter, and a contextual post-processor, generally based on
between characters using heuristic image processing tech-         a stochastic grammar, which selects the best grammatically
niques, and subsequently selecting the best combination of        correct answer from the hypotheses generated by the recog-
cuts based on scores given for each candidate character by        nizer. In most cases, the information carried from module
the recognizer. In such a model, the accuracy of the sys-         to module is best represented as graphs with numerical in-
tem depends upon the quality of the cuts generated by the         formation attached to the arcs. For example, the output
heuristics, and on the ability of the recognizer to distin-       of the recognizer module can be represented as an acyclic
guish correctly segmented characters from pieces of char-         graph where each arc contains the label and the score of
acters, multiple characters, or otherwise incorrectly seg-        a candidate character, and where each path represent a
mented characters. Training a recognizer to perform this          alternative interpretation of the input string. Typically,
task poses a major challenge because of the diculty in cre-      each module is manually optimized, or sometimes trained,
ating a labeled database of incorrectly segmented charac-         outside of its context. For example, the character recog-
ters. The simplest solution consists in running the images        nizer would be trained on labeled images of pre-segmented
of character strings through the segmenter, and then man-         characters. Then the complete system is assembled, and
ually labeling all the character hypotheses. Unfortunately,       a subset of the parameters of the modules is manually ad-
not only is this an extremely tedious and costly task, it is      justed to maximize the overall performance. This last step
also dicult to do the labeling consistently. For example,        is extremely tedious, time-consuming, and almost certainly
should the right half of a cut up 4 be labeled as a 1 or as       suboptimal.
a non-character? should the right half of a cut up 8 be              A better alternative would be to somehow train the en-
labeled as a 3?                                                   tire system so as to minimize a global error measure such as
   The rst solution, described in Section V consists in          the probability of character misclassications at the docu-
training the system at the level of whole strings of char-        ment level. Ideally, we would want to nd a good minimum
acters, rather than at the character level. The notion of         of this global loss function with respect to all the param-
Gradient-Based Learning can be used for this purpose. The         eters in the system. If the loss function E measuring the
system is trained to minimize an overall loss function which      performance can be made di erentiable with respect to the
measures the probability of an erroneous answer. Section V        system's tunable parameters W , we can nd a local min-
explores various ways to ensure that the loss function is dif-    imum of E using Gradient-Based Learning. However, at

<!-- page 5 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                5

rst glance, it appears that the sheer size and complexity          tion system is best represented by graphs with numerical
of the system would make this intractable.                          information attached to the arcs. In this case, each module,
   To ensure that the global loss function E p (Z p  W ) is dif-   called a Graph Transformer, takes one or more graphs as
ferentiable, the overall system is built as a feed-forward net-     input, and produces a graph as output. Networks of such
work of di erentiable modules. The function implemented             modules are called Graph Transformer Networks (GTN).
by each module must be continuous and di erentiable al-             Sections IV, VI and VIII develop the concept of GTNs,
most everywhere with respect to the internal parameters of          and show that Gradient-Based Learning can be used to
the module (e.g. the weights of a Neural Net character rec-         train all the parameters in all the modules so as to mini-
ognizer in the case of a character recognition module), and         mize a global loss function. It may seem paradoxical that
with respect to the module's inputs. If this is the case, a         gradients can be computed when the state information is
simple generalization of the well-known back-propagation            represented by essentially discrete objects such as graphs,
procedure can be used to eciently compute the gradients            but that diculty can be circumvented, as shown later.
of the loss function with respect to all the parameters in
the system 22]. For example, let us consider a system                    II. Convolutional Neural Networks for
built as a cascade of modules, each of which implements a                      Isolated Character Recognition
function Xn = Fn (Wn  Xn; ), where Xn is a vector rep-
                                                                       The ability of multi-layer networks trained with gradi-
resenting the output of the module, Wn is the vector of             ent descent to learn complex, high-dimensional, non-linear
tunable parameters in the module (a subset of W ), and              mappings from large collections of examples makes them
Xn; is the module's input vector (as well as the previous
                                                                    obvious candidates for image recognition tasks. In the tra-
module's output vector). The input X to the rst module             ditional model of pattern recognition, a hand-designed fea-
is the input pattern Z p . If the partial derivative of E p with

                                                                    ture extractor gathers relevant information from the input
respect to Xn is known, then the partial derivatives of E p         and eliminates irrelevant variabilities. A trainable classier
with respect to Wn and Xn; can be computed using the
                                                                    then categorizes the resulting feature vectors into classes.
backward recurrence                                                 In this scheme, standard, fully-connected multi-layer net-
              @E p = @F (W  X ) @E p                               works can be used as classiers. A potentially more inter-
              @Wn          @W n n; @Xn         1
                                                                    esting scheme is to rely on as much as possible on learning
             @E p = @F (W  X ) @E p                                in the feature extractor itself. In the case of character
            @Xn;           @X n n; @Xn         1        (4)         recognition, a network could be fed with almost raw in-
                                                                    puts (e.g. size-normalized images). While this can be done

where @W@F (Wn  Xn; ) is the Jacobian of F with respect to
                       1                                            with an ordinary fully connected feed-forward network with
W evaluated at the point (Wn  Xn; ), and @X
                                            @F (Wn  Xn; )    1     some success for tasks such as character recognition, there
is the Jacobian of F with respect to X . The Jacobian of            are problems.
a vector function is a matrix containing the partial deriva-           Firstly, typical images are large, often with several hun-
tives of all the outputs with respect to all the inputs.            dred variables (pixels). A fully-connected rst layer with,
The rst equation computes some terms of the gradient               say one hundred hidden units in the rst layer, would al-
of E p (W ), while the second equation generates a back-            ready contain several tens of thousands of weights. Such
ward recurrence, as in the well-known back-propagation              a large number of parameters increases the capacity of the
procedure for neural networks. We can average the gradi-            system and therefore requires a larger training set. In ad-
ents over the training patterns to obtain the full gradient.        dition, the memory requirement to store so many weights
It is interesting to note that in many instances there is           may rule out certain hardware implementations. But, the
no need to explicitly compute the Jacobian matrix. The              main deciency of unstructured nets for image or speech
above formula uses the product of the Jacobian with a vec-          applications is that they have no built-in invariance with
tor of partial derivatives, and it is often easier to compute       respect to translations, or local distortions of the inputs.
this product directly without computing the Jacobian be-            Before being sent to the xed-size input layer of a neural
forehand. In By analogy with ordinary multi-layer neural            net, character images, or other 2D or 1D signals, must be
networks, all but the last module are called hidden layers          approximately size-normalized and centered in the input
because their outputs are not observable from the outside.          eld. Unfortunately, no such preprocessing can be perfect:
more complex situations than the simple cascade of mod-             handwriting is often normalized at the word level, which
ules described above, the partial derivative notation be-           can cause size, slant, and position variations for individual
comes somewhat ambiguous and awkward. A completely                  characters. This, combined with variability in writing style,
rigorous derivation in more general cases can be done using         will cause variations in the position of distinctive features
Lagrange functions 20], 21], 22].                                   in input objects. In principle, a fully-connected network of
   Traditional multi-layer neural networks are a special case       sucient size could learn to produce outputs that are in-
of the above where the state information Xn is represented          variant with respect to such variations. However, learning
with xed-sized vectors, and where the modules are al-              such a task would probably result in multiple units with
ternated layers of matrix multiplications (the weights) and         similar weight patterns positioned at various locations in
component-wise sigmoid functions (the neurons). However,            the input so as to detect distinctive features wherever they
as stated earlier, the state information in complex recogni-        appear on the input. Learning these weight congurations

<!-- page 6 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                             6

requires a very large number of training instances to cover      planes, each of which is a feature map. A unit in a feature
the space of possible variations. In convolutional networks,     map has 25 inputs connected to a 5 by 5 area in the input,
described below, shift invariance is automatically obtained      called the receptive eld of the unit. Each unit has 25 in-
by forcing the replication of weight congurations across        puts, and therefore 25 trainable coecients plus a trainable
space.                                                           bias. The receptive elds of contiguous units in a feature
   Secondly, a deciency of fully-connected architectures is     map are centered on correspondingly contiguous units in
that the topology of the input is entirely ignored. The in-      the previous layer. Therefore receptive elds of neighbor-
put variables can be presented in any (xed) order without       ing units overlap. For example, in the rst hidden layer
a ecting the outcome of the training. On the contrary,           of LeNet-5, the receptive elds of horizontally contiguous
images (or time-frequency representations of speech) have        units overlap by 4 columns and 5 rows. As stated earlier,
a strong 2D local structure: variables (or pixels) that are      all the units in a feature map share the same set of 25
spatially or temporally nearby are highly correlated. Local      weights and the same bias so they detect the same feature
correlations are the reasons for the well-known advantages       at all possible locations on the input. The other feature
of extracting and combining local features before recogniz-      maps in the layer use di erent sets of weights and biases,
ing spatial or temporal objects, because congurations of        thereby extracting di erent types of local features. In the
neighboring variables can be classied into a small number       case of LeNet-5, at each input location six di erent types
of categories (e.g. edges, corners...). Convolutional Net-       of features are extracted by six units in identical locations
works force the extraction of local features by restricting      in the six feature maps. A sequential implementation of
the receptive elds of hidden units to be local.                 a feature map would scan the input image with a single
                                                                 unit that has a local receptive eld, and store the states
A. Convolutional Networks                                        of this unit at corresponding locations in the feature map.
   Convolutional Networks combine three architectural            This operation is equivalent to a convolution, followed by
ideas to ensure some degree of shift, scale, and distor-         an additive bias and squashing function, hence the name
tion invariance: local receptive elds, shared weights (or       convolutional network. The kernel of the convolution is the
weight replication), and spatial or temporal sub-sampling.       set of connection weights used by the units in the feature
A typical convolutional network for recognizing characters,      map. An interesting property of convolutional layers is that
dubbed LeNet-5, is shown in gure 2. The input plane             if the input image is shifted, the feature map output will
receives images of characters that are approximately size-       be shifted by the same amount, but will be left unchanged
normalized and centered. Each unit in a layer receives in-       otherwise. This property is at the basis of the robustness
puts from a set of units located in a small neighborhood         of convolutional networks to shifts and distortions of the
in the previous layer. The idea of connecting units to local     input.
receptive elds on the input goes back to the Perceptron in         Once a feature has been detected, its exact location
the early 60s, and was almost simultaneous with Hubel and        becomes less important. Only its approximate position
Wiesel's discovery of locally-sensitive, orientation-selective   relative to other features is relevant. For example, once
neurons in the cat's visual system 30]. Local connections        we know that the input image contains the endpoint of a
have been used many times in neural models of visual learn-      roughly horizontal segment in the upper left area, a corner
ing 31], 32], 18], 33], 34], 2]. With local receptive            in the upper right area, and the endpoint of a roughly ver-
elds, neurons can extract elementary visual features such       tical segment in the lower portion of the image, we can tell
as oriented edges, end-points, corners (or similar features in   the input image is a 7. Not only is the precise position of
other signals such as speech spectrograms). These features       each of those features irrelevant for identifying the pattern,
are then combined by the subsequent layers in order to de-       it is potentially harmful because the positions are likely to
tect higher-order features. As stated earlier, distortions or    vary for di erent instances of the character. A simple way
shifts of the input can cause the position of salient features   to reduce the precision with which the position of distinc-
to vary. In addition, elementary feature detectors that are      tive features are encoded in a feature map is to reduce the
useful on one part of the image are likely to be useful across   spatial resolution of the feature map. This can be achieved
the entire image. This knowledge can be applied by forcing       with a so-called sub-sampling layers which performs a local
a set of units, whose receptive elds are located at di erent    averaging and a sub-sampling, reducing the resolution of
places on the image, to have identical weight vectors 32],       the feature map, and reducing the sensitivity of the output
 15], 34]. Units in a layer are organized in planes within       to shifts and distortions. The second hidden layer of LeNet-
which all the units share the same set of weights. The set       5 is a sub-sampling layer. This layer comprises six feature
of outputs of the units in such a plane is called a feature      maps, one for each feature map in the previous layer. The
map. Units in a feature map are all constrained to per-          receptive eld of each unit is a 2 by 2 area in the previous
form the same operation on di erent parts of the image.          layer's corresponding feature map. Each unit computes the
A complete convolutional layer is composed of several fea-       average of its four inputs, multiplies it by a trainable coef-
ture maps (with di erent weight vectors), so that multiple       cient, adds a trainable bias, and passes the result through
features can be extracted at each location. A concrete ex-       a sigmoid function. Contiguous units have non-overlapping
ample of this is the rst layer of LeNet-5 shown in Figure 2.    contiguous receptive elds. Consequently, a sub-sampling
Units in the rst hidden layer of LeNet-5 are organized in 6     layer feature map has half the number of rows and columns

<!-- page 7 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                            7

                                                C3: f. maps 16@10x10
                          C1: feature maps                        S4: f. maps 16@5x5
       INPUT              6@28x28
       32x32                                   S2: f. maps                           C5: layer F6: layer   OUTPUT
                                               6@14x14                               120                   10

                                                                                           Full connection       Gaussian connections
                  Convolutions               Subsampling        Convolutions     Subsampling        Full connection

Fig. 2. Architecture of LeNet-5, a Convolutional Neural Network, here for digits recognition. Each plane is a feature map, i.e. a set of units
    whose weights are constrained to be identical.

as the feature maps in the previous layer. The trainable                B. LeNet-5
coecient and bias control the e ect of the sigmoid non-
linearity. If the coecient is small, then the unit operates               This section describes in more detail the architecture of
in a quasi-linear mode, and the sub-sampling layer merely               LeNet-5, the Convolutional Neural Network used in the
blurs the input. If the coecient is large, sub-sampling                experiments. LeNet-5 comprises 7 layers, not counting the
units can be seen as performing a \noisy OR" or a \noisy                input, all of which contain trainable parameters (weights).
AND" function depending on the value of the bias. Succes-               The input is a 32x32 pixel image. This is signicantly larger
sive layers of convolutions and sub-sampling are typically              than the largest character in the database (at most 20x20
alternated, resulting in a \bi-pyramid": at each layer, the             pixels centered in a 28x28 eld). The reason is that it is
number of feature maps is increased as the spatial resolu-              desirable that potential distinctive features such as stroke
tion is decreased. Each unit in the third hidden layer in g-           end-points or corner can appear in the center of the recep-
ure 2 may have input connections from several feature maps              tive eld of the highest-level feature detectors. In LeNet-5
in the previous layer. The convolution/sub-sampling com-                the set of centers of the receptive elds of the last convolu-
bination, inspired by Hubel and Wiesel's notions of \sim-               tional layer (C3, see below) form a 20x20 area in the center
ple" and \complex" cells, was implemented in Fukushima's                of the 32x32 input. The values of the input pixels are nor-
Neocognitron 32], though no globally supervised learning                malized so that the background level (white) corresponds
procedure such as back-propagation was available then. A                to a value of -0.1 and the foreground (black) corresponds
large degree of invariance to geometric transformations of              to 1.175. This makes the mean input roughly 0, and the
the input can be achieved with this progressive reduction               variance roughly 1 which accelerates learning 46].
of spatial resolution compensated by a progressive increase                In the following, convolutional layers are labeled Cx, sub-
of the richness of the representation (the number of feature            sampling layers are labeled Sx, and fully-connected layers
maps).                                                                  are labeled Fx, where x is the layer index.
   Since all the weights are learned with back-propagation,                Layer C1 is a convolutional layer with 6 feature maps.
convolutional networks can be seen as synthesizing their                Each unit in each feature map is connected to a 5x5 neigh-
own feature extractor. The weight sharing technique has                 borhood in the input. The size of the feature maps is 28x28
the interesting side e ect of reducing the number of free               which prevents connection from the input from falling o
parameters, thereby reducing the \capacity" of the ma-                  the boundary. C1 contains 156 trainable parameters, and
chine and reducing the gap between test error and training              122,304 connections.
error 34]. The network in gure 2 contains 340,908 con-                    Layer S2 is a sub-sampling layer with 6 feature maps of
nections, but only 60,000 trainable free parameters because             size 14x14. Each unit in each feature map is connected to a
of the weight sharing.                                                  2x2 neighborhood in the corresponding feature map in C1.
   Fixed-size Convolutional Networks have been applied                  The four inputs to a unit in S2 are added, then multiplied
to many applications, among other handwriting recogni-                  by a trainable coecient, and added to a trainable bias.
tion 35], 36], machine-printed character recognition 37],               The result is passed through a sigmoidal function. The
on-line handwriting recognition 38], and face recogni-                  2x2 receptive elds are non-overlapping, therefore feature
tion 39]. Fixed-size convolutional networks that share                  maps in S2 have half the number of rows and column as
weights along a single temporal dimension are known as                  feature maps in C1. Layer S2 has 12 trainable parameters
Time-Delay Neural Networks (TDNNs). TDNNs have been                     and 5,880 connections.
used in phoneme recognition (without sub-sampling) 40],                    Layer C3 is a convolutional layer with 16 feature maps.
 41], spoken word recognition (with sub-sampling) 42],                  Each unit in each feature map is connected to several 5x5
 43], on-line recognition of isolated handwritten charac-               neighborhoods at identical locations in a subset of S2's
ters 44], and signature verication 45].                                feature maps. Table I shows the set of S2 feature maps

<!-- page 8 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                             8

   0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15                        where A is the amplitude of the function and S determines
 0 X       X X X     X X X X       X X                          its slope at the origin. The function f is odd, with horizon-
 1 X X       X X X     X X X X        X                         tal asymptotes at +A and ;A. The constant A is chosen
 2 X X X       X X X      X     X X X                           to be 1:7159. The rationale for this choice of a squashing
 3   X X X     X X X X       X     X X                          function is given in Appendix A.
 4     X X X     X X X X     X X      X                            Finally, the output layer is composed of Euclidean Radial
 5       X X X     X X X X      X X X                           Basis Function units (RBF), one for each class, with 84
                          TABLE I                               inputs each. The outputs of each RBF unit yi is computed
Each column indicates which feature map in S2 are combined      as follows:               X
                                                                                     yi = (xj ; wij ) : 2
                                                                                                                          (7)
      by the units in a particular feature map of C3.
                                                                                           j
                                                                In other words, each output RBF unit computes the Eu-
                                                                clidean distance between its input vector and its parameter
combined by each C3 feature map. Why not connect ev-            vector. The further away is the input from the parameter
ery S2 feature map to every C3 feature map? The rea-            vector, the larger is the RBF output. The output of a
son is twofold. First, a non-complete connection scheme         particular RBF can be interpreted as a penalty term mea-
keeps the number of connections within reasonable bounds.       suring the t between the input pattern and a model of the
More importantly, it forces a break of symmetry in the net-     class associated with the RBF. In probabilistic terms, the
work. Di erent feature maps are forced to extract di erent      RBF output can be interpreted as the unnormalized nega-
(hopefully complementary) features because they get dif-        tive log-likelihood of a Gaussian distribution in the space
ferent sets of inputs. The rationale behind the connection      of congurations of layer F6. Given an input pattern, the
scheme in table I is the following. The rst six C3 feature     loss function should be designed so as to get the congu-
maps take inputs from every contiguous subsets of three         ration of F6 as close as possible to the parameter vector
feature maps in S2. The next six take input from every          of the RBF that corresponds to the pattern's desired class.
contiguous subset of four. The next three take input from       The parameter vectors of these units were chosen by hand
some discontinuous subsets of four. Finally the last one        and kept xed (at least initially). The components of those
takes input from all S2 feature maps. Layer C3 has 1,516        parameters vectors were set to -1 or +1. While they could
trainable parameters and 151,600 connections.                   have been chosen at random with equal probabilities for -1
   Layer S4 is a sub-sampling layer with 16 feature maps of     and +1, or even chosen to form an error correcting code
size 5x5. Each unit in each feature map is connected to a       as suggested by 47], they were instead designed to repre-
2x2 neighborhood in the corresponding feature map in C3,        sent a stylized image of the corresponding character class
in a similar way as C1 and S2. Layer S4 has 32 trainable        drawn on a 7x12 bitmap (hence the number 84). Such a
parameters and 2,000 connections.                               representation is not particularly useful for recognizing iso-
   Layer C5 is a convolutional layer with 120 feature maps.     lated digits, but it is quite useful for recognizing strings of
Each unit is connected to a 5x5 neighborhood on all 16          characters taken from the full printable ASCII set. The
of S4's feature maps. Here, because the size of S4 is also      rationale is that characters that are similar, and therefore
5x5, the size of C5's feature maps is 1x1: this amounts         confusable, such as uppercase O, lowercase O, and zero, or
to a full connection between S4 and C5. C5 is labeled           lowercase l, digit 1, square brackets, and uppercase I, will
as a convolutional layer, instead of a fully-connected layer,   have similar output codes. This is particularly useful if the
because if LeNet-5 input were made bigger with everything       system is combined with a linguistic post-processor that
else kept constant, the feature map dimension would be          can correct such confusions. Because the codes for confus-
larger than 1x1. This process of dynamically increasing the     able classes are similar, the output of the corresponding
size of a convolutional network is described in the section     RBFs for an ambiguous character will be similar, and the
Section VII. Layer C5 has 48,120 trainable connections.         post-processor will be able to pick the appropriate interpre-
   Layer F6, contains 84 units (the reason for this number      tation. Figure 3 gives the output codes for the full ASCII
comes from the design of the output layer, explained be-        set.
low) and is fully connected to C5. It has 10,164 trainable         Another reason for using such distributed codes, rather
parameters.                                                     than the more common \1 of N" code (also called place
   As in classical neural networks, units in layers up to F6    code, or grand-mother cell code) for the outputs is that
compute a dot product between their input vector and their      non distributed codes tend to behave badly when the num-
weight vector, to which a bias is added. This weighted sum,     ber of classes is larger than a few dozens. The reason is
denoted ai for unit i, is then passed through a sigmoid         that output units in a non-distributed code must be o
squashing function to produce the state of unit i, denoted      most of the time. This is quite dicult to achieve with
by xi :                                                         sigmoid units. Yet another reason is that the classiers are
                          xi = f (ai )                    (5)   often used to not only recognize characters, but also to re-
The squashing function is a scaled hyperbolic tangent:          ject non-characters. RBFs with distributed codes are more
                                                                appropriate for that purpose because unlike sigmoids, they
                    f (a) = A tanh(Sa)                   (6)    are activated within a well circumscribed region of their in-

<!-- page 9 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                9

                                                                                penalties, it means that in addition to pushing down the
       !   "   #    $   %   &    ’       (    )   *   +   ,   −   .   /
                                                                                penalty of the correct class like the MSE criterion, this
                                                                                criterion also pulls up the penalties of the incorrect classes:
                                                                                              XP (y (Z p W ) + log(e j + X e y Z W ))
   0   1   2   3    4   5   6    7       8    9   :   ;   <   =   >   ?

   @   A   B   C    D   E   F    G       H    I   J   K   L   M   N   O
                                                                                E (W ) = P1            Dp              ;           ; i( p   )

                                                                                              p   =1                           i
                                                                                                                                           (9)
   P   Q   R   S    T   U   V    W       X    Y   Z   [   \   ]   ^   _

   ‘   a   b   c    d   e   f    g       h    i   j   k   l   m   n   o         The negative of the second term plays a \competitive" role.
                                                                                It is necessarily smaller than (or equal to) the rst term,
   p   q   r   s    t   u   v    w       x    y   z   {   |   }   ~
                                                                                therefore this loss function is positive. The constant j is
Fig. 3. Initial parameters of the output RBFs for recognizing the               positive, and prevents the penalties of classes that are al-
    full ASCII set.                                                             ready very large from being pushed further up. The pos-
                                                                                terior probability of this P
                                                                                                           rubbish class label would be the
put space that non-typical patterns are more likely to fall                     ratio of e;j and e;j + i e;yi Z p W . This discrimina-
                                                                                                                   (       )

outside of.                                                                     tive criterion prevents the previously mentioned \collaps-
   The parameter vectors of the RBFs play the role of target                    ing e ect" when the RBF parameters are learned because
vectors for layer F6. It is worth pointing out that the com-                    it keeps the RBF centers apart from each other. In Sec-
ponents of those vectors are +1 or -1, which is well within                     tion VI, we present a generalization of this criterion for
the range of the sigmoid of F6, and therefore prevents those                    systems that learn to classify multiple objects in the input
sigmoids from getting saturated. In fact, +1 and -1 are the                     (e.g., characters in words or in documents).
points of maximum curvature of the sigmoids. This forces                           Computing the gradient of the loss function with respect
the F6 units to operate in their maximally non-linear range.                    to all the weights in all the layers of the convolutional
Saturation of the sigmoids must be avoided because it is                        network is done with back-propagation. The standard al-
known to lead to slow convergence and ill-conditioning of                       gorithm must be slightly modied to take account of the
the loss function.                                                              weight sharing. An easy way to implement it is to rst com-
                                                                                pute the partial derivatives of the loss function with respect
C. Loss Function                                                                to each connection, as if the network were a conventional
                                                                                multi-layer network without weight sharing. Then the par-
   The simplest output loss function that can be used with                      tial derivatives of all the connections that share a same
the above network is the Maximum Likelihood Estimation                          parameter are added to form the derivative with respect to
criterion (MLE), which in our case is equivalent to the Min-                    that parameter.
imum Mean Squared Error (MSE). The criterion for a set                             Such a large architecture can be trained very eciently,
of training samples is simply:                                                  but doing so requires the use of a few techniques that are

                   E (W ) = P1
                                 X y (Z p W )
                                     P
                                                                          (8)
                                                                                described in the appendix. Section A of the appendix
                                                                                describes details such as the particular sigmoid used, and
                                             Dp
                                 p                                              the weight initialization. Section B and C describe the
                                                                                minimization procedure used, which is a stochastic version
                                     =1

where yDp is the output of the Dp -th RBF unit, i.e. the                        of a diagonal approximation to the Levenberg-Marquardt
one that corresponds to the correct class of input pattern                      procedure.
Z p . While this cost function is appropriate for most cases,                        III. Results and Comparison with Other
it lacks three important properties. First, if we allow the
parameters of the RBF to adapt, E (W ) has a trivial, but                                              Methods
totally unacceptable, solution. In this solution, all the RBF                      While recognizing individual digits is only one of many
parameter vectors are equal, and the state of F6 is constant                    problems involved in designing a practical recognition sys-
and equal to that parameter vector. In this case the net-                       tem, it is an excellent benchmark for comparing shape
work happily ignores the input, and all the RBF outputs                         recognition methods. Though many existing method com-
are equal to zero. This collapsing phenomenon does not                          bine a hand-crafted feature extractor and a trainable clas-
occur if the RBF weights are not allowed to adapt. The                          sier, this study concentrates on adaptive methods that
second problem is that there is no competition between                          operate directly on size-normalized images.
the classes. Such a competition can be obtained by us-
ing a more discriminative training criterion, dubbed the                        A. Database: the Modied NIST set
MAP (maximum a posteriori) criterion, similar to Maxi-                             The database used to train and test the systems de-
mum Mutual Information criterion sometimes used to train                        scribed in this paper was constructed from the NIST's Spe-
HMMs 48], 49], 50]. It corresponds to maximizing the                            cial Database 3 and Special Database 1 containing binary
posterior probability of the correct class Dp (or minimiz-                      images of handwritten digits. NIST originally designated
ing the logarithm of the probability of the correct class),                     SD-3 as their training set and SD-1 as their test set. How-
given that the input image can come from one of the classes                     ever, SD-3 is much cleaner and easier to recognize than SD-
or from a background \rubbish" class label. In terms of                         1. The reason for this can be found on the fact that SD-3

<!-- page 10 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                10

was collected among Census Bureau employees, while SD-1
was collected among high-school students. Drawing sensi-
ble conclusions from learning experiments requires that the
result be independent of the choice of training set and test
among the complete set of samples. Therefore it was nec-
essary to build a new database by mixing NIST's datasets.
   SD-1 contains 58,527 digit images written by 500 dif-
ferent writers. In contrast to SD-3, where blocks of data
from each writer appeared in sequence, the data in SD-1 is
scrambled. Writer identities for SD-1 are available and we
used this information to unscramble the writers. We then
split SD-1 in two: characters written by the rst 250 writers
went into our new training set. The remaining 250 writers
were placed in our test set. Thus we had two sets with
nearly 30,000 examples each. The new training set was
completed with enough examples from SD-3, starting at
pattern # 0, to make a full set of 60,000 training patterns.
Similarly, the new test set was completed with SD-3 exam-
ples starting at pattern # 35,000 to make a full set with           Fig. 4. Size-normalized examples from the MNIST database.
60,000 test patterns. In the experiments described here, we
only used a subset of 10,000 test images (5,000 from SD-1
and 5,000 from SD-3), but we used the full 60,000 training       three, 0.0001 for the next three, 0.00005 for the next 4,
samples. The resulting database was called the Modied           and 0.00001 thereafter. Before each iteration, the diagonal
NIST, or MNIST, dataset.                                         Hessian approximation was reevaluated on 500 samples, as
   The original black and white (bilevel) images were size       described in Appendix C and kept xed during the entire
normalized to t in a 20x20 pixel box while preserving           iteration. The parameter was set to 0.02. The resulting
their aspect ratio. The resulting images contain grey lev-       e ective learning rates during the rst pass varied between
els as result of the anti-aliasing (image interpolation) tech-   approximately 7  10; and 0:016 over the set of parame-

nique used by the normalization algorithm. Three ver-            ters. The test error rate stabilizes after around 10 passes
sions of the database were used. In the rst version,            through the training set at 0.95%. The error rate on the
the images were centered in a 28x28 image by comput-             training set reaches 0.35% after 19 passes. Many authors
ing the center of mass of the pixels, and translating the        have reported observing the common phenomenon of over-
image so as to position this point at the center of the          training when training neural networks or other adaptive
28x28 eld. In some instances, this 28x28 eld was ex-           algorithms on various tasks. When over-training occurs,
tended to 32x32 with background pixels. This version of          the training error keeps decreasing over time, but the test
the database will be referred to as the regular database.        error goes through a minimum and starts increasing after
In the second version of the database, the character im-         a certain number of iterations. While this phenomenon is
ages were deslanted and cropped down to 20x20 pixels im-         very common, it was not observed in our case as the learn-
ages. The deslanting computes the second moments of in-          ing curves in gure 5 show. A possible reason is that the
ertia of the pixels (counting a foreground pixel as 1 and a      learning rate was kept relatively large. The e ect of this is
background pixel as 0), and shears the image by horizon-         that the weights never settle down in the local minimum
tally shifting the lines so that the principal axis is verti-    but keep oscillating randomly. Because of those uctua-
cal. This version of the database will be referred to as the     tions, the average cost will be lower in a broader minimum.
deslanted database. In the third version of the database,        Therefore, stochastic gradient will have a similar e ect as
used in some early experiments, the images were reduced          a regularization term that favors broader minima. Broader
to 16x16 pixels. The regular database (60,000 training           minima correspond to solutions with large entropy of the
examples, 10,000 test examples size-normalized to 20x20,         parameter distribution, which is benecial to the general-
and centered by center of mass in 28x28 elds) is avail-         ization error.
able at http://www.research.att.com/~yann/ocr/mnist.                The inuence of the training set size was measured by
Figure 4 shows examples randomly picked from the test set.       training the network with 15,000, 30,000, and 60,000 exam-
                                                                 ples. The resulting training error and test error are shown
B. Results                                                       in gure 6. It is clear that, even with specialized architec-
  Several versions of LeNet-5 were trained on the regular        tures such as LeNet-5, more training data would improve
MNIST database. 20 iterations through the entire train-          the accuracy.
ing data were performed for each session. The values of             To verify this hypothesis, we articially generated more
the global learning rate (see Equation 21 in Appendix C          training examples by randomly distorting the original
for a denition) was decreased using the following sched-        training images. The increased training set was composed
ule: 0.0005 for the rst two passes, 0.0002 for the next         of the 60,000 original patterns plus 540,000 instances of

<!-- page 11 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                                    11

                   Error Rate (%)

         5%

         4%

         3%

         2%

         1%
                                                                           Test
         0%
                                                                           Training
              0           4             8         12         16      20
                                                         Training set Iterations

Fig. 5. Training and test error of LeNet-5 as a function of the num-
    ber of passes through the 60,000 pattern training set (without
    distortions). The average training error is measured on-the-y as
    training proceeds. This explains why the training error appears
    to be larger than the test error. Convergence is attained after 10
    to 12 passes through the training set.
                                                                                             Fig. 7. Examples of distortions of ten training patterns.
               Error Rate (%)
         1.8

         1.6
                                                                                              4−>6   3−>5   8−>2   2−>1   5−>3   4−>8   2−>8   3−>5   6−>5   7−>3

         1.4                                                                                  9−>4   8−>0   7−>8   5−>3   8−>7   0−>6   3−>7   2−>7   8−>3   9−>4
                                        Test error (no distortions)
         1.2
                                                                                              8−>2   5−>3   4−>8   3−>9   6−>0   9−>8   4−>9   6−>1   9−>4   9−>1

                                                                                              9−>4   2−>0   6−>1   3−>5   3−>2   9−>5   6−>0   6−>0   6−>0   6−>8
         0.8
                                                                   Test error
                                                                                              4−>6   7−>3   9−>4   4−>6   2−>7   9−>7   4−>3   9−>4   9−>4   9−>4
                                                                   (with distortions)
         0.6

                                                                                              8−>7   4−>2   8−>4   3−>5   8−>4   6−>5   8−>5   3−>8   3−>8   9−>8
         0.4

         0.2
                                            Training error (no distortions)                   1−>5   9−>8   6−>3   0−>2   6−>5   9−>5   0−>7   1−>6   4−>9   2−>1

          0                                                                                   2−>8   8−>5   4−>9   7−>2   7−>2   6−>5   9−>7   6−>1   5−>6   5−>0
               0     10       20   30        40   50    60    70   80     90   100

                                             Training Set Size (x1000)
                                                                                              4−>9   2−>8

Fig. 6. Training and test errors of LeNet-5 achieved using training
    sets of various sizes. This graph suggests that a larger training Fig. 8. The 82 test patterns misclassied by LeNet-5. Below each
    set could improve the performance of LeNet-5. The hollow square       image is displayed the correct answers (left) and the network an-
    show the test error when more training patterns are articially       swer (right). These errors are mostly caused either by genuinely
    generated using random distortions. The test patterns are not         ambiguous patterns, or by digits written in a style that are under-
    distorted.                                                            represented in the training set.

distorted patterns with randomly picked distortion param-                               perfectly identiable by humans, although they are writ-
eters. The distortions were combinations of the follow-                                 ten in an under-represented style. This shows that further
ing planar ane transformations: horizontal and verti-                                  improvements are to be expected with more training data.
cal translations, scaling, squeezing (simultaneous horizon-                             C. Comparison with Other Classiers
tal compression and vertical elongation, or the reverse),
and horizontal shearing. Figure 7 shows examples of dis-                                   For the sake of comparison, a variety of other trainable
torted patterns used for training. When distorted data was                              classiers was trained and tested on the same database. An
used for training, the test error rate dropped to 0.8% (from                            early subset of these results was presented in 51]. The error
0.95% without deformation). The same training parame-                                   rates on the test set for the various methods are shown in
ters were used as without deformations. The total length of                             gure 9.
the training session was left unchanged (20 passes of 60,000
patterns each). It is interesting to note that the network                              C.1 Linear Classier, and Pairwise Linear Classier
e ectively sees each individual sample only twice over the                                Possibly the simplest classier that one might consider is
course of these 20 passes.                                                              a linear classier. Each input pixel value contributes to a
   Figure 8 shows all 82 misclassied test examples. some                               weighted sum for each output unit. The output unit with
of those examples are genuinely ambiguous, but several are                              the highest sum (including the contribution of a bias con-

<!-- page 12 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                                             12

                          Linear        −−−− 12.0 −−−−>
                 [deslant] Linear       −−−− 8.4 −−−−>
                        Pairwise        −−−− 7.6 −−−−>

               K−NN Euclidean                                                                                                                                            5

       [deslant] K−NN Euclidean                                                                           2.4

            40 PCA + quadratic                                                                                             3.3

              1000 RBF + linear                                                                                                        3.6

       [16x16] Tangent Distance                                           1.1

                    SVM poly 4                                            1.1

                RS−SVM poly 5                                         1

            [dist] V−SVM poly 9                             0.8

                 28x28−300−10                                                                                                                                  4.7
            [dist] 28x28−300−10                                                                                                        3.6
        [deslant] 20x20−300−10                                                        1.6
                28x28−1000−10                                                                                                                            4.5
           [dist] 28x28−1000−10                                                                                                              3.8
            28x28−300−100−10                                                                                        3.05
       [dist] 28x28−300−100−10                                                                        2.5
            28x28−500−150−10                                                                                        2.95
       [dist] 28x28−500−150−10                                                                        2.45

                [16x16] LeNet−1                                                         1.7

                        LeNet−4                                           1.1

                LeNet−4 / Local                                           1.1

               LeNet−4 / K−NN                                             1.1

                        LeNet−5                                       0.95

                  [dist] LeNet−5                            0.8

          [dist] Boosted LeNet−4                          0.7

                                    0             0.5             1             1.5           2     2.5         3                3.5               4   4.5           5

Fig. 9. Error rate on the test set (%) for various classication methods. deslant] indicates that the classier was trained and tested on
    the deslanted version of the database. dist] indicates that the training set was augmented with articially distorted examples. 16x16]
    indicates that the system used the 16x16 pixel images. The uncertainty in the quoted error rates is about 0.1%.

stant) indicates the class of the input character. On the                                         However, the memory requirement and recognition time are
regular data, the error rate is 12%. The network has 7850                                         large: the complete 60,000 twenty by twenty pixel training
free parameters. On the deslanted images, the test error                                          images (about 24 Megabytes at one byte per pixel) must be
rate is 8.4% The network has 4010 free parameters. The                                            available at run time. Much more compact representations
deciencies of the linear classier are well documented 1]                                        could be devised with modest increase in error rate. On the
and it is included here simply to form a basis of comparison                                      regular test set the error rate was 5.0%. On the deslanted
for more sophisticated classiers. Various combinations of                                        data, the error rate was 2.4%, with k = 3. Naturally, a
sigmoid units, linear units, gradient descent learning, and                                       realistic Euclidean distance nearest-neighbor system would
learning by directly solving linear systems gave similar re-                                      operate on feature vectors rather than directly on the pix-
sults.                                                                                            els, but since all of the other systems presented in this
   A simple improvement of the basic linear classier was                                         study operate directly on the pixels, this result is useful for
tested 52]. The idea is to train each unit of a single-layer                                      a baseline comparison.
network to separate each class from each other class. In our
case this layer comprises 45 units labeled 0/1, 0/2,...0/9,                                       C.3 Principal Component Analysis (PCA) and Polynomial
1/2....8/9. Unit i=j is trained to produce +1 on patterns                                              Classier
of class i, -1 on patterns of class j , and is not trained on                                        Following 53], 54], a preprocessing stage was con-
other patterns. The nal score for class i is the sum of                                          structed which computes the projection of the input pat-
the outputs all the units labeled i=x minus the sum of the                                        tern on the 40 principal components of the set of training
output of all the units labeled y=i, for all x and y. The                                         vectors. To compute the principal components, the mean of
error rate on the regular test set was 7.6%.                                                      each input component was rst computed and subtracted
                                                                                                  from the training vectors. The covariance matrix of the re-
C.2 Baseline Nearest Neighbor Classier                                                           sulting vectors was then computed and diagonalized using
   Another simple classier is a K-nearest neighbor classi-                                       Singular Value Decomposition. The 40-dimensional feature
er with a Euclidean distance measure between input im-                                           vector was used as the input of a second degree polynomial
ages. This classier has the advantage that no training                                           classier. This classier can be seen as a linear classier
time, and no brain on the part of the designer, are required.                                     with 821 inputs, preceded by a module that computes all

<!-- page 13 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                      13

products of pairs of input variables. The error on the reg- only marginally improved error rates: 2.95%. Training
ular test set was 3.3%.                                     with distorted patterns improved the performance some-
                                                            what: 2.50% error for the 28x28-300-100-10 network, and
C.4 Radial Basis Function Network                           2.45% for the 28x28-1000-150-10 network.
   Following 55], an RBF network was constructed. The
rst layer was composed of 1,000 Gaussian RBF units with C.7 A Small Convolutional Network: LeNet-1
28x28 inputs, and the second layer was a simple 1000 inputs    Convolutional Networks are an attempt to solve the
/ 10 outputs linear classier. The RBF units were divided dilemma between small networks that cannot learn
into 10 groups of 100. Each group of units was trained the training set, and large networks that seem over-
on all the training examples of one of the 10 classes using parameterized. LeNet-1 was an early embodiment of the
the adaptive K-means algorithm. The second layer weights Convolutional Network architecture which is included here
were computed using a regularized pseudo-inverse method. for comparison purposes. The images were down-sampled
The error rate on the regular test set was 3.6%             to 16x16 pixels and centered in the 28x28 input layer. Al-
                                                            though about 100,000 multiply/add steps are required to
C.5 One-Hidden Layer Fully Connected Multilayer Neural evaluate LeNet-1, its convolutional nature keeps the num-
      Network                                               ber of free parameters to only about 2600. The LeNet-
   Another classier that we tested was a fully connected 1 architecture was developed using our own version of
multi-layer neural network with two layers of weights (one the USPS (US Postal Service zip codes) database and its
hidden layer) trained with the version of back-propagation size was tuned to match the available data 35]. LeNet-1
described in Appendix C. Error on the regular test set was achieved 1.7% test error. The fact that a network with such
4.7% for a network with 300 hidden units, and 4.5% for a a small number of parameters can attain such a good error
network with 1000 hidden units. Using articial distortions rate is an indication that the architecture is appropriate
to generate more training data brought only marginal im- for the task.
provement: 3.6% for 300 hidden units, and 3.8% for 1000
hidden units. When deslanted images were used, the test C.8 LeNet-4
error jumped down to 1.6% for a network with 300 hidden        Experiments with LeNet-1 made it clear that a larger
units.                                                      convolutional network was needed to make optimal use of
   It remains somewhat of a mystery that networks with the large size of the training set. LeNet-4 and later LeNet-
such a large number of free parameters manage to achieve 5 were designed to address this problem. LeNet-4 is very
reasonably low testing errors. We conjecture that the dy- similar to LeNet-5, except for the details of the architec-
namics of gradient descent learning in multilayer nets has ture. It contains 4 rst-level feature maps, followed by
a \self-regularization" e ect. Because the origin of weight 8 subsampling maps connected in pairs to each rst-layer
space is a saddle point that is attractive in almost every feature maps, then 16 feature maps, followed by 16 sub-
direction, the weights invariably shrink during the rst sampling map, followed by a fully connected layer with
few epochs (recent theoretical analysis seem to conrm 120 units, followed by the output layer (10 units). LeNet-4
this 56]). Small weights cause the sigmoids to operate contains about 260,000 connections and has about 17,000
in the quasi-linear region, making the network essentially free parameters. Test error was 1.1%. In a series of ex-
equivalent to a low-capacity, single-layer network. As the periments, we replaced the last layer of LeNet-4 with a
learning proceeds, the weights grow, which progressively Euclidean Nearest Neighbor classier, and with the \local
increases the e ective capacity of the network. This seems learning" method of Bottou and Vapnik 58], in which a lo-
to be an almost perfect, if fortuitous, implementation of cal linear classier is retrained each time a new test pattern
Vapnik's \Structural Risk Minimization" principle 6]. A is shown. Neither of those methods improved the raw error
better theoretical understanding of these phenomena, and rate, although they did improve the rejection performance.
more empirical evidence, are denitely needed.
                                                            C.9 Boosted LeNet-4
C.6 Two-Hidden Layer Fully Connected Multilayer Neural         Following theoretical work by R. Schapire 59], Drucker
      Network                                               et al. 60] developed the \boosting" method for combining
   To see the e ect of the architecture, several two-hidden multiple classiers. Three LeNet-4s are combined: the rst
layer multilayer neural networks were trained. Theoreti- one is trained the usual way. the second one is trained on
cal results have shown that any function can be approxi- patterns that are ltered by the rst net so that the second
mated by a one-hidden layer neural network 57]. However, machine sees a mix of patterns, 50% of which the rst net
several authors have observed that two-hidden layer archi- got right, and 50% of which it got wrong. Finally, the
tectures sometimes yield better performance in practical third net is trained on new patterns on which the rst and
situations. This phenomenon was also observed here. The the second nets disagree. During testing, the outputs of
test error rate of a 28x28-300-100-10 network was 3.05%, the three nets are simply added. Because the error rate of
a much better result than the one-hidden layer network, LeNet-4 is very low, it was necessary to use the articially
obtained using marginally more weights and connections. distorted images (as with LeNet-5) in order to get enough
Increasing the network size to 28x28-1000-150-10 yielded samples to train the second and third nets. The test error

<!-- page 14 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                                                                      14

rate was 0.7%, the best of any of our classiers. At rst         [deslant] K−NN Euclidean                                                                                            8.1

glance, boosting appears to be three times more expensive          [16x16] Tangent Distance
                                                                                SVM poly 4
                                                                                                                           1.9
                                                                                                                           1.8

as a single net. In fact, when the rst net produces a
high condence answer, the other nets are not called. The           [deslant] 20x20−300−10                                            3.2

average computational cost is about 1.75 times that of a                   [16x16] LeNet−1                                                      3.7

single net.                                                                        LeNet−4
                                                                           LeNet−4 / Local                       1.4
                                                                                                                           1.8

                                                                           LeNet−4 / K−NN                                  1.6

C.10 Tangent Distance Classier (TDC)                                [dist] Boosted LeNet−4            0.5

   The Tangent Distance classier (TDC) is a nearest-
                                                                                              0            1           2         3          4           5         6     7         8         9

neighbor method where the distance function is made in-
sensitive to small distortions and translations of the input Fig. 10. Rejection Performance: percentage of test patterns that
image 61]. If we consider an image as a point in a high           must be rejected to achieve 0.5% error for some of the systems.
dimensional pixel space (where the dimensionality equals
the number of pixels), then an evolving distortion of a char-                        Linear       4

acter traces out a curve in pixel space. Taken together,
                                                                                   Pairwise

all these distortions dene a low-dimensional manifold in         [deslant] K−NN Euclidean    −−−− 24,000 −−−−>

pixel space. For small distortions, in the vicinity of the
                                                                        40 PCA+quadratic              39
                                                                                 1000 RBF                                                                                                       794

original image, this manifold can be approximated by a            [16x16] Tangent Distance    −−−− 20,000 −−−−>

plane, known as the tangent plane. An excellent measure
                                                                               SVM poly 4     −−−− 14,000 −−−−>
                                                                           RS−SVM poly 5                                                                                    650

of "closeness" for character images is the distance between            [dist] V−SVM poly 9    −−−− 28,000 −−−−>

their tangent planes, where the set of distortions used to         [deslant] 20x20−300−10                       123

generate the planes includes translations, scaling, skewing,               28x28−1000−10                                                                                                        795

squeezing, rotation, and line thickness variations. A test             28x28−300−100−10                                          267
                                                                                                                                                            469

error rate of 1.1% was achieved using 16x16 pixel images.
                                                                       28x28−500−150−10

Preltering techniques using simple Euclidean distance at                  [16x16] LeNet−1                     100

multiple resolutions allowed to reduce the number of nec-
                                                                                  LeNet−4                                        260
                                                                           LeNet−4 / Local    −−−− 20,000 −−−−>

essary Tangent Distance calculations.                                     LeNet−4 / K−NN
                                                                                  LeNet−5
                                                                                              −−−− 10,000 −−−−>
                                                                                                                                                      401

C.11 Support Vector Machine (SVM)
                                                                          Boosted LeNet−4                                                                   460

                                                                                              0                                      300                              600                             900

   Polynomial classiers are well-studied methods for gen-       Fig. 11. Number of multiply-accumulate operations for the recogni-
erating complex decision surfaces. Unfortunately, they                tion of a single character starting with a size-normalized image.
are impractical for high-dimensional problems, because the
number of product terms is prohibitive. The Support Vec-
tor technique is an extremely economical way of represent-       has reached 0.8% using a modied version of the V-SVM.
ing complex surfaces in high-dimensional spaces, including       Unfortunately, V-SVM is extremely expensive: about twice
polynomials and many other types of surfaces 6].                 as much as regular SVM. To alleviate this problem, Burges
   A particularly interesting subset of decision surfaces is     has proposed the Reduced Set Support Vector technique
the ones that correspond to hyperplanes that are at a max-       (RS-SVM), which attained 1.1% on the regular test set 63],
imum distance from the convex hulls of the two classes in        with a computational cost of only 650,000 multiply-adds
the high-dimensional space of the product terms. Boser,          per recognition, i.e. only about 60% more expensive than
Guyon, and Vapnik 62] realized that any polynomial of            LeNet-5.
degree k in this \maximum margin" set can be computed
by rst computing the dot product of the input image with        D. Discussion
a subset of the training samples (called the \support vec-          A summary of the performance of the classiers is shown
tors"), elevating the result to the k-th power, and linearly     in Figures 9 to 12. Figure 9 shows the raw error rate of the
combining the numbers thereby obtained. Finding the sup-         classiers on the 10,000 example test set. Boosted LeNet-4
port vectors and the coecients amounts to solving a high-       performed best, achieving a score of 0.7%, closely followed
dimensional quadratic minimization problem with linear           by LeNet-5 at 0.8%.
inequality constraints. For the sake of comparison, we in-          Figure 10 shows the number of patterns in the test set
clude here the results obtained by Burges and Sch"olkopf         that must be rejected to attain a 0.5% error for some of
reported in 63]. With a regular SVM, their error rate            the methods. Patterns are rejected when the value of cor-
on the regular test set was 1.4%. Cortes and Vapnik had          responding output is smaller than a predened threshold.
reported an error rate of 1.1% with SVM on the same              In many applications, rejection performance is more signif-
data using a slightly di erent technique. The computa-           icant than raw error rate. The score used to decide upon
tional cost of this technique is very high: about 14 million     the rejection of a pattern was the di erence between the
multiply-adds per recognition. Using Sch"olkopf's Virtual        scores of the top two classes. Again, Boosted LeNet-4 has
Support Vectors technique (V-SVM), 1.0% error was at-            the best performance. The enhanced versions of LeNet-4
tained. More recently, Sch"olkopf (personal communication)       did better than the original LeNet-4, even though the raw

<!-- page 15 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                   15

                   Linear       4
                                                                                        ing the template images. Not surprisingly, neural networks
                Pairwise             35
                                                                                        require much less memory than memory-based methods.
[deslant] K−NN Euclidean    −−− 24,000 −−−>                                                The Overall performance depends on many factors in-
       40 PCA+quadratic              40
                                                                            794         cluding accuracy, running time, and memory requirements.
                                                                                        As computer technology improves, larger-capacity recog-
               1000 RBF
[16x16] Tangent Distance    −−− 25,000 −−−>
             SVM poly 4     −−−− 14,000 −−−−>
                                                                                        nizers become feasible. Larger recognizers in turn require
                                                                                        larger training sets. LeNet-1 was appropriate to the avail-
         RS−SVM poly 5                                                650
     [dist] V−SVM poly 5    −−−− 28,000 −−−−>

                                                                                        able technology in 1989, just as LeNet-5 is appropriate now.
                                                                                        In 1989 a recognizer as complex as LeNet-5 would have re-
 [deslant] 20x20−300−10                       123
         28x28−1000−10                                                      795
     28x28−300−100−10                               267
                                                                                        quired several weeks' training, and more data than was
                                                                                        available, and was therefore not even considered. For quite
     28x28−500−150−10                                     469

         [16x16] LeNet 1        3                                                       a long time, LeNet-1 was considered the state of the art.
                 LeNet 4            17
                            −−− 24,000 −−−>                                             The local learning classier, the optimal margin classier,
                                                                                        and the tangent distance classier were developed to im-
         LeNet 4 / Local
         LeNet 4 / K−NN     −−− 24,000 −−−>
                 LeNet 5                 60
                                                                                        prove upon LeNet-1 { and they succeeded at that. How-
                                                                                        ever, they in turn motivated a search for improved neural
        Boosted LeNet 4                  51

                                                                                        network architectures. This search was guided in part by
                            0                       300         600               900

Fig. 12. Memory requirements, measured in number of variables, for                      estimates of the capacity of various learning machines, de-
     each of the methods. Most of the methods only require one byte                     rived from measurements of the training and test error as
     per variable for adequate performance.                                             a function of the number of training examples. We dis-
                                                                                        covered that more capacity was needed. Through a series
accuracies were identical.                                                              of experiments in architecture, combined with an analy-
   Figure 11 shows the number of multiply-accumulate op-                                sis of the characteristics of recognition errors, LeNet-4 and
erations necessary for the recognition of a single size-                                LeNet-5 were crafted.
normalized image for each method. Expectedly, neural                                       We nd that boosting gives a substantial improvement in
networks are much less demanding than memory-based                                      accuracy, with a relatively modest penalty in memory and
methods. Convolutional Neural Networks are particu-                                     computing expense. Also, distortion models can be used
larly well suited to hardware implementations because of                                to increase the e ective size of a data set without actually
their regular structure and their low memory requirements                               requiring to collect more data.
for the weights. Single chip mixed analog-digital imple-                                   The Support Vector Machine has excellent accuracy,
mentations of LeNet-5's predecessors have been shown to                                 which is most remarkable, because unlike the other high
operate at speeds in excess of 1000 characters per sec-                                 performance classiers, it does not include a priori knowl-
ond 64]. However, the rapid progress of mainstream com-                                 edge about the problem. In fact, this classier would do
puter technology renders those exotic technologies quickly                              just as well if the image pixels were permuted with a xed
obsolete. Cost-e ective implementations of memory-based                                 mapping and lost their pictorial structure. However, reach-
techniques are more elusive, due to their enormous memory                               ing levels of performance comparable to the Convolutional
requirements, and computational requirements.                                           Neural Networks can only be done at considerable expense
   Training time was also measured. K-nearest neighbors                                 in memory and computational requirements. The reduced-
and TDC have essentially zero training time. While the                                  set SVM requirements are within a factor of two of the
single-layer net, the pairwise net, and PCA+quadratic net                               Convolutional Networks, and the error rate is very close.
could be trained in less than an hour, the multilayer net                               Improvements of those results are expected, as the tech-
training times were expectedly much longer, but only re-                                nique is relatively new.
quired 10 to 20 passes through the training set. This                                      When plenty of data is available, many methods can at-
amounts to 2 to 3 days of CPU to train LeNet-5 on a Sil-                                tain respectable accuracy. The neural-net methods run
icon Graphics Origin 2000 server, using a single 200MHz                                 much faster and require much less space than memory-
R10000 processor. It is important to note that while the                                based techniques. The neural nets' advantage will become
training time is somewhat relevant to the designer, it is of                            more striking as training databases continue to increase in
little interest to the nal user of the system. Given the                               size.
choice between an existing technique, and a new technique
that brings marginal accuracy improvements at the price                                 E. Invariance and Noise Resistance
of considerable training time, any nal user would chose                                   Convolutional networks are particularly well suited for
the latter.                                                                             recognizing or rejecting shapes with widely varying size,
   Figure 12 shows the memory requirements, and therefore                               position, and orientation, such as the ones typically pro-
the number of free parameters, of the various classiers                                duced by heuristic segmenters in real-world string recogni-
measured in terms of the number of variables that need                                  tion systems.
to be stored. Most methods require only about one byte                                     In an experiment like the one described above, the im-
per variable for adequate performance. However, Nearest-                                portance of noise resistance and distortion invariance is
Neighbor methods may get by with 4 bits per pixel for stor-                             not obvious. The situation in most real applications is

<!-- page 16 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                   16

quite di erent. Characters must generally be segmented                                          W1       W2

out of their context prior to recognition. Segmentation
algorithms are rarely perfect and often leave extraneous
marks in character images (noise, underlines, neighboring                                F1(X0,X1,W1)

characters), or sometimes cut characters too much and pro-               t
                                                                        Input
                                                                                                              X3
                                                                                                                   F3(X3,X4)

duce incomplete characters. Those images cannot be re-
                                                                                                                               X5
                                                                          Z

liably size-normalized and centered. Normalizing incom-
                                                                                                                                    Loss
                                                                                         X1
                                                                                                                                    Function   E

plete characters can be very dangerous. For example, an
                                                                                                                     X4
                                                                                F0(X0)

enlarged stray mark can look like a genuine 1. Therefore
                                                                                                     F2(X2,W2)
                                                                                           X2

many systems have resorted to normalizing the images at                                                                     D

the level of elds or words. In our case, the upper and lower                                                         Desired Output

proles of entire elds (amounts in a check) are detected         Fig. 14. A trainable system composed of heterogeneous modules.
and used to normalize the image to a xed height. While
this guarantees that stray marks will not be blown up into
character-looking images, this also creates wide variations      words, that can be trained to simultaneously segment and
of the size and vertical position of characters after segmen-    recognize words, without ever being given the correct seg-
tation. Therefore it is preferable to use a recognizer that is   mentation.
robust to such variations. Figure 13 shows several exam-           Figure 14 shows an example of a trainable multi-modular
ples of distorted characters that are correctly recognized by    system. A multi-module system is dened by the function
LeNet-5. It is estimated that accurate recognition occurs        implemented by each of the modules, and by the graph of
for scale variations up to about a factor of 2, vertical shift   interconnection of the modules to each other. The graph
variations of plus or minus about half the height of the         implicitly denes a partial order according to which the
character, and rotations up to plus or minus 30 degrees.         modules must be updated in the forward pass. For exam-
While fully invariant recognition of complex shapes is still     ple in Figure 14, module 0 is rst updated, then modules 1
an elusive goal, it seems that Convolutional Networks o er       and 2 are updated (possibly in parallel), and nally mod-
a partial answer to the problem of invariance or robustness      ule 3. Modules may or may not have trainable parameters.
with respect to geometrical distortions.                         Loss functions, which measure the performance of the sys-
   Figure 13 includes examples of the robustness of LeNet-       tem, are implemented as module 4. In the simplest case,
5 under extremely noisy conditions. Processing those             the loss function module receives an external input that
images would pose unsurmountable problems of segmen-             carries the desired output. In this framework, there is no
tation and feature extraction to many methods, but               qualitative di erence between trainable parameters (W1,W2
LeNet-5 seems able to robustly extract salient features          in the gure), external inputs and outputs (Z,D,E), and
from these cluttered images. The training set used for           intermediate state variables(X1,X2,X3,X4,X5).
the network shown here was the MNIST training set
with salt and pepper noise added. Each pixel was ran-            A. An Object-Oriented Approach
domly inverted with probability 0.1. More examples                  Object-Oriented programming o ers a particularly con-
of LeNet-5 in action are available on the Internet at            venient way of implementing multi-module systems. Each
http://www.research.att.com/~yann/ocr.                           module is an instance of a class. Module classes have a \for-
                                                                 ward propagation" method (or member function) called
       IV. Multi-Module Systems and Graph                        fprop whose arguments are the inputs and outputs of the
               Transformer Networks                              module. For example, computing the output of module 3
   The classical back-propagation algorithm, as described        in Figure 14 can be done by calling the method fprop on
and used in the previous sections, is a simple form of           module 3 with the arguments X3,X4,X5. Complex mod-
Gradient-Based Learning. However, it is clear that the           ules can be constructed from simpler modules by simply
gradient back-propagation algorithm given by Equation 4          dening a new class whose slots will contain the member
describes a more general situation than simple multi-layer       modules and the intermediate state variables between those
feed-forward networks composed of alternated linear trans-       modules. The fprop method for the class simply calls the
formations and sigmoidal functions. In principle, deriva-        fprop methods of the member modules, with the appro-
tives can be back-propagated through any arrangement of          priate intermediate state variables or external input and
functional modules, as long as we can compute the prod-          outputs as arguments. Although the algorithms are eas-
uct of the Jacobians of those modules by any vector. Why         ily generalizable to any network of such modules, including
would we want to train systems composed of multiple het-         those whose inuence graph has cycles, we will limit the dis-
erogeneous modules? The answer is that large and complex         cussion to the case of directed acyclic graphs (feed-forward
trainable systems need to be built out of simple, specialized    networks).
modules. The simplest example is LeNet-5, which mixes               Computing derivatives in a multi-module system is just
convolutional layers, sub-sampling layers, fully-connected       as simple. A \backward propagation" method, called
layers, and RBF layers. Another less trivial example, de-        bprop, for each module class can be dened for that pur-
scribed in the next two sections, is a system for recognizing    pose. The bprop method of a module takes the same ar-

<!-- page 17 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                         17

                                                                                       C1   S2 C3 S4     C5

                                         4                                  4                                 4    Output

                                                                                                                       F6

                                         4                                  3                                 8

                                         4                                  3                                 3

Fig. 13. Examples of unusual, distorted, and noisy characters correctly recognized by LeNet-5. The grey-level of the output label represents
     the penalty (lighter for higher penalties).

guments as the fprop method. All the derivatives in the                used to extend the procedures to networks with recurrent
system can be computed by calling the bprop method on all              connections.
the modules in reverse order compared to the forward prop-
agation phase. The state variables are assumed to contain              B. Special Modules
slots for storing the gradients computed during the back-                 Neural networks and many other standard pattern recog-
ward pass, in addition to storage for the states computed in           nition techniques can be formulated in terms of multi-
the forward pass. The backward pass e ectively computes                modular systems trained with Gradient-Based Learning.
the partial derivatives of the loss E with respect to all the          Commonly used modules include matrix multiplications
state variables and all the parameters in the system. There            and sigmoidal modules, the combination of which can be
is an interesting duality property between the forward and             used to build conventional neural networks. Other mod-
backward functions of certain modules. For example, a                  ules include convolutional layers, sub-sampling layers, RBF
sum of several variables in the forward direction is trans-            layers, and \softmax" layers 65]. Loss functions are also
formed into a simple fan-out (replication) in the backward             represented as modules whose single output produces the
direction. Conversely, a fan-out in the forward direction              value of the loss. Commonly used modules have simple
is transformed into a sum in the backward direction. The               bprop methods. In general, the bprop method of a func-
software environment used to obtain the results described              tion F is a multiplication by the Jacobian of F . Here are
in this paper, called SN3.1, uses the above concepts. It is            a few commonly used examples. The bprop method of a
based on a home-grown object-oriented dialect of Lisp with             fanout (a \Y" connection) is a sum, and vice versa. The
a compiler to C.                                                       bprop method of a multiplication by a coecient is a mul-
   The fact that derivatives can be computed by propaga-               tiplication by the same coecient. The bprop method of a
tion in the reverse graph is easy to understand intuitively.           multiplication by a matrix is a multiplication by the trans-
The best way to justify it theoretically is through the use of         pose of that matrix. The bprop method of an addition with
Lagrange functions 21], 22]. The same formalism can be                 a constant is the identity.

<!-- page 18 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                  18

                                                                      tween the modules, are all xed-size vectors. The limited
                                                                      exibility of xed-size vectors for data representation is a
                                                                      serious deciency for many applications, notably for tasks
                Layer                 Graph
                                      Transformer
                                                                      that deal with variable length inputs (e.g continuous speech
                                                                      recognition and handwritten word recognition), or for tasks
                                                                      that require encoding relationships between objects or fea-
                                                                      tures whose number and nature can vary (invariant per-
                                                                      ception, scene analysis, recognition of composite objects).
               Layer                  Graph
                                      Transformer                     An important special case is the recognition of strings of
                                                                      characters or words.
                            (a)
                                                      (b)
                                                                         More generally, xed-size vectors lack exibility for tasks
                                                                      in which the state must encode probability distributions
                                                                      over sequences of vectors or symbols as is the case in lin-
Fig. 15. Traditional neural networks, and multi-module systems com-   guistic processing. Such distributions over sequences are
     municate xed-size vectors between layer. Multi-Layer Graph
     Transformer Networks are composed of trainable modules that      best represented by stochastic grammars, or, in the more
     operate on and produce graphs whose arcs carry numerical in-     general case, directed graphs in which each arc contains a
     formation.                                                       vector (stochastic grammars are special cases in which the
                                                                      vector contains probabilities and symbolic information).
   Interestingly, certain non-di erentiable modules can be            Each path in the graph represents a di erent sequence of
inserted in a multi-module system without adverse e ect.              vectors. Distributions over sequences can be represented
An interesting example of that is the multiplexer module.             by interpreting elements of the data associated with each
It has two (or more) regular inputs, one switching input,             arc as parameters of a probability distribution or simply
and one output. The module selects one of its inputs, de-             as a penalty. Distributions over sequences are particularly
pending upon the (discrete) value of the switching input,             handy for modeling linguistic knowledge in speech or hand-
and copies it on its output. While this module is not dif-            writing recognition systems: each sequence, i.e., each path
ferentiable with respect to the switching input, it is di er-         in the graph, represents an alternative interpretation of the
entiable with respect to the regular inputs. Therefore the            input. Successive processing modules progressively rene
overall function of a system that includes such modules will          the interpretation. For example, a speech recognition sys-
be di erentiable with respect to its parameters as long as            tem might start with a single sequence of acoustic vectors,
the switching input does not depend upon the parameters.              transform it into a lattice of phonemes (distribution over
For example, the switching input can be an external input.            phoneme sequences), then into a lattice of words (distribu-
                                                                      tion over word sequences), and then into a single sequence
   Another interesting case is the min module. This mod-              of words representing the best interpretation.
ule has two (or more) inputs and one output. The output                  In our work on building large-scale handwriting recog-
of the module is the minimum of the inputs. The func-                 nition systems, we have found that these systems could
tion of this module is di erentiable everywhere, except on            much more easily and quickly be developed and designed
the switching surface which is a set of measure zero. In-             by viewing the system as a networks of modules that take
terestingly, this function is continuous and reasonably reg-          one or several graphs as input and produce graphs as out-
ular, and that is sucient to ensure the convergence of a             put. Such modules are called Graph Transformers, and the
Gradient-Based Learning algorithm.                                    complete systems are called Graph Transformer Networks,
   The object-oriented implementation of the multi-module             or GTN. Modules in a GTN communicate their states and
idea can easily be extended to include a bbprop method                gradients in the form of directed graphs whose arcs carry
that propagates Gauss-Newton approximations of the sec-               numerical information (scalars or vectors) 66].
ond derivatives. This leads to a direct generalization for               From the statistical point of view, the xed-size state
modular systems of the second-derivative back-propagation             vectors of conventional networks can be seen as represent-
Equation 22 given in the Appendix.                                    ing the means of distributions in state space. In variable-
   The multiplexer module is a special case of a much                 size networks such as the Space-Displacement Neural Net-
more general situation, described at length in Section VIII,          works described in section VII, the states are variable-
where the architecture of the system changes dynamically              length sequences of xed size vectors. They can be seen
with the input data. Multiplexer modules can be used to               as representing the mean of a probability distribution over
dynamically rewire (or recongure) the architecture of the            variable-length sequences of xed-size vectors. In GTNs,
system for each new input pattern.                                    the states are represented as graphs, which can be seen
C. Graph Transformer Networks                                         as representing mixtures of probability distributions over
                                                                      structured collections (possibly sequences) of vectors (Fig-
  Multi-module systems are a very exible tool for build-             ure 15).
ing large trainable system. However, the descriptions in                 One of the main points of the next several sections is
the previous sections implicitly assumed that the set of              to show that Gradient-Based Learning procedures are not
parameters, and the state information communicated be-                limited to networks of simple modules that communicate

<!-- page 19 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                           19

through xed-size vectors, but can be generalized to GTNs.
Gradient back-propagation through a Graph Transformer
takes gradients with respect to the numerical informa-
tion in the output graph, and computes gradients with re-
spect to the numerical information attached to the input
graphs, and with respect to the module's internal param-          Fig. 16. Building a segmentation graph with Heuristic Over-
eters. Gradient-Based Learning can be applied as long as               Segmentation.
di erentiable functions are used to produce the numerical
data in the output graph from the numerical data in the
input graph and the functions parameters.                         that it avoids making hard decisions about the segmenta-
   The second point of the next several sections is to show       tion by taking a large number of di erent segmentations
that the functions implemented by many of the modules             into consideration. The idea is to use heuristic image pro-
used in typical document processing systems (and other            cessing techniques to nd candidate cuts of the word or
image recognition systems), though commonly thought to            string, and then to use the recognizer to score the alter-
be combinatorial in nature, are indeed di erentiable with         native segmentations thereby generated. The process is
respect to their internal parameters as well as with respect      depicted in Figure 16. First, a number of candidate cuts
to their inputs, and are therefore usable as part of a globally   are generated. Good candidate locations for cuts can be
trainable system.                                                 found by locating minima in the vertical projection prole,
   In most of the following, we will purposely avoid making       or minima of the distance between the upper and lower
references to probability theory. All the quantities manip-       contours of the word. Better segmentation heuristics are
ulated are viewed as penalties, or costs, which if necessary      described in section X. The cut generation heuristic is de-
can be transformed into probabilities by taking exponen-          signed so as to generate more cuts than necessary, in the
tials and normalizing.                                            hope that the \correct" set of cuts will be included. Once
                                                                  the cuts have been generated, alternative segmentations are
   V. Multiple Object Recognition: Heuristic                      best represented by a graph, called the segmentation graph.
                Over-Segmentation                                 The segmentation graph is a Directed Acyclic Graph (DAG)
   One of the most dicult problems of handwriting recog-         with a start node and an end node. Each internal node is
nition is to recognize not just isolated characters, but          associated with a candidate cut produced by the segmen-
strings of characters, such as zip codes, check amounts,          tation algorithm. Each arc between a source node and a
or words. Since most recognizers can only deal with one           destination node is associated with an image that contains
character at a time, we must rst segment the string into         all the ink between the cut associated with the source node
individual character images. However, it is almost impos-         and the cut associated with the destination node. An arc
sible to devise image analysis techniques that will infallibly    is created between two nodes if the segmentor decided that
segment naturally written sequences of characters into well       the ink between the corresponding cuts could form a can-
formed characters.                                                didate character. Typically, each individual piece of ink
   The recent history of automatic speech recognition 28],        would be associated with an arc. Pairs of successive pieces
 67] is here to remind us that training a recognizer by opti-     of ink would also be included, unless they are separated by
mizing a global criterion (at the word or sentence level) is      a wide gap, which is a clear indication that they belong
much preferable to merely training it on hand-segmented           to di erent characters. Each complete path through the
phonemes or other units. Several recent works have shown          graph contains each piece of ink once and only once. Each
that the same is true for handwriting recognition 38]: op-        path corresponds to a di erent way of associating pieces of
timizing a word-level criterion is preferable to solely train-    ink together so as to form characters.
ing a recognizer on pre-segmented characters because the
recognizer can learn not only to recognize individual char-B. Recognition Transformer and Viterbi Transformer
acters, but also to reject mis-segmented characters thereby   A simple GTN to recognize character strings is shown
minimizing the overall word error.                         in Figure 17. It is composed of two graph transformers
   This section and the next describe in detail a simple ex-
                                                           called the recognition transformer Trec , and the Viterbi
ample of GTN to address the problem of reading strings of  transformer Tvit . The goal of the recognition transformer
characters, such as words or check amounts. The method     is to generate a graph, called the interpretation graph or
avoids the expensive and unreliable task of hand-truthing  recognition graph Gint , that contains all the possible inter-
the result of the segmentation often required in more tra- pretations for all the possible segmentations of the input.
ditional systems trained on individually labeled character Each path in Gint represents one possible interpretation of
images.                                                    one particular segmentation of the input. The role of the
                                                           Viterbi transformer is to extract the best interpretation
A. Segmentation Graph                                      from the interpretation graph.
  A now classical method for word segmentation and recog-     The recognition transformer Trec takes the segmentation
nition is called Heuristic Over-Segmentation 68], 69]. Its graph Gseg as input, and applies the recognizer for single
main advantages over other approaches to segmentation are characters to the images associated with each of the arcs

<!-- page 20 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                                                           20

                             Viterbi Penalty                                                                                           class
                                                                                                                                       label
                                                                                                                                                     character recognizer
                                                                                                                                                     penalty for each class

                                           Σ
                                                                                                     PIECE OF THE                               7.9
                                                                                                   INTERPRETATION   "0" 6.7               "0"
                                                                                                       GRAPH                              "1"       11.2
                                                                2         Viterbi                                   "1" 10.3
                         3                                                                                                                "2"       6.8
                                                                          Path
         Gvit                                                                                                                             "3"   0.2

                                               4                                                                     "8" 0.3             "8"    13.5
                                                                                                                    "9" 12.5             "9"    8.4

                 T vit             Viterbi
                                   Transformer

                                                                                                                    Character              Character
                                                                                                     W              Recognizer             Recognizer
                         3             3               1        2
                         2             4               4        3        Interpretation
          Gint                                                  4           Graph
                               3                   1
                                   4           4

                                                                                                                      8                         3                  candidate
                                                                                                                          0.1
                                                                          Recognition                                                                               segment
        T rec       NN        NN       NN          NN      NN       NN    Transformer                                PIECE OF THE
                                                                                                                    SEGMENTATION
                                                                                                                                                      0.5
                                                                                                                                                                     image
                                                                                                                        GRAPH                                   penalty given by
                                                                                                                                                                the segmentor

                                                                         Segmentation     Fig. 18. The recognition transformer renes each arc of the segmen-
           Gseg                                                             Graph              tation arc into a set of arcs in the interpretation graph, one per
                                                                                               character class, with attached penalties and labels.

Fig. 17. Recognizing a character string with a GTN. For readability,
     only the arcs with low penalties are shown.                     famous Viterbi algorithm 70], an application of the prin-
                                                                                          ciple of dynamic programming to nd the shortest path
                                                                                          in a graph eciently. Let ci be the penalty associated to
in the segmentation graph. The interpretation graph Gint                                  arc i, with source node si , and destination node di (note
has almost the same structure as the segmentation graph,                                  that there can be multiple arcs between two nodes). In
except that each arc is replaced by a set of arcs from and                                the interpretation graph, arcs also have a label li . The
to the same node. In this set of arcs, there is one arc for                               Viterbi algorithm proceeds as follows. Each node n is as-
each possible class for the image associated with the cor-                                sociated with a cumulated Viterbi penalty vn . Those cu-
responding arc in Gseg . As shown in Figure 18, to each                                   mulated penalties are computed in any order that satises
arc is attached a class label, and the penalty that the im-                               the partial order dened by the interpretation graph (which
age belongs to this class as produced by the recognizer. If                               is directed and acyclic). The start node is initialized with
the segmentor has computed penalties for the candidate                                    the cumulated penalty v        = 0. The other nodes cu-
segments, these penalties are combined with the penalties
                                                                                                                               start

                                                                                          mulated penalties vn are computed recursively from the v
computed by the character recognizer, to obtain the penal-                                values of their parent nodes, through the upstream arcs
ties on the arcs of the interpretation graph. Although com-                                Un = farc i with destination di = ng:
bining penalties of di erent nature seems highly heuristic,
the GTN training procedure will tune the penalties and
take advantage of this combination anyway. Each path in                                                             vn = imin (ci + vsi ):                                               (10)
the interpretation graph corresponds to a possible inter-                                                                 2U      n
pretation of the input word. The penalty of a particular
interpretation for a particular segmentation is given by the
sum of the arc penalties along the corresponding path in                                  Furthermore, the value of i for each node n which minimizes
the interpretation graph. Computing the penalty of an in-                                 the right hand side is noted mn , the minimizing entering
terpretation independently of the segmentation requires to                                arc. When the end node is reached we obtain in v the                                     end

combine the penalties of all the paths with that interpre-                                total penalty of the path with the smallest total penalty.
tation. An appropriate rule for combining the penalties of                                We call this penalty the Viterbi penalty, and this sequence
parallel paths is given in section VI-C.                                                  of arcs and nodes the Viterbi path. To obtain the Viterbi
   The Viterbi transformer produces a graph Gvit with a                                   path with nodes n : : : nT and arcs i : : : iT ; , we trace back
                                                                                                               1                                    1              1

single path. This path is the path of least cumulated                                     these nodes and arcs as follows, starting with nT = the end
penalty in the Interpretation graph. The result of the                                    node, and recursively using the minimizing entering arc:
recognition can be produced by reading o the labels of                                    it = mnt+1 , and nt = sit until the start node is reached.
the arcs along the graph Gvit extracted by the Viterbi                                    The label sequence can then be read o the arcs of the
transformer. The Viterbi transformer owes its name to the                                 Viterbi path.

<!-- page 21 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                         21

 VI. Global Training for Graph Transformer                                                 Constrained Viterbi Penalty      Ccvit
                   Networks
                                                                                                           Σ
   The previous section describes the process of recognizing
a string using Heuristic Over-Segmentation, assuming that                  Best Constrained Path                Gcvit
the recognizer is trained so as to give low penalties for the
correct class label of correctly segmented characters, high                                       Viterbi Transformer

penalties for erroneous categories of correctly segmented                  Constrained
characters, and high penalties for all categories for badly                Interpretation Graph                 Gc
formed characters. This section explains how to train the
system at the string level to do the above without requiring            Desired Sequence              Path Selector

manual labeling of character segments. This training will                  Interpretation Graph                    Gint
be performed with a GTN whose architecture is slightly
di erent from the recognition architecture described in the                                          Recognition
previous section.
                                                                                                     Transformer

   In many applications, there is enough a priori knowl-        Fig. 19. Viterbi Training GTN Architecture for a character string
edge about what is expected from each of the modules in              recognizer based on Heuristic Over-Segmentation.
order to train them separately. For example, with Heuris-
tic Over-Segmentation one could individually label single-
character images and train a character recognizer on them,      Neural Networks (RNN). Unfortunately, despite early en-
but it might be dicult to obtain an appropriate set of         thusiasm, the training of RNNs with gradient-based tech-
non-character images to train the model to reject wrongly       niques has proved very dicult in practice 79].
segmented candidates. Although separate training is sim-          The GTN techniques presented below simplify and gen-
ple, it requires additional supervision information that is     eralize the global training methods developed for speech
often lacking or incomplete (the correct segmentation and       recognition.
the labels of incorrect candidate segments). Furthermore
it can be shown that separate training is sub-optimal 67].
   The following section describes three di erent gradient-     A. Viterbi Training
based methods for training GTN-based handwriting recog-
nizers at the string level: Viterbi training, discriminative       During recognition, we select the path in the Interpre-
Viterbi training, forward training, and discriminative for-     tation Graph that has the lowest penalty with the Viterbi
ward training. The last one is a generalization to graph-       algorithm. Ideally, we would like this path of lowest penalty
based systems of the MAP criterion introduced in Sec-           to be associated with the correct label sequence as often as
tion II-C. Discriminative forward training is somewhat          possible. An obvious loss function to minimize is therefore
similar to the so-called Maximum Mutual Information cri-        the average over the training set of the penalty of the path
terion used to train HMM in speech recognition. However,        associated with the correct label sequence that has the low-
our rationale di ers from the classical one. We make no         est penalty. The goal of training will be to nd the set of
recourse to a probabilistic interpretation, but show that,      recognizer parameters (the weights, if the recognizer is a
within the Gradient-Based Learning approach, discrimina-        neural network) that minimize the average penalty of this
tive training is a simple instance of the pervasive principle   \correct" lowest penalty path. The gradient of this loss
of error correcting learning.                                   function can be computed by back-propagation through
   Training methods for graph-based sequence recognition        the GTN architecture shown in gure 19. This training
systems such as HMMs have been extensively studied in           architecture is almost identical to the recognition archi-
the context of speech recognition 28]. Those methods re-        tecture described in the previous section, except that an
quire that the system be based on probabilistic generative      extra graph transformer called a path selector is inserted
models of the data, which provide normalized likelihoods        between the Interpretation Graph and the Viterbi Trans-
over the space of possible input sequences. Popular HMM         former. This transformer takes the interpretation graph
learning methods, such as the the Baum-Welsh algorithm,         and the desired label sequence as input. It extracts from
rely on this normalization. The normalization cannot be         the interpretation graph those paths that contain the cor-
preserved when non-generative models such as neural net-        rect (desired) label sequence. Its output graph G is called   c

works are integrated into the system. Other techniques,         the constrained interpretation graph (also known as forced
such as discriminative training methods, must be used in        alignment in the HMM literature), and contains all the
this case. Several authors have proposed such methods to        paths that correspond to the correct label sequence. The
train neural network/HMM speech recognizers at the word         constrained interpretation graph is then sent to the Viterbi
or sentence level 71], 72], 73], 74], 75], 76], 77], 78],       transformer which produces a graph G with a single cvit

 29], 67].                                                      path. This path is the \correct" path with the lowest
   Other globally trainable sequence recognition systems        penalty. Finally, a path scorer transformer takes G , and         cvit

avoid the diculties of statistical modeling by not resorting   simply computes its cumulated penalty C by adding up cvit

to graph-based techniques. The best example is Recurrent        the penalties along the path. The output of this GTN is

<!-- page 22 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                            22

the loss function for the current pattern:                      that integrate neural networks with time alignment 71],
                                                                 72], 76] or hybrid neural-network/HMM systems 29], 74],
                         E =C
                          vit        cvit                (11)    75].
The only label information that is required by the above           While it seems simple and satisfying, this training ar-
system is the sequence of desired character labels. No          chitecture has a aw that can potentially be fatal. The
knowledge of the correct segmentation is required on the        problem was already mentioned in Section II-C. If the
part of the supervisor, since it chooses among the segmen-      recognizer is a simple neural network with sigmoid out-
tations in the interpretation graph the one that yields the     put units, the minimum of the loss function is attained,
lowest penalty.                                                 not when the recognizer always gives the right answer, but
   The process of back-propagating gradients through the        when it ignores the input, and sets its output to a constant
Viterbi training GTN is now described. As explained in          vector with small values for all the components. This is
section IV, the gradients must be propagated backwards          known as the collapse problem. The collapse only occurs if
through all modules of the GTN, in order to compute gra-        the recognizer outputs can simultaneously take their min-
dients in preceding modules and thereafter tune their pa-       imum value. If on the other hand the recognizer's out-
rameters. Back-propagating gradients through the path           put layer contains RBF units with xed parameters, then
scorer is quite straightforward. The partial derivatives of     there is no such trivial solution. This is due to the fact
the loss function with respect to the individual penalties on   that a set of RBF with xed distinct parameter vectors
the constrained Viterbi path G are equal to 1, since the        cannot simultaneously take their minimum value. In this
                                    cvit

loss function is simply the sum of those penalties. Back-       case, the complete collapse described above does not occur.
propagating through the Viterbi Transformer is equally          However, this does not totally prevent the occurrence of a
simple. The partial derivatives of E with respect to the        milder collapse because the loss function still has a \at
                                            vit

penalties on the arcs of the constrained graph G are 1          spot" for a trivial solution with constant recognizer out-
                                                     c

for those arcs that appear in the constrained Viterbi path      put. This at spot is a saddle point, but it is attractive in
G , and 0 for those that do not. Why is it legitimate           almost all directions and is very dicult to get out of using
 cvit

to back-propagate through an essentially discrete function      gradient-based minimization procedures. If the parameters
such as the Viterbi Transformer? The answer is that the         of the RBFs are allowed to adapt, then the collapse prob-
Viterbi Transformer is nothing more than a collection of        lems reappears because the RBF centers can all converge
min functions and adders put together. It was shown in          to a single vector, and the underlying neural network can
Section IV that gradients can be back-propagated through        learn to produce that vector, and ignore the input. A dif-
min functions without adverse e ects. Back-propagation          ferent kind of collapse occurs if the width of the RBFs are
through the path selector transformer is similar to back-       also allowed to adapt. The collapse only occurs if a train-
propagation through the Viterbi transformer. Arcs in G          able module such as a neural network feeds the RBFs. The
that appear in G have the same gradient as the corre-
                                                          int
                                                                collapse does not occur in HMM-based speech recognition
                  c

sponding arc in G , i.e. 1 or 0, depending on whether the       systems because they are generative systems that produce
                  c

arc appear in G . The other arcs, i.e. those that do            normalized likelihoods for the input data (more on this
                 cvit

not have an alter ego in G because they do not contain          later). Another way to avoid the collapse is to train the
                                c

the right label have a gradient of 0. During the forward        whole system with respect to a discriminative training cri-
propagation through the recognition transformer, one in-        terion, such as maximizing the conditional probability of
stance of the recognizer for single character was created       the correct interpretations (correct sequence of class labels)
for each arc in the segmentation graph. The state of rec-       given the input image.
ognizer instances was stored. Since each arc penalty in            Another problem with Viterbi training is that the
G is produced by an individual output of a recognizer           penalty of the answer cannot be used reliably as a mea-
 int

instance, we now have a gradient (1 or 0) for each out-         sure of condence because it does not take low-penalty (or
put of each instance of the recognizer. Recognizer outputs      high-scoring) competing answers into account.
that have a non zero gradient are part of the correct an-       B. Discriminative Viterbi Training
swer, and will therefore have their value pushed down. The
gradients present on the recognizer outputs can be back-           A modication of the training criterion can circumvent
propagated through each recognizer instance. For each rec-      the collapse problem described above and at the same time
ognizer instance, we obtain a vector of partial derivatives     produce more reliable condence values. The idea is to not
of the loss function with respect to the recognizer instance    only minimize the cumulated penalty of the lowest penalty
parameters. All the recognizer instances share the same pa-     path with the correct interpretation, but also to somehow
rameter vector, since they are merely clones of each other,     increase the penalty of competing and possibly incorrect
therefore the full gradient of the loss function with respect   paths that have a dangerously low penalty. This type of
to the recognizer's parameter vector is simply the sum of       criterion is called discriminative, because it plays the good
the gradient vectors produced by each recognizer instance.      answers against the bad ones. Discriminative training pro-
Viterbi training, though formulated di erently, is often use    cedures can be seen as attempting to build appropriate
in HMM-based speech recognition systems 28]. Similar al-        separating surfaces between classes rather than to model
gorithms have been applied to speech recognition systems        individual classes independently of each other. For exam-

<!-- page 23 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                         23

                                                                       Loss Function
                                                                                    [0.1](+1)

                                                           [0.7](+1)
                                                                           +   Σ      −
                                                                                             [0.6](−1)
                                                      +
                                                                                                       +

                           3 [0.1](+1)
                                                                                                                          Gvit
                   Gcvit                          4 [0.6](+1)
                                                                                    3 [0.1](−1)    4 [0.4](−1)    1 [0.1](−1)
                                 Viterbi Tansformer

                           3 [0.1](+1)
                                                       4 [2.4](0)
                  Gc                                                                      Viterbi Transformer
                               3 [3.4](0)         4 [0.6](+1)

               "34"                 Path Selector
               Desired
               Answer                    3 [0.1](0)             4 [0.4](−1)               1 [0.1](−1)            Interpretation
                                         5 [2.3](0)             2 [1.3](0)                4 [2.4](0)                Graph

                                                                                                                     G int
                                              3 [3.4](0)                       4 [0.6](+1)
                                              4 [4.4](0)                       9 [1.2](0)

                                                                       (−1)        (+1)         (−1)
                                                                                                                 Recognition
                                                                       4       4            1                    Transfomer
                           W                    NN         NN       NN         NN          NN
                     Neural Net
                                                                                                                   T rec
                     Weights

                                                                                                                  Segmentation
                                                                                                                     Graph

                                                                                                                      Gseg

                                                                Segmenter

Fig. 20. Discriminative Viterbi Training GTN Architecture for a character string recognizer based on Heuristic Over-Segmentation. Quantities
     in square brackets are penalties computed during the forward propagation. Quantities in parentheses are partial derivatives computed
     during the backward propagation.

<!-- page 24 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                              24

ple, modeling the conditional distribution of the classes                         low penalty, but should have had a higher penalty since it
given the input image is more discriminative (focus-sing                          is not part of the desired answer.
more on the classication surface) than having a separate                            Variations of this technique have been used for the speech
generative model of the input data associated to each class                       recognition. Driancourt and Bottou 76] used a version of
(which, with class priors, yields the whole joint distribu-                       it where the loss function is saturated to a xed value.
tion of classes and inputs). This is because the conditional                      This can be seen as a generalization of the Learning Vector
approach does not need to assume a particular form for the                        Quantization 2 (LVQ-2) loss function 80]. Other variations
distribution of the input data.                                                   of this method use not only the Viterbi path, but the K-
   One example of discriminative criterion is the di erence                       best paths. The Discriminative Viterbi algorithm does not
between the penalty of the Viterbi path in the constrained                        have the aws of the non-discriminative version, but there
graph, and the penalty of the Viterbi path in the (uncon-                         are problems nonetheless. The main problem is that the
strained) interpretation graph, i.e. the di erence between                        criterion does not build a margin between the classes. The
the penalty of the best correct path, and the penalty of                          gradient is zero as soon as the penalty of the constrained
the best path (correct or incorrect). The corresponding                           Viterbi path is equal to that of the Viterbi path. It would
GTN training architecture is shown in gure 20. The left                          be desirable to push up the penalties of the wrong paths
side of the diagram is identical to the GTN used for non-                         when they are dangerously close to the good one. The
discriminative Viterbi training. This loss function reduces                       following section presents a solution to this problem.
the risk of collapse because it forces the recognizer to in-
creases the penalty of wrongly recognized objects. Dis-                           C. Forward Scoring, and Forward Training
criminative training can also be seen as another example
of error correction procedure, which tends to minimize the                           While the penalty of the Viterbi path is perfectly appro-
di erence between the desired output computed in the left                         priate for the purpose of recognition, it gives only a partial
half of the GTN in gure 20 and the actual output com-                            picture of the situation. Imagine the lowest penalty paths
puted in the right half of gure 20.                                              corresponding to several dierent segmentations produced
   Let the discriminative Viterbi loss function be denoted                        the same answer (the same label sequence). Then it could
E , and let us call C the penalty of the Viterbi path in
 dvit                         cvit                                                be argued that the overall penalty for the interpretation
the constrained graph, and C the penalty of the Viterbi
                                      vit                                         should be smaller than the penalty obtained when only one
path in the unconstrained interpretation graph:                                   path produced that interpretation, because multiple paths
                     E =C ;C                             (12)                     with identical label sequences are more evidence that the
                         dvit         cvit         vit
                                                                                  label sequence is correct. Several rules can be used com-
E is always positive since the constrained graph is a                             pute the penalty associated to a graph that contains several
                                                                                  parallel paths. We use a combination rule borrowed from
 dvit

subset of the paths in the interpretation graph, and the
Viterbi algorithm selects the path with the lowest total                          a probabilistic interpretation of the penalties as negative
penalty. In the ideal case, the two paths C and C                                 log posteriors. In a probabilistic framework, the posterior
                                                                                  probability for the interpretation should be the sum of the
                                                               cvit         vit

coincide, and E is zero.
                                                                                  posteriors for all the paths that produce that interpreta-
                dvit

   Back-propagating gradients through the discriminative
Viterbi GTN adds some \negative" training to the pre-                             tion. Translated in terms of penalties, the penalty of an
viously described non-discriminative training. Figure 20                          interpretation should be the negative logarithm of the sum
shows how the gradients are back-propagated. The left                             of the negative exponentials of the penalties of the individ-
half is identical to the non-discriminative Viterbi training                      ual paths. The overall penalty will be smaller than all the
GTN, therefore the back-propagation is identical. The gra-                        penalties of the individual paths.
dients back-propagated through the right half of the GTN                             Given an interpretation, there is a well known method,
are multiplied by -1, since C contributes to the loss with
                                     vit                                          called the forward algorithm for computing the above quan-
a negative sign. Otherwise the process is similar to the left                     tity eciently 28]. The penalty computed with this pro-
half. The gradients on arcs of G get positive contribu-
                                             int                                  cedure for a particular interpretation is called the forward
tions from the left half and negative contributions from the                      penalty. Consider again the concept of constrained graph,
right half. The two contributions must be added, since the                        the subgraph of the interpretation graph which contains
penalties on G arcs are sent to the two halves through
                int                                                               only the paths that are consistent with a particular label
a \Y" connection in the forward pass. Arcs in G that                  int         sequence. There is one constrained graph for each pos-
appear neither in G nor in G have a gradient of zero.
                        vit                cvit                                   sible label sequence (some may be empty graphs, which
They do not contribute to the cost. Arcs that appear in                           have innite penalties). Given an interpretation, running
both G and G also have zero gradient. The -1 contri-
          vit    cvit                                                             the forward algorithm on the corresponding constrained
bution from the right half cancels the the +1 contribution                        graph gives the forward penalty for that interpretation.
from the left half. In other words, when an arc is rightfully                     The forward algorithm proceeds in a way very similar to
part of the answer, there is no gradient. If an arc appears                       the Viterbi algorithm, except that the operation used at
in G but not in G , the gradient is +1. The arc should
    cvit                 vit                                                      each node to combine the incoming cumulated penalties,
have had a lower penalty to make it to G . If an arc is  vit                      instead of being the min function is the so-called logadd
in G but not in G , the gradient is -1. The arc had a
    vit                  cvit                                                     operation, which can be seen as a \soft" version of the min

<!-- page 25 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                             25

function:                                                                                                        Edforw
                    fn = logaddi2Un (ci + fsi ):                   (13)
where f = 0, Un is the set of upstream arcs of node n,
        start                                                                                         Cdforw
                                                                                                               +         −
ci is the penalty on arc i, and                                                                                              Cforw

                                               Xn e x )
            logadd(x  x  : : :  xn ) = ; log(         ; i       (14)        Constrained
                                                                                                Forward Scorer

                      1   2
                                                                                                                 Gc       Forward Scorer
                                                   i=1
                                                                               Interpretation Graph

                                                                               Desired
Note that because of numerical inaccuracies, it is better                      Sequence               Path Selector
to factorize the largest e;xi (corresponding to the smallest                                                                    Gint
penalty) out of the logarithm.                                                            Interpretation Graph
   An interesting analogy can be drawn if we consider that
a graph on which we apply the forward algorithm is equiv-                                                          Recognition
                                                                                                                   Transformer
alent to a neural network on which we run a forward prop-
agation, except that multiplications are replaced by addi-                Fig. 21.    Discriminative Forward Training GTN Architecture
tions, the additions are replaced by logadds, and there are                   for a character string recognizer based on Heuristic Over-
no sigmoids.                                                                  Segmentation.
   One way to understand the forward algorithm is to think
about multiplicative scores (e.g., probabilities) instead of              G
additive penalties on the arcs: score = exp(; penalty ). In                              @E = e;fn X @E efdi ;ci
                                                                           c

that case the Viterbi algorithm selects the path with the                                @fn                                    (15)
largest cumulative score (with scores multiplied along the                                         i2Dn @fdi
path), whereas the forward score is the sum of the cumula-                where Dn = farc i with source si = ng is the set of down-
tive scores associated to each of the possible paths from the             stream arcs from node n. From the above derivatives, the
start to the end node. The forward penalty is always lower                derivatives with respect to the arc penalties are obtained:
than the cumulated penalty on any of the paths, but if one                                     @E = @E e;ci ;fsi fdi
path \dominates" (with a much lower penalty), its penalty                                                                +
                                                                                                                                           (16)
is almost equal to the forward penalty. The forward algo-                                      @ci @fdi
rithm gets its name from the forward pass of the well-known               This can be seen as a \soft" version of the back-propagation
Baum-Welsh algorithm for training Hidden Markov Mod-                      through a Viterbi scorer and transformer. All the arcs in
els 28]. Section VIII-E gives more details on the relation                G have an inuence on the loss function. The arcs that
between this work and HMMs.
                                                                           c

                                                                          belong to low penalty paths have a larger inuence. Back-
   The advantage of the forward penalty with respect to                   propagation through the path selector is the same as before.
the Viterbi penalty is that it takes into account all the                 The derivative with respect to G arcs that have an alter
di erent ways to produce an answer, and not just the one
                                                                                                                   int

                                                                          ego in G are simply copied from the corresponding arc in
with the lowest penalty. This is important if there is some
                                                                                    c

                                                                          G . The derivatives with respect to the other arcs are 0.
ambiguity in the segmentation, since the combined forward
                                                                           c

                                                                             Several authors have applied the idea of back-
penalty of two paths C and C associated with the same
                              1      2                                    propagating gradients through a forward scorer to train
label sequence may be less than the penalty of a path C              3    speech recognition systems, including Bridle and his -net
associated with another label sequence, even though the                   model 73] and Ha ner and his  -TDNN model 81], but
penalty of C might be less than any one of C or C .
                3                                         1    2          these authors recommended discriminative training as de-
   The Forward training GTN is only a slight modica-                     scribed in the next section.
tion of the previously introduced Viterbi training GTN. It
suces to turn the Viterbi transformers in Figure 19 into                 D. Discriminative Forward Training
Forward Scorers that take an interpretation graph as input                   The information contained in the forward penalty can be
an produce the forward penalty of that graph on output.                   used in another discriminative training criterion which we
Then the penalties of all the paths that contain the correct              will call the discriminative forward criterion. This criterion
answer are lowered, instead of just that of the best one.                 corresponds to maximization of the posterior probability of
   Back-propagating through the forward penalty computa-                  choosing the paths associated with the correct interpreta-
tion (the forward transformer) is quite di erent from back-               tion. This posterior probability is dened as the exponen-
propagating through a Viterbi transformer. All the penal-                 tial of the minus the constrained forward penalty, normal-
ties of the input graph have an inuence on the forward                   ized by the exponential of minus the unconstrained forward
penalty, but penalties that belong to low-penalty paths                   penalty. Note that the forward penalty of the constrained
have a stronger inuence. Computing derivatives with re-                  graph is always larger or equal to the forward penalty of the
spect to the forward penalties fn computed at each n node                 unconstrained interpretation graph. Ideally, we would like
of a graph is done by back-propagation through the graph                  the forward penalty of the constrained graph to be equal to

<!-- page 26 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                             26

the forward penalty of the complete interpretation graph.        E. Remarks on Discriminative Training
Equality between those two quantities is achieved when the          In the above discussion, the global training criterion
combined penalties of the paths with the correct label se-       was given a probabilistic interpretation, but the individ-
quence is negligibly small compared to the penalties of all      ual penalties on the arcs of the graphs were not. There are
the other paths, or that the posterior probability associ-       good reasons for that. For example, if some penalties are
ated to the paths with the correct interpretation is almost      associated to the di erent class labels, they would (1) have
1, which is precisely what we want. The corresponding            to sum to 1 (class posteriors), or (2) integrate to 1 over the
GTN training architecture is shown in gure 21.                  input domain (likelihoods).
   Let the di erence be denoted E , and let us call
                                      dforw
                                                                    Let us rst discuss the rst case (class posteriors normal-
Ccforw  the forward penalty of the constrained graph, and        ization). This local normalization of penalties may elimi-
C the forward penalty of the complete interpretation
 forw
                                                                 nate information that is important for locally rejecting all
graph:                                                           the classes 82], e.g., when a piece of image does not cor-
                   Edforw =C  cforw ;C forw               (17)   respond to a valid character class, because some of the
Edforw  is always positive since the constrained graph is a      segmentation candidates may be wrong. Although an ex-
subset of the paths in the interpretation graph, and the         plicit \garbage class" can be introduced in a probabilistic
forward penalty of a graph is always larger than the for-        framework to address that question, some problems remain
ward penalty of a subgraph of this graph. In the ideal case,     because it is dicult to characterize such a class probabilis-
the penalties of incorrect paths are innitely large, there-     tically and to train a system in this way (it would require
fore the two penalties coincide and E  dforw is zero. Readers    a density model of unseen or unlabeled samples).
familiar with the Boltzmann machine connectionist model             The probabilistic interpretation of individual variables
might recognize the constrained and unconstrained graphs         plays an important role in the Baum-Welsh algorithm
as analogous to the \clamped" (constrained by the ob-            in combination with the Expectation-Maximization proce-
served values of the output variable) and \free" (uncon-         dure. Unfortunately, those methods cannot be applied to
strained) phases of the Boltzmann machine algorithm 13].         discriminative training criteria, and one is reduced to us-
   Back-propagating derivatives through the discriminative       ing gradient-based methods. Enforcing the normalization
Forward GTN distributes gradients more evenly than in the        of the probabilistic quantities while performing gradient-
Viterbi case. Derivatives are back-propagated through the        based learning is complex, inecient, time consuming, and
left half of the the GTN in Figure 21 down to the interpre-      creates ill-conditioning of the loss-function.
tation graph. Derivatives are negated and back-propagated           Following 82], we therefore prefer to postpone normal-
through the right-half, and the result for each arc is added     ization as far as possible (in fact, until the nal decision
to the contribution from the left half. Each arc in G      int   stage of the system). Without normalization, the quanti-
now has a derivative. Arcs that are part of a correct path       ties manipulated in the system do not have a direct prob-
have a positive derivative. This derivative is very large if     abilistic interpretation.
an incorrect path has a lower penalty than all the correct          Let us now discuss the second case (using a generative
paths. Similarly, the derivatives with respect to arcs that      model of the input). Generative models build the boundary
are part of a low-penalty incorrect path have a large nega-      indirectly, by rst building an independent density model
tive derivative. On the other hand, if the penalty of a path     for each class, and then performing classication decisions
associated with the correct interpretation is much smaller       on the basis of these models. This is not a discriminative
than all other paths, the loss function is very close to 0       approach in that it does not focus on the ultimate goal of
and almost no gradient is back-propagated. The training          learning, which in this case is to learn the classication de-
therefore concentrates on examples of images which yield a       cision surface. Theoretical arguments 6], 7] suggest that
classication error, and furthermore, it concentrates on the     estimating input densities when the real goal is to obtain
pieces of the image which cause that error. Discriminative       a discriminant function for classication is a suboptimal
forward training is an elegant and ecient way of solving        strategy. In theory, the problem of estimating densities in
the infamous credit assignment problem for learning ma-          high-dimensional spaces is much more ill-posed than nd-
chines that manipulate \dynamic" data structures such as         ing decision boundaries.
graphs. More generally, the same idea can be used in all            Even though the internal variables of the system do not
situations where a learning machine must choose between          have a direct probabilistic interpretation, the overall sys-
discrete alternative interpretations.                            tem can still be viewed as producing posterior probabilities
   As previously, the derivatives on the interpretation graph    for the classes. In fact, assuming that a particular label se-
penalties can then be back-propagated into the character         quence is given as the \desired sequence" to the GTN in
recognizer instances. Back-propagation through the char-         gure 21, the exponential of minus E     dforw  can be inter-
acter recognizer gives derivatives on its parameters. All the    preted as an estimate of the posterior probability of that
gradient contributions for the di erent candidate segments       label sequence given the input. The sum of those posteriors
are added up to obtain the total gradient associated to one      for all the possible label sequences is 1. Another approach
pair (input image, correct label sequence), that is, one ex-     would consists of directly minimizing an approximation of
ample in the training set. A step of stochastic gradient         the number of misclassications 83] 76]. We prefer to use
descent can then be applied to update the parameters.            the discriminative forward loss function because it causes

<!-- page 27 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                  27

                 "U"

             Recognizer

Fig. 22. Explicit segmentation can be avoided by sweeping a recog-
     nizer at every possible location in the input eld.

less numerical problems during the optimization. We will                               $
see in Section X-C that this is a good way to obtain scores
on which to base a rejection strategy. The important point           Fig. 23. A Space Displacement Neural Network is a convolutional
being made here is that one is free to choose any param-                  network that has been replicated over a wide input eld.
eterization deemed appropriate for a classication model.
The fact that a particular parameterization uses internal
variables with no clear probabilistic interpretation does not        characters within a string may have widely varying sizes
make the model any less legitimate than models that ma-              and baseline positions. Therefore the recognizer must be
nipulate normalized quantities.                                      very robust to shifts and size variations.
   An important advantage of global and discriminative                  These three problems are elegantly circumvented if a
training is that learning focuses on the most important              convolutional network is replicated over the input eld.
errors, and the system learns to integrate the ambigui-              First of all, as shown in section III, convolutional neu-
ties from the segmentation algorithm with the ambigui-               ral networks are very robust to shifts and scale varia-
ties of the character recognizer. In Section IX we present           tions of the input image, as well as to noise and extra-
experimental results with an on-line handwriting recogni-            neous marks in the input. These properties take care of
tion system that conrm the advantages of using global               the latter two problems mentioned in the previous para-
training versus separate training. Experiments in speech             graph. Second, convolutional networks provide a drastic
recognition with hybrids of neural networks and HMMs                 saving in computational requirement when replicated over
also showed marked improvements brought by global train-             large input elds. A replicated convolutional network, also
ing 77], 29], 67], 84].                                              called a Space Displacement Neural Network or SDNN 27],
                                                                     is shown in Figure 23. While scanning a recognizer can
     VII. Multiple Object Recognition: Space                         be prohibitively expensive in general, convolutional net-
            Displacement Neural Network                              works can be scanned or replicated very eciently over
   .                                                                 large, variable-size input elds. Consider one instance of
   There is a simple alternative to explicitly segmenting im-        a convolutional net and its alter ego at a nearby location.
ages of character strings using heuristics. The idea is to           Because of the convolutional nature of the network, units
sweep a recognizer at all possible locations across a nor-           in the two instances that look at identical locations on the
malized image of the entire word or string as shown in               input have identical outputs, therefore their states do not
Figure 22. With this technique, no segmentation heuris-              need to be computed twice. Only a thin \slice" of new
tics are required since the system essentially examines all          states that are not shared by the two network instances
the possible segmentations of the input. However, there              needs to be recomputed. When all the slices are put to-
are problems with this approach. First, the method is in             gether, the result is simply a larger convolutional network
general quite expensive. The recognizer must be applied              whose structure is identical to the original network, except
at every possible location on the input, or at least at a            that the feature maps are larger in the horizontal dimen-
large enough subset of locations so that misalignments of            sion. In other words, replicating a convolutional network
characters in the eld of view of the recognizers are small          can be done simply by increasing the size of the elds over
enough to have no e ect on the error rate. Second, when              which the convolutions are performed, and by replicating
the recognizer is centered on a character to be recognized,          the output layer accordingly. The output layer e ectively
the neighbors of the center character will be present in the         becomes a convolutional layer. An output whose receptive
eld of view of the recognizer, possibly touching the cen-           eld is centered on an elementary object will produce the
ter character. Therefore the recognizer must be able to              class of this object, while an in-between output may indi-
correctly recognize the character in the center of its input         cate no character or contain rubbish. The outputs can be
eld, even if neighboring characters are very close to, or           interpreted as evidences for the presence of objects at all
touching the central character. Third, a word or charac-             possible positions in the input eld.
ter string cannot be perfectly size normalized. Individual              The SDNN architecture seems particularly attractive for

<!-- page 28 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                28

recognizing cursive handwriting where no reliable segmen-                                             Viterbi Answer
tation heuristic exists. Although the idea of SDNN is quite
old, and very attractive by its simplicity, it has not gener-             Viterbi Graph
ated wide interest until recently because, as stated above,
it puts enormous demands on the recognizer 26], 27]. In                                            Viterbi Transformer
speech recognition, where the recognizer is at least one
order of magnitude smaller, replicated convolutional net-                 Interpretation Graph
works are easier to implement, for instance in Ha ner's                   Character
Multi-State TDNN model 78], 85].                                          Model                          Compose
                                                                          Transducer
A. Interpreting the Output of an SDNN with a GTN                                                        S....c.....r......i....p....t
   The output of an SDNN is a sequence of vectors which                   SDNN Output                   s....e.....n.....e.j...o.T
                                                                                                        5......a...i...u......p.....f
encode the likelihoods, penalties, or scores of nding char-
acter of a particular class label at the corresponding lo-                                             SDNN
cation in the input. A post-processor is required to pull                                              Transformer
out the best possible label sequence from this vector se-
quence. An example of SDNN output is shown in Fig-               Fig. 24. A Graph Transformer pulls out the best interpretation from
ure 25. Very often, individual characters are spotted by              the output of the SDNN.
several neighboring instances of the recognizer, a conse-
quence of the robustness of the recognizer to horizontal                    C1      C3 C5
translations. Also quite often, characters are erroneously                                           2345                           Answer

detected by recognizer instances that see only a piece of                                      Compose + Viterbi
a character. For example a recognizer instance that only                                       2     33 4 5
                                                                                                                                    SDNN
                                                                                                                                    Output
sees the right third of a \4" might output the label 1. How                                                                             F6
can we eliminate those extraneous characters from the out-
put sequence and pull-out the best interpretation? This                                                                                 Input
can be done using a new type of Graph Transformer with
two input graphs as shown in Figure 24. The sequence of
vectors produced by the SDNN is rst coded into a linear
graph with multiple arcs between pairs of successive nodes.
Each arc between a particular pair of nodes contains the         Fig. 25. An example of multiple character recognition with SDNN.
                                                                      With SDNN, no explicit segmentation is performed.
label of one of the possible categories, together with the
penalty produced by the SDNN for that class label at that
location. This graph is called the SDNN Output Graph.            B. Experiments with SDNN
The second input graph to the transformer is a grammar
transducer, more specically a nite-state transducer 86],          In a series of experiments, LeNet-5 was trained with the
that encodes the relationship between input strings of class     goal of being replicated so as to recognize multiple char-
labels and corresponding output strings of recognized char-      acters without segmentations. The data was generated
acters.The transducer is a weighted nite state machine (a       from the previously described Modied NIST set as fol-
graph) where each arc contains a pair of labels and possibly     lows. Training images were composed of a central char-
a penalty. Like a nite-state machine, a transducer is in a      acter, anked by two side characters picked at random in
state and follows an arc to a new state when an observed         the training set. The separation between the bounding
input symbol matches the rst symbol in the symbol pair          boxes of the characters were chosen at random between -1
attached to the arc. At this point the transducer emits the      and 4 pixels. In other instances, no central character was
second symbol in the pair together with a penalty that com-      present, in which case the desired output of the network
bines the penalty of the input symbol and the penalty of         was the blank space class. In addition, training images
the arc. A transducer therefore transforms a weighted sym-       were degraded with 10% salt and pepper noise (random
bol sequence into another weighted symbol sequence. The          pixel inversions).
graph transformer shown in gure 24 performs a composi-             Figures 25 and 26 show a few examples of success-
tion between the recognition graph and the grammar trans-        ful recognitions of multiple characters by the LeNet-5
ducer. This operation takes every possible sequence corre-       SDNN. Standard techniques based on Heuristic Over-
sponding to every possible path in the recognition graph         Segmentation would fail miserably on many of those ex-
and matches them with the paths in the grammar trans-            amples. As can be seen on these examples, the network
ducer. The composition produces the interpretation graph,        exhibits striking invariance and noise resistance properties.
which contains a path for each corresponding output label        While some authors have argued that invariance requires
sequence. This composition operation may seem combina-           more sophisticated models than feed-forward neural net-
torially intractable, but it turns out there exists an ecient   works 87], LeNet-5 exhibits these properties to a large ex-
algorithm for it described in more details in Section VIII.      tent.

<!-- page 29 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                    29

                                              540                                  1114
                                  55            4 0                             1 1 1 441

                 Answer              678                                           3514
                 SDNN
                 output           6 777  88                                     3 55 114
                 F6

                 Input

Fig. 26. An SDNN applied to a noisy image of digit string. The digits shown in the SDNN output represent the winning class labels, with
     a lighter grey level for high-penalty answers.

   Similarly, it has been suggested that accurate recognition        which they can be implemented on parallel hardware. Spe-
of multiple overlapping objects require explicit mechanisms          cialized analog/digital chips have been designed and used
that would solve the so-called feature binding problem 87].          in character recognition, and in image preprocessing appli-
As can be seen on Figures 25 and 26, the network is able to          cations 88]. However the rapid progress of conventional
tell the characters apart, even when they are closely inter-         processor technology with reduced-precision vector arith-
twined, a task that would be impossible to achieve with the          metic instructions (such as Intel's MMX) make the success
more classical Heuristic Over-Segmentation technique. The            of specialized hardware hypothetical at best.
SDNN is also able to correctly group disconnected pieces                Short video clips of the LeNet-5 SDNN can be viewed at
of ink that form characters. Good examples of that are               http://www.research.att.com/~yann/ocr.
shown in the upper half of gure 26. In the top left ex-
ample, the 4 and the 0 are more connected to each other              C. Global Training of SDNN
than they are connected with themselves, yet the system                 In the above experiments, the string image were arti-
correctly identies the 4 and the 0 as separate objects. The         cially generated from individual character. The advantage
top right example is interesting for several reasons. First          is that we know in advance the location and the label of
the system correctly identies the three individual ones.            the important character. With real training data, the cor-
Second, the left half and right half of disconnected 4 are           rect sequence of labels for a string is generally available,
correctly grouped, even though no geometrical information            but the precise locations of each corresponding character
could decide to associate the left half to the vertical bar on       in the input image are unknown.
its left or on its right. The right half of the 4 does cause            In the experiments described in the previous section, the
the appearance of an erroneous 1 on the SDNN output,                 best interpretation was extracted from the SDNN output
but this one is removed by the character model transducer            using a very simple graph transformer. Global training of
which prevents characters from appearing on contiguous               an SDNN can be performed by back-propagating gradients
outputs.                                                             through such graph transformers arranged in architectures
   Another important advantage of SDNN is the ease with              similar to the ones described in section VI.

<!-- page 30 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                           30

                                   Edforw                                      zip code recognition 91], and more recent experiments in
                                                                               on-line handwriting recognition 38] have demonstrated the
                                                                               idea of globally-trained SDNN/HMM hybrids. SDNN is an
                       Cdforw
                                  +              −                             extremely promising and attractive technique for OCR, but
                                                             Cforw             so far it has not yielded better results than Heuristic Over-
                                                                               Segmentation. We hope that these results will improve as
                   Forward Scorer
                                                                               more experience is gained with these models.
   Constrained
   Interpretation Graph
                                   Gc               Forward Scorer
                                                                               D. Object Detection and Spotting with SDNN
   Desired
                          Path Selector                                           An interesting application of SDNNs is object detection
   Sequence
                                                                               and spotting. The invariance properties of Convolutional
                                                                        Gint   Networks, combined with the eciency with which they
          Interpretation Graph
                                                                               can be replicated over large elds suggest that they can
          Character                                                            be used for \brute force" object spotting and detection in
          Model
          Transducer
                                          Compose
                                                                               large images. The main idea is to train a single Convolu-
                                                                               tional Network to distinguish images of the object of inter-
          SDNN Output
                                        S....c.....r......i....p....t
                                        s....e.....n.....e.j...o.T             est from images present in the background. In utilization
                                                                               mode, the network is replicated so as to cover the entire
                                        5......a...i...u......p.....f

                                      SDNN                                     image to be analyzed, thereby forming a two-dimensional
                                      Transformer
                                                                               Space Displacement Neural Network. The output of the
                                                                               SDNN is a two-dimensional plane in which activated units
Fig. 27. A globally trainable SDNN/HMM hybrid system expressed                 indicate the presence of the object of interest in the corre-
     as a GTN.                                                                 sponding receptive eld. Since the sizes of the objects to
                                                                               be detected within the image are unknown, the image can
                                                                               be presented to the network at multiple resolutions, and
   This is somewhat equivalent to modeling the output                          the results at multiple resolutions combined. The idea has
of an SDNN with a Hidden Markov Model. Globally                                been applied to face location, 93], address block location
trained, variable-size TDNN/HMM hybrids have been used                         on envelopes 94], and hand tracking in video 95].
for speech recognition and on-line handwriting recogni-                           To illustrate the method, we will consider the case of
tion 77], 89], 90], 67]. Space Displacement Neural Net-                        face detection in images as described in 93]. First, images
works have been used in combination with HMMs or other                         containing faces at various scales are collected. Those im-
elastic matching methods for handwritten word recogni-                         ages are ltered through a zero-mean Laplacian lter so as
tion 91], 92].                                                                 to remove variations in global illumination and low spatial
   Figure 27 shows the graph transformer architecture for                      frequency illumination gradients. Then, training samples
training an SDNN/HMM hybrid with the Discriminative                            of faces and non-faces are manually extracted from those
Forward Criterion. The top part is comparable to the top                       images. The face sub-images are then size normalized so
part of gure 21. On the right side the composition of the                     that the height of the entire face is approximately 20 pixels
recognition graph with the grammar gives the interpreta-                       while keeping fairly large variations (within a factor of two).
tion graph with all the possible legal interpretations. On                     The scale of background sub-images are picked at random.
the left side the composition is performed with a grammar                      A single convolutional network is trained on those samples
that only contains paths with the desired sequence of la-                      to classify face sub-images from non-face sub-images.
bels. This has a somewhat similar function to the path                            When a scene image is to be analyzed, it is rst ltered
selector used in the previous section. Like in Section VI-D                    through the Laplacian lter, and sub-sampled at powers-
the loss function is the di erence between the forward score                   of-two resolutions. The network is replicated over each of
obtained from the left half and the forward score obtained                     multiple resolution images. A simple voting technique is
from the right half. To back-propagate through the com-                        used to combine the results from multiple resolutions.
position transformer, we need to keep a record of which arc                       A two-dimensional version of the global training method
in the recognition graph originated which arcs in the inter-                   described in the previous section can be used to allevi-
pretation graph. The derivative with respect to an arc in                      ate the need to manually locate faces when building the
the recognition graph is equal to the sum of the derivatives                   training sample 93]. Each possible location is seen as an
with respect to all the arcs in the interpretation graph that                  alternative interpretation, i.e. one of several parallel arcs
originated from it. Derivative can also be computed for the                    in a simple graph that only contains a start node and an
penalties on the grammar graph, allowing to learn them as                      end node.
well. As in the previous example, a discriminative criterion                      Other authors have used Neural Networks, or other clas-
must be used, because using a non-discriminative criterion                     siers such as Support Vector Machines for face detection
could result in a collapse e ect if the network's output RBF                   with great success 96], 97]. Their systems are very similar
are adaptive. The above training procedure can be equiv-                       to the one described above, including the idea of presenting
alently formulated in term of HMM. Early experiments in                        the image to the network at multiple scales. But since those

<!-- page 31 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                           31

systems do not use Convolutional Networks, they cannot         path and a corresponding pair of input/output sequences
take advantage of the speedup described here, and have to      (S ,S ) in the transducer graph. The weights on the arcs
                                                                 out   in

rely on other techniques, such as pre-ltering and real-time   of the output graph are obtained by adding the weights
tracking, to keep the computational requirement within         from the matching arcs in the input acceptor and trans-
reasonable limits. In addition, because those classiers are   ducer graphs. In the rest of the paper, we will call this
much less invariant to scale variations than Convolutional     graph composition operation using transducers the (stan-
Networks, it is necessary to multiply the number of scales     dard) transduction operation.
at which the images are presented to the classier.               A simple example of transduction is shown in Figure 28.
                                                               In this simple example, the input and output symbols on
    VIII. Graph Transformer Networks and                       the transducer arcs are always identical. This type of trans-
                   Transducers                                 ducer graph is called a grammar graph. To better under-
   In Section IV, Graph Transformer Networks (GTN)             stand the transduction operation, imagine two tokens sit-
were introduced as a generalization of multi-layer, multi-     ting each on the start nodes of the input acceptor graph
module networks where the state information is repre-          and the transducer graph. The tokens can freely follow
sented as graphs instead of xed-size vectors. This section    any arc labeled with a null input symbol. A token can
re-interprets the GTNs in the framework of Generalized         follow an arc labeled with a non-null input symbol if the
Transduction, and proposes a powerful Graph Composition        other token also follows an arc labeled with the same in-
algorithm.                                                     put symbol. We have an acceptable trajectory when both
                                                               tokens reach the end nodes of their graphs (i.e. the tokens
A. Previous Work                                               have reached the terminal conguration). This trajectory
   Numerous authors in speech recognition have used            represents a sequence of input symbols that complies with
Gradient-Based Learning methods that integrate graph-          both the acceptor and the transducer. We can then collect
based statistical models (notably HMM) with acoustic           the corresponding sequence of output symbols along the
recognition modules, mainly Gaussian mixture models, but       trajectory of the transducer token. The above procedure
also neural networks 98], 78], 99], 67]. Similar ideas have    produces a tree, but a simple technique described in Sec-
been applied to handwriting recognition (see 38] for a re-     tion VIII-C can be used to avoid generating multiple copies
view). However, there has been no proposal for a system-       of certain subgraphs by detecting when a particular output
atic approach to multi-layer graph-based trainable systems.    state has already been seen.
The idea of transforming graphs into other graphs has re-         The transduction operation can be performed very e-
ceived considerable interest in computer science, through      ciently 106], but presents complex book-keeping problems
the concept of weighted nite-state transducers 86]. Trans-    concerning the handling of all combinations of null and non
ducers have been applied to speech recognition 100] and        null symbols. If the weights are interpreted as probabilities
language translation 101], and proposals have been made        (normalized appropriately) then an acceptor graph repre-
for handwriting recognition 102]. This line of work has        sents a probability distribution over the language dened
been mainly focused on ecient search algorithms 103]          by the set of label sequences associated to all possible paths
and on the algebraic aspects of combining transducers and      (from the start to the end node) in the graph.
graphs (called acceptors in this context), but very little        An example of application of the transduction opera-
e ort has been devoted to building globally trainable sys-     tion is the incorporation of linguistic constraints (a lexicon
tems out of transducers. What is proposed in the follow-       or a grammar) when recognizing words or other character
ing sections is a systematic approach to automatic training    strings. The recognition transformer produces the recog-
in graph-manipulating systems. A di erent approach to          nition graph (an acceptor graph) by applying the neural
graph-based trainable systems, called Input-Output HMM,        network recognizer to each candidate segment. This ac-
was proposed in 104], 105].                                    ceptor graph is composed with a transducer graph for the
                                                               grammar. The grammar transducer contains a path for
B. Standard Transduction                                       each legal sequence of symbol, possibly augmented with
  In the established framework of nite-state transduc-        penalties to indicate the relative likelihoods of the possi-
ers 86], discrete symbols are attached to arcs in the graphs.  ble sequences. The arcs contain identical input and output
Acceptor graphs have a single symbol attached to each          symbols. Another example of transduction was mentioned
arc whereas transducer graphs have two symbols (an input       in Section V: the path selector used in the heuristic over-
symbol and an output symbol). A special null symbol is         segmentation training GTN is implementable by a compo-
absorbed by any other symbol (when concatenating sym-          sition. The transducer graph is linear graph which con-
bols to build a symbol sequence). Weighted transducers         tains the correct label sequence. The composition of the
and acceptors also have a scalar quantity attached to each     interpretation graph with this linear graph yields the con-
arc. In this framework, the composition operation takes as     strained graph.
input an acceptor graph and a transducer graph and builds C. Generalized Transduction
an output acceptor graph. Each path in this output graph
(with symbol sequence S ) corresponds to one path (with
                         out                                  If the data structures associated to each arc took only
symbol sequence S ) in the input acceptor graph and one a nite number of values, composing the input graph and
                   in

<!-- page 32 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                                     32

an appropriate transducer would be a sound solution. For                                        interpretation graph
                                                                                                                                 interpretations:
                                                                                                                                 cut (2.0)
our applications however, the data structures attached to                                                         "t" 0.8
                                                                                                                                 cap (0.8)
                                                                                                                                 cat (1.4)

the arcs of the graphs may be vectors, images or other                                                      0.8
high-dimensional objects that are not readily enumerated.
                                                                                                      "u"
                                                                                                                      "p" 0.2

We present a new composition operation that solves this
                                                                                   "c" 0.4
                                                                                                       "a" 0.2
                                                                                                                                             grammar graph
problem.
                                                                                                                      "t"
                                                                                                                         0.8
                                                                                                                                                      "r"      "n"
   Instead of only handling graphs with discrete symbols                                                                                   "a"

and penalties on the arcs, we are interested in considering

                                                                            Graph Composition
                                                                                                                                                       "t"

graphs whose arcs may carry complex data structures, in-                                           match      match      match
                                                                                                                                  "b"       "u"

cluding continuous-valued data structures such as vectors
                                                                                                   & add      & add      & add                          "t"    "e"
                                                                                                                                            "u"

and images. Composing such graphs requires additional
                                                                                                                                  "c"                 "r"     "e"
                                                                                                                                            "a"        "p"
information:                                                                                                                                      "t" "r"

   When examining a pair of arcs (one from each input
                                                                                                                                                              "d"

graph), we need a criterion to decide whether to create cor-                                      "c" 0.4              "x" 0.1
responding arc(s) and node(s) in the output graph, based
                                                                                                                                  "p" 0.2

on the information attached to the input arcs. We can de-                                         "o" 1.0              "a" 0.2                       Recognition
                                                                                                                                                     Graph
cide to build an arc, several arcs, or an entire sub-graph                                         "d" 1.8            "u" 0.8
                                                                                                                                 "t" 0.8

with several nodes and arcs.
   When that criterion is met, we must build the corre- Fig. the    28. Example of composition of the recognition graph with
                                                                        grammar graph in order to build an interpretation that is
sponding arc(s) and node(s) in the output graph and com-            consistent with both of them. During the forward propagation
pute the information attached to the newly created arc(s)           (dark arrows), the methods check and fprop are used. Gradients
as a function the the information attached to the input             (dashed arrows) are back-propagated with the application of the
arcs.                                                               method bprop.
   These functions are encapsulated in an object called a
Composition Transformer. An instance of Composition
Transformer implements three methods:                          jectory is acceptable (i.e. both tokens simultaneously reach
   check(arc1, arc2)
                                                               the end nodes of their graphs). The management of null
compares the data structures pointed to by arcs arc1 (from transitions
                                                               simulation
                                                                             is a straightforward modication of the token
                                                                             function.  Before enumerating the possible non
the rst graph) and arc2 (from the second graph) and re- null joint token transitions,           we loop on the possible null
turns a boolean indicating whether corresponding arc(s) transitions of each token, recursively              call the token sim-
should be created in the output graph.                         ulation  function,   and  nally  call the  method   fprop. The
   fprop(ngraph, upnode, downnode, arc1, arc2)
                                                               safest way   for identifying  acceptable
is called when check(arc1, arc2) returns true. This running a preliminary pass for identifying the token trajectories  consists in
method creates new arcs and nodes between nodes upnode gurations from which we can reach the terminal congu-                con-
and downnode in the output graph ngraph, and computes ration (i.e. both tokens on the end nodes). This is easily
the information attached to these newly created arcs as a achieved by enumerating the trajectories in the opposite
function of the attached information of the input arcs arc1 direction. We start on the end nodes and follow the arcs
and arc2.                                                      upstream. During the main pass, we only build the nodes
   bprop(ngraph, upnode, downnode, arc1, arc2)
is called during training in order to propagate gradient in- that allow the tokens to reach the terminal conguration.
formation from the output sub-graph between upnode and            Graph composition using transducers (i.e. standard
downnode into the data structures on the arc1 and arc2, transduction) is easily and eciently implemented as a gen-
as well as with respect to the parameters that were used in eralized transduction. The method check simply tests the
the fprop call with the same arguments. This assumes that equality of the input symbols on the two arcs, and the
the function used by fprop to compute the values attached method fprop creates a single arc whose symbol is the
to its output arcs is di erentiable.                           output symbol on the transducer's arc.
   The check method can be seen as constructing a dy-             The composition between pairs of graphs is particularly
namic architecture of functional dependencies, while the useful for incorporating linguistic constraints in a hand-
fprop method performs a forward propagation through writing recognizer. Examples of its use are given in the
that architecture to compute the numerical information at- on-line handwriting recognition system described in Sec-
tached to the arcs. The bprop method performs a back- tion IX) and in the check reading system described in Sec-
ward propagation through the same architecture to com- tion X).
pute the partial derivatives of the loss function with respect    In the rest of the paper, the term Composition Trans-
to the information attached to the arcs. This is illustrated former will denote a Graph Transformer based on the gen-
in Figure 28.                                                  eralized transductions of multiple graphs. The concept of
   Figure 29 shows a simplied generalized graph composi- generalized transduction is a very general one. In fact,
tion algorithm. This simplied algorithm does not handle many of the graph transformers described earlier in this
null transitions, and does not check whether the tokens tra- paper, such as the segmenter and the recognizer, can be

<!-- page 33 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                              33

                                                                 formulated in terms of generalized transduction. In this
                                                                 case the, the generalized transduction does not take two in-
                                                                 put graphs but a single input graph. The method fprop of
                                                                 the transformer may create several arcs or even a complete
Function generalized_composition(PGRAPH graph1,                  subgraph for each arc of the initial graph. In fact the pair
                                 PGRAPH graph2,                  (check, fprop) itself can be seen as procedurally dening
                                 PTRANS trans)                   a transducer.
Returns PGRAPH                                                     In addition, It can be shown that the generalized trans-
{                                                                duction of a single graph is theoretically equivalent to the
  // Create new graph                                            standard composition of this graph with a particular trans-
  PGRAPH ngraph = new_graph()                                    ducer graph. However, implementing the operation this
                                                                 way may be very inecient since the transducer can be
  // Create map between token positions                          very complicated.
  // and nodes of the new graph                                    In practice, the graph produced by a generalized trans-
  PNODE mapPNODE,PNODE] = new_empty_map()                       duction is represented procedurally, in order to avoid build-
  mapendnode(graph1), endnode(graph2)] =                        ing the whole output graph (which may be huge when for
    endnode(newgraph)                                            example the interpretation graph is composed with the
                                                                 grammar graph). We only instantiate the nodes which
  // Recursive subroutine for simulating tokens                  are visited by the search algorithm during recognition (e.g.
  Function simtokens(PNODE node1, PNODE node2)                   Viterbi). This strategy propagates the benets of pruning
  Returns PNODE                                                  algorithms (e.g. Beam Search) in all the Graph Transformer
  {                                                              Network.
    PNODE currentnode = mapnode1, node2]
    // Check if already visited
    If (currentnode == nil)                                      D. Notes on the Graph Structures
      // Record new configuration
      currentnode = ngraph->create_node()                               Section VI has discussed the idea of global training
      mapnode1, node2] = currentnode                                by back-propagating gradient through simple graph trans-
      // Enumerate the possible non-null                             formers. The bprop method is the basis of the back-
      // joint token transitions                                     propagation algorithm for generic graph transformers. A
      For ARC arc1 in down_arcs(node1)                               generalized composition transformer can be seen as dynam-
        For ARC arc2 in down_arcs(node2)                             ically establishing functional relationships between the nu-
          If (trans->check(arc1, arc2))                              merical quantities on the input and output arcs. Once the
            PNODE newnode =                                          check function has decided that a relationship should be es-
             simtokens(down_node(arc1),                              tablished, the fprop function implements the numerical re-
                       down_node(arc2))                              lationship. The check function establishes the structure of
            trans->fprop(ngraph, currentnode,                        the ephemeral network inside the composition transformer.
                         newnode, arc1, arc2)                           Since fprop is assumed to be di erentiable, gradients can
    // Return node in composed graph                                 be back-propagated through that structure. Most param-
    Return currentnode                                               eters a ect the scores stored on the arcs of the successive
  }                                                                  graphs of the system. A few threshold parameters may de-
                                                                     termine whether an arc appears or not in the graph. Since
   // Perform token simulation                                       non existing arcs are equivalent to arcs with very large
   simtokens(startnode(graph1), startnode(graph2)) penalties, we only consider the case of parameters a ect-
   Delete map                                                        ing the penalties.
   Return ngraph                                                        In the kind of systems we have discussed until now (and
}                                                                    the application described in Section X), much of the knowl-
                                                                     edge about the structure of the graph that is produced by
Fig. 29. Pseudo-code for a simplied generalized composition algo- a Graph Transformer is determined by the nature of the
     rithm. For simplifying the presentation, we do not handle null Graph Transformer, but it may also depend on the value
     transitions nor implement dead end avoidance. The two main
     component of the composition appear clearly here: (a) the re- of the parameters and on the input. It may also be interest-
     cursive function simtoken() enumerating the token trajectories, ing to consider Graph Transformer modules which attempt
     and, (b) the associative array map used for remembering which to learn the structure of the output graph. This might
     nodes of the composed graph have been visited.
                                                                     be considered a combinatorial problem and not amenable
                                                                     to Gradient-Based Learning, but a solution to this prob-
                                                                     lem is to generate a large graph that contains the graph
                                                                     candidates as sub-graphs, and then select the appropriate
                                                                     sub-graph.

<!-- page 34 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                           34

E. GTN and Hidden Markov Models                                 arcs are simply added in order to obtain the complete out-
   GTNs can be seen as a generalization and an extension of     put graph. The input values of the emission and transition
HMMs. On the one hand, the probabilistic interpretation         modules are read o the data structure on the input arcs
can be either kept (with penalties being log-probabilities),    of the IOHMM Graph Transformer. In practice, the out-
pushed to the nal decision stage (with the di erence of the    put graph may be very large, and needs not be completely
constrained forward penalty and the unconstrained forward       instantiated (i.e., it is pruned: only the low penalty paths
penalty being interpreted as negative log-probabilities of      are created).
label sequences), or dropped altogether (the network just
represents a decision surface for label sequences in input      IX. An On-Line Handwriting Recognition System
space). On the other hand, Graph Transformer Networks              Natural handwriting is often a mixture of di erent
extend HMMs by allowing to combine in a well-principled         \styles", lower case printed, upper case, and cursive. A
framework multiple levels of processing, or multiple mod-       reliable recognizer for such handwriting would greatly im-
els (e.g., Pereira et al. have been using the transducer        prove interaction with pen-based devices, but its imple-
framework for stacking HMMs representing di erent levels        mentation presents new technical challenges. Characters
of processing in automatic speech recognition 86]).             taken in isolation can be very ambiguous, but consider-
   Unfolding a HMM in time yields a graph that is very sim-     able information is available from the context of the whole
ilar to our interpretation graph (at the nal stage of pro-     word. We have built a word recognition system for pen-
cessing of the Graph Transformer Network, before Viterbi        based devices based on four main modules: a preprocessor
recognition). It has nodes n(t i) associated to each time      that normalizes a word, or word group, by tting a geomet-
step t and state i in the model. The penalty ci for an arc      rical model to the word structure$ a module that produces
from n(t ; 1 j ) to n(t i) then corresponds to the nega-      an \annotated image" from the normalized pen trajectory$
tive log-probability of emitting observed data ot at posi-      a replicated convolutional neural network that spots and
tion t and going from state j to state i in the time interval   recognizes characters$ and a GTN that interprets the net-
(t ; 1 t). With this probabilistic interpretation, the for-    works output by taking word-level constraints into account.
ward penalty is the negative logarithm of the likelihood of     The network and the GTN are jointly trained to minimize
whole observed data sequence (given the model).                 an error measure dened at the word level.
   In Section VI we mentioned that the collapsing phe-             In this work, we have compared a system based on
nomenon can occur when non-discriminative loss functions        SDNNs (such as described in Section VII), and a system
are used to train neural networks/HMM hybrid systems.           based on Heuristic Over-Segmentation (such as described
With classical HMMs with xed preprocessing, this prob-         in Section V). Because of the sequential nature of the infor-
lem does not occur because the parameters of the emission       mation in the pen trajectory (which reveals more informa-
and transition probability models are forced to satisfy cer-    tion than the purely optical input from in image), Heuristic
tain probabilistic constraints: the sum or the integral of      Over-Segmentation can be very ecient in proposing can-
the probabilities of a random variable over its possible val-   didate character cuts, especially for non-cursive script.
ues must be 1. Therefore, when the probability of certain
events is increased, the probability of other events must au-   A. Preprocessing
tomatically be decreased. On the other hand, if the prob-
abilistic assumptions in an HMM (or other probabilistic           Input normalization reduces intra-character variability,
model) are not realistic, discriminative training, discussed    simplifying character recognition. We have used a word
in Section VI, can improve performance as this has been         normalization scheme 92] based on tting a geometrical
clearly shown for speech recognition systems 48], 49], 50],     model of the word structure. Our model has four \exi-
 107], 108].                                                    ble" lines representing respectively the ascenders line, the
   The Input-Output HMM model (IOHMM) 105], 109],               core line, the base line and the descenders line. The lines
is strongly related to graph transformers. Viewed as a          are tted to local minima or maxima of the pen trajectory.
probabilistic model, an IOHMM represents the conditional        The parameters of the lines are estimated with a modied
distribution of output sequences given input sequences (of      version of the EM algorithm to maximize the joint prob-
the same or a di erent length). It is parameterized from        ability of observed points and parameter values, using a
an emission probability module and a transition probabil-       prior on parameters that prevents the lines from collapsing
ity module. The emission probability module computes            on each other.
the conditional emission probability of an output variable        The recognition of handwritten characters from a pen
(given an input value and the value of discrete \state" vari-   trajectory on a digitizing surface is often done in the time
able). The transition probability module computes condi-        domain 110], 44], 111]. Typically, trajectories are nor-
tional transition probabilities of a change in the value of     malized, and local geometrical or dynamical features are
the \state" variable, given the an input value. Viewed as a     extracted. The recognition may then be performed us-
graph transformer, it assigns an output graph (representing     ing curve matching 110], or other classication techniques
a probability distribution over the sequences of the output     such as TDNNs 44], 111]. While these representations
variable) to each path in the input graph. All these output     have several advantages, their dependence on stroke order-
graphs have the same structure, and the penalties on their      ing and individual writing styles makes them dicult to

<!-- page 35 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                35

                                     "Script"                                                             "Script"

        Viterbi Graph                                                      Viterbi Graph

                                 Beam Search                                                           Beam Search
                                 Transformer                                                           Transformer

        Interpretation Graph                                               Interpretation Graph

        Language                                                           Language
        Model                      Compose                                                              Compose
                                                                           Model

        Recognition Graph                                                  Recognition Graph

                                  Recognition                              Character
                                                                           Model                        Compose
                                  Transformer
                                                                           SDNN Output
        AMAP Graph
                                                                                                       SDNN
                               AMAP Computation                                                        Transformer

        Segmentation Graph
                                                                           AMAP

                                 Segmentation
                                 Transformer                                                       AMAP Computation
        Normalized Word
                                                                           Normalized Word

                               Word Normalization                                                  Word Normalization

Fig. 30. An on-line handwriting recognition GTN based on heuristic Fig. 31. An on-line handwriting recognition GTN based on Space-
     over-segmentation                                                  Displacement Neural Network

                                                                  B. Network Architecture
                                                                     One of the best networks we found for both online and
use in high accuracy, writer independent systems that in-         o%ine character recognition is a 5-layer convolutional net-
tegrate the segmentation with the recognition.                    work somewhat similar to LeNet-5 (Figure 2), but with
                                                                  multiple input planes and di erent numbers of units on the
   Since the intent of the writer is to produce a legible im-     last two layers$ layer 1: convolution with 8 kernels of size
age, it seems natural to preserve as much of the pictorial        3x3, layer 2: 2x2 sub-sampling, layer 3: convolution with
nature of the signal as possible, while at the same time ex-      25 kernels of size 5x5, layer 4 convolution with 84 kernels
ploit the sequential information in the trajectory. For this      of size 4x4, layer 5: 2x1 sub-sampling, classication layer:
purpose we have designed a representation scheme, called          95 RBF units (one per class in the full printable ASCII
AMAP 38], where pen trajectories are represented by low-          set). The distributed codes on the output are the same as
resolution images in which each picture element contains          for LeNet-5, except they are adaptive unlike with LeNet-5.
information about the local properties of the trajectory. An      When used in the heuristic over-segmentation system, the
AMAP can be viewed as an \annotated image" in which               input to above network consisted of an AMAP with ve
each pixel is a 5-element feature vector: 4 features are as-      planes, 20 rows and 18 columns. It was determined that
sociated to four orientations of the pen trajectory in the        this resolution was sucient for representing handwritten
area around the pixel, and the fth one is associated to          characters. In the SDNN version, the number of columns
local curvature in the area around the pixel. A particu-          was varied according to the width of the input word. Once
larly useful feature of the AMAP representation is that it        the number of sub-sampling layers and the sizes of the ker-
makes very few assumptions about the nature of the input          nels are chosen, the sizes of all the layers, including the
trajectory. It does not depend on stroke ordering or writ-        input, are determined unambiguously. The only architec-
ing speed, and it can be used with all types of handwriting       tural parameters that remain to be selected are the num-
(capital, lower case, cursive, punctuation, symbols). Un-         ber of feature maps in each layer, and the information as
like many other representations (such as global features),        to what feature map is connected to what other feature
AMAPs can be computed for complete words without re-              map. In our case, the sub-sampling rates were chosen as
quiring segmentation.                                             small as possible (2x2), and the kernels as small as pos-

<!-- page 36 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                            36

sible in the rst layer (3x3) to limit the total number of      In this application, the language model simply constrains
connections. Kernel sizes in the upper layers are chosen to     the nal output graph to represent sequences of character
be as small as possible while satisfying the size constraints   labels from a given dictionary. Furthermore, the interpre-
mentioned above. Larger architectures did not necessarily       tation graph is not actually completely instantiated: the
perform better and required considerably more time to be        only nodes created are those that are needed by the Beam
trained. A very small architecture with half the input eld     Search module. The interpretation graph is therefore rep-
also performed worse, because of insucient input resolu-       resented procedurally rather than explicitly.
tion. Note that the input resolution is nonetheless much           A crucial contribution of this research was the joint train-
less than for optical character recognition, because the an-    ing of all graph transformer modules within the network
gle and curvature provide more information than would a         with respect to a single criterion, as explained in Sec-
single grey level at each pixel.                                tions VI and VIII. We used the Discriminative Forward loss
                                                                function on the nal output graph: minimize the forward
C. Network Training                                             penalty of the constrained interpretation (i.e., along all the
   Training proceeded in two phases. First, we kept the         \correct" paths) while maximizing the forward penalty of
centers of the RBFs xed, and trained the network weights       the whole interpretation graph (i.e., along all the paths).
so as to minimize the output distance of the RBF unit              During global training, the loss function was optimized
corresponding to the correct class. This is equivalent to       with the stochastic diagonal Levenberg-Marquardt proce-
minimizing the mean-squared error between the previous          dure described in Appendix C, that uses second derivatives
layer and the center of the correct-class RBF. This boot-       to compute optimal learning rates. This optimization op-
strap phase was performed on isolated characters. In the        erates on all the parameters in the system, most notably
second phase, all the parameters, network weights and RBF       the network weights and the RBF centers.
centers were trained globally to minimize a discriminative
criterion at the word level.                                    D. Experimental Results
   With the Heuristic Over-Segmentation approach, the              In the rst set of experiments, we evaluated the general-
GTN was composed of four main Graph Transformers:               ization ability of the neural network classier coupled with
1. The Segmentation Transformer performs the                    the word normalization preprocessing and AMAP input
Heuristic Over-Segmentation, and outputs the segmenta-          representation. All results are in writer independent mode
tion graph. An AMAP is then computed for each image             (di erent writers in training and testing). Initial train-
attached to the arcs of this graph.                             ing on isolated characters was performed on a database of
2. The Character Recognition Transformer applies                approximately 100,000 hand printed characters (95 classes
the the convolutional network character recognizer to each      of upper case, lower case, digits, and punctuation). Tests
candidate segment, and outputs the recognition graph,           on a database of isolated characters were performed sepa-
with penalties and classes on each arc.                         rately on the four types of characters: upper case (2.99%
3. The Composition Transformer composes the recog-              error on 9122 patterns), lower case (4.15% error on 8201
nition graph with a grammar graph representing a language       patterns), digits (1.4% error on 2938 patterns), and punc-
model incorporating lexical constraints.                        tuation (4.3% error on 881 patterns). Experiments were
4. The Beam Search Transformer extracts a good inter-           performed with the network architecture described above.
pretation from the interpretation graph. This task could        To enhance the robustness of the recognizer to variations
have been achieved with the usual Viterbi Transformer.          in position, size, orientation, and other distortions, addi-
The Beam Search algorithm however implements pruning            tional training data was generated by applying local ane
strategies which are appropriate for large interpretation       transformations to the original characters.
graphs.                                                            The second and third set of experiments concerned the
   With the SDNN approach, the main Graph Transformers          recognition of lower case words (writer independent). The
are the following:                                              tests were performed on a database of 881 words. First
1. The SDNN Transformer replicates the convolutional            we evaluated the improvements brought by the word nor-
network over the a whole word image, and outputs a recog-       malization to the system. For the SDNN/HMM system
nition graph that is a linear graph with class penalties for    we have to use word-level normalization since the net-
every window centered at regular intervals on the input         work sees one whole word at a time. With the Heuris-
image.                                                          tic Over-Segmentation system, and before doing any word-
2. The Character-Level Composition Transformer                  level training, we obtained with character-level normaliza-
composes the recognition graph with a left-to-right HMM         tion 7.3% and 3.5% word and character errors (adding in-
for each character class (as in Figure 27).                     sertions, deletions and substitutions) when the search was
3. The Word-Level Composition Transformer com-                  constrained within a 25461-word dictionary. When using
poses the output of the previous transformer with a lan-        the word normalization preprocessing instead of a charac-
guage model incorporating lexical constraints, and outputs      ter level normalization, error rates dropped to 4.6% and
the interpretation graph.                                       2.0% for word and character errors respectively, i.e., a rel-
4. The Beam Search Transformer extracts a good in-              ative drop of 37% and 43% in word and character error
terpretation from the interpretation graph.                     respectively. This suggests that normalizing the word in

<!-- page 37 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                             37

its entirety is better than rst segmenting it and then nor- the system as 50% correct / 49% reject / 1% error. The
malizing and processing each of the segments.                     system presented here was one of the rst to cross that
                                                                  threshold on representative mixtures of business and per-
         SDNN/HMM
         no global training
                                   No Language Model
                                                                  sonal checks.
                                                                             12.4
        with global training                                     8.2
                                                                     Checks contain at least two versions of the amount. The
               HOS                 No Language Model              Courtesy amount is written with numerals, while the Legal
         no global training
        with global training                               6.3    amount is written with letters. On business checks, which
                                                                  8.5

                                                                  are generally machine-printed, these amounts are relatively
                                                                  easy to read, but quite dicult to nd due to the lack of
               HOS                 25K Word Lexicon
         no global training                   2
        with global training            1.4
                                                                  standard for business check layout. On the other hand,
                                                                  these amounts on personal checks are easy to nd but much
                               0                       5                10          15

Fig. 32. Comparative results (character error rates) showing the harder to read.
     improvement brought by global training on the SDNN/HMM          For simplicity (and speed requirements), our initial task
     hybrid, and on the Heuristic Over-Segmentation system (HOS),
     without and with a 25461 words dictionary.                   is to read the Courtesy amount only. This task consists of
                                                                  two main steps:
   In the third set of experiments, we measured the im-              The system has to nd, among all the elds (lines of
provements obtained with the joint training of the neural text), the candidates that are the most likely to contain the
network and the post-processor with the word-level crite- courtesy amount. This is obvious for many personal checks,
rion, in comparison to training based only on the errors where the position of the amount is standardized. However,
performed at the character level. After initial training on as already noted, nding the amount can be rather dicult
individual characters as above, global word-level discrim- in business checks, even for the human eye. There are
inative training was performed with a database of 3500 many strings of digits, such as the check number, the date,
lower case words. For the SDNN/HMM system, without or even \not to exceed" amounts, that can be confused
any dictionary constraints, the error rates dropped from with the actual amount. In many cases, it is very dicult
38% and 12.4% word and character error to 26% and 8.2% to decide which candidate is the courtesy amount before
respectively after word-level training, i.e., a relative drop performing a full recognition.
of 32% and 34%. For the Heuristic Over-Segmentation sys- In order to read (and choose) some Courtesy amount
tem and a slightly improved architecture, without any dic- candidates, the system has to segment the elds into char-
tionary constraints, the error rates dropped from 22.5% acters, read and score the candidate characters, and nally
and 8.5% word and character error to 17% and 6.3% re- nd the best interpretation of the amount using contextual
spectively, i.e., a relative drop of 24.4% and 25.6%. With a knowledge represented by a stochastic grammar for check
25461-word dictionary, errors dropped from 4.6% and 2.0% amounts.
word and character errors to 3.2% and 1.4% respectively              The GTN methodology was used to build a check amount
after word-level training, i.e., a relative drop of 30.4% and reading system that handles both personal checks and busi-
30.0%. Even lower error rates can be obtained by dras- ness checks.
tically reducing the size of the dictionary to 350 words,
yielding 1.6% and 0.94% word and character errors.                A. A GTN for Check Amount Recognition
   These results clearly demonstrate the usefulness of glob-
ally trained Neural-Net/HMM hybrids for handwriting                  We now describe the successive graph transformations
recognition. This conrms similar results obtained earlier        that  allow this network to read the check amount (cf. Fig-
in speech recognition 77].                                        ure  33). Each Graph Transformer produces a graph whose
                                                                  paths encode and score the current hypotheses considered
               X. A Check Reading System                          at this stage of the system.
   This section describes a GTN based Check Reading Sys-             The input to the system is a trivial graph with a single
tem, intended for immediate industrial deployment. It also        arc that carries the image of the whole check (cf. Figure 33).
shows how the use of Gradient Based-Learning and GTNs                The eld location transformer Tfield rst performs
make this deployment fast and cost-e ective while yielding classical image analysis (including connected component
an accurate and reliable solution.                                analysis, ink density histograms, layout analysis, etc...)
   The verication of the amount on a check is a task that and heuristically extracts rectangular zones that may con-
is extremely time and money consuming for banks. As a tain the check amount. Tfield produces an output graph,
consequence, there is a very high interest in automating the called the eld graph (cf. Figure 33) such that each can-
process as much as possible (see for example 112], 113], didate zone is associated with one arc that links the start
 114]). Even a partial automation would result in consid- node to the end node. Each arc contains the image of the
erable cost reductions. The threshold of economic viability zone, and a penalty term computed from simple features
for automatic check readers, as set by the bank, is when extracted from the zone (absolute position, size, aspect ra-
50% of the checks are read with less than 1% error. The tio, etc...). The penalty term is close to zero if the features
other 50% of the check being rejected and sent to human suggest that the eld is a likely candidate, and is large if
operators. In such a case, we describe the performance of the eld is deemed less likely to be an amount. The penalty

<!-- page 38 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                   38

                                 Viterbi Answer                                           The segmenter uses a variety of heuristics to nd candi-
                                                                                       date cut. One of the most important ones is called \hit and
        Best Amount Graph                                                              deect" 115]. The idea is to cast lines downward from the
                                                                                       top of the eld image. When a line hits a black pixel, it is
                               Viterbi Transformer                                     deected so as to follow the contour of the object. When a
                                                      "$" 0.2                          line hits a local minimum of the upper prole, i.e. when it
        Interpretation Graph                          "*" 0.4
                                                      "3" 0.1                          cannot continue downward without crossing a black pixel,
                                                                                       it is just propagated vertically downward through the ink.
                                                      .......

        Grammar                     Compose                                            When two such lines meet each other, they are merged into
                                                                                       a single cut. The procedure can be repeated from the bot-
                                                                                       tom up. This strategy allows the separation of touching
                                                          "$" 0.2
                                                          "*" 0.4
        Recognition Graph                                 "3" 0.1
                                                          "B" 23.6
                                                          .......                      characters such as double zeros.
                                   Recognition                                            The recognition transformer Trec iterates over all
                                   Transformer
                                                                                       segment arcs in the segmentation graph and runs a charac-
                                                    $          *         3             ter recognizer on the corresponding segment image. In our
        Segmentation Graph
                                                          **             45            case, the recognizer is LeNet-5, the Convolutional Neural
                             Segmentation Transf.                                      Network described in Section II, whose weights constitute
                                                                                       the largest and most important subset of tunable parame-
        Field Graph              45/xx
                                                  $ *** 3.45
                                                                                       ters. The recognizer classies segment images into one of
                                                                                       95 classes (full printable ASCII set) plus a rubbish class for
                                                   $10,000.00

                             Field Location Transf.                                    unknown symbols or badly-formed characters. Each arc in
        Check Graph
                                          2nd Nat. Bank
                                          not to exceed $10,000.00        $ *** 3.45
                                                                                       the input graph Trec is replaced by 96 arcs in the output
                                                                                       graph. Each of those 96 arcs contains the label of one of
                                               three dollars and 45/xx

Fig. 33. A complete check amount reader implemented as a single                        the classes, and a penalty that is the sum of the penalty
     cascade of Graph Transformer modules. Successive graph trans-                     of the corresponding arc in the input (segmentation) graph
     formations progressively extract higher level information.                        and the penalty associated with classifying the image in
                                                                                       the corresponding class, as computed by the recognizer. In
                                                                                       other words, the recognition graph represents a weighted
function is di erentiable, therefore its parameter are glob-                           trellis of scored character classes. Each path in this graph
ally tunable.                                                                          represents a possible character string for the correspond-
   An arc may represent separate dollar and cent amounts                               ing eld. We can compute a penalty for this interpretation
as a sequence of elds. In fact, in handwritten checks, the                            by adding the penalties along the path. This sequence of
cent amount may be written over a fractional bar, and not                              characters may or may not be a valid check amount.
aligned at all with the dollar amount. In the worst case,                                 The composition transformer Tgram selects the
one may nd several cent amount candidates (above and                                  paths of the recognition graph that represent valid char-
below the fraction bar) for the same dollar amount.                                    acter sequences for check amounts. This transformer takes
   The segmentation transformer Tseg , similar to the                                  two graphs as input: the recognition graph, and the gram-
one described in Section VIII examines each zone contained                             mar graph. The grammar graph contains all possible se-
in the eld graph, and cuts each image into pieces of ink                              quences of symbols that constitute a well-formed amount.
using heuristic image processing techniques. Each piece                                The output of the composition transformer, called the in-
of ink may be a whole character or a piece of character.                               terpretation graph, contains all the paths in the recognition
Each arc in the eld graph is replaced by its correspond-                              graph that are compatible with the grammar. The oper-
ing segmentation graph that represents all possible group-                             ation that combines the two input graphs to produce the
ings of pieces of ink. Each eld segmentation graph is ap-                             output is a generalized transduction (see Section VIII).A
pended to an arc that contains the penalty of the eld in                              di erentiable function is used to compute the data attached
the eld graph. Each arc carries the segment image, to-                                to the output arc from the data attached to the input arcs.
gether with a penalty that provides a rst evaluation of                               In our case, the output arc receives the class label of the
the likelihood that the segment actually contains a charac-                            two arcs, and a penalty computed by simply summing the
ter. This penalty is obtained with a di erentiable function                            penalties of the two input arcs (the recognizer penalty, and
that combines a few simple features such as the space be-                              the arc penalty in the grammar graph). Each path in the
tween the pieces of ink or the compliance of the segment                               interpretation graph represents one interpretation of one
image with a global baseline, and a few tunable parame-                                segmentation of one eld on the check. The sum of the
ters. The segmentation graph represents all the possible                               penalties along the path represents the \badness" of the
segmentations of all the eld images. We can compute the                               corresponding interpretation and combines evidence from
penalty for one segmented eld by adding the arc penalties                             each of the modules along the process, as well as from the
along the corresponding path. As before using a di eren-                               grammar.
tiable function for computing the penalties will ensure that                              The Viterbi transformer nally selects the path with
the parameters can be optimized globally.                                              the lowest accumulated penalty, corresponding to the best

<!-- page 39 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                               39

                                         Edforw                     as described in Figure 21, using as our desired sequence the
                                                                    Viterbi answer. This is summarized in Figure 34, with:
                                         +
                             Cdforw           −
                                                                                     condence = exp(E   dforw   )
                                                     Cforw
                         Forward Scorer                             D. Results
                                                  Forward Scorer       A version of the above system was fully implemented
                                                                    and tested on machine-print business checks. This sys-
                                                                    tem is basically a generic GTN engine with task specic
        Viterbi
        Answer               Path Selector

                  Interpretation Graph                              heuristics encapsulated in the check and fprop method.
                                                                    As a consequence, the amount of code to write was min-
Fig. 34. Additional processing required to compute the condence.   imal: mostly the adaptation of an earlier segmenter into
                                                                    the segmentation transformer. The system that deals with
                                                                    hand-written or personal checks was based on earlier im-
grammatically correct interpretations.                              plementations that used the GTN concept in a restricted
                                                                    way.
B. Gradient-Based Learning                                             The neural network classier was initially trained on
   Each stage of this check reading system contains tun-            500,000 images of character images from various origins
able parameters. While some of these parameters could be            spanning the entire printable ASCII set. This contained
manually adjusted, for example the parameters of the eld           both handwritten and machine-printed characters that had
locator and segmenter, the vast majority of them must be            been previously size normalized at the string level. Addi-
learned, particularly the weights of the neural net recog-          tional images were generated by randomly distorting the
nizer.                                                              original images using simple ane transformations of the
   Prior to globally optimizing the system, each module pa-         images. The network was then further trained on character
rameters are initialized with reasonable values. The param-         images that had been automatically segmented from check
eters of the eld locator and the segmenter are initialized         images and manually truthed. The network was also ini-
by hand, while the parameters of the neural net charac-             tially trained to reject non-characters that resulted from
ter recognizer are initialized by training on a database of         segmentation errors. The recognizer was then inserted in
pre-segmented and labeled characters. Then, the entire              the check reading system and a small subset of the parame-
system is trained globally from whole check images labeled          ters were trained globally (at the eld level) on whole check
with the correct amount. No explicit segmentation of the            images.
amounts is needed to train the system: it is trained at the            On 646 business checks that were automatically catego-
check level.                                                        rized as machine printed the performance was 82% cor-
   The loss function E minimized by our global train-               rectly recognized checks, 1% errors, and 17% rejects. This
ing procedure is the Discriminative Forward criterion de-           can be compared to the performance of the previous sys-
scribed in Section VI: the di erence between (a) the for-           tem on the same test set: 68% correct, 1% errors, and
ward penalty of the constrained interpretation graph (con-          31% rejects. A check is categorized as machine-printed
strained by the correct label sequence), and (b) the forward        when characters that are near a standard position Dollar
penalty of the unconstrained interpretation graph. Deriva-          sign are detected as machine printed, or when, if nothing
tives can be back-propagated through the entire structure,          is found in the standard position, at least one courtesy
although it only practical to do it down to the segmenter.          amount candidate is found somewhere else. The improve-
                                                                    ment is attributed to three main causes. First the neural
C. Rejecting Low Condence Checks                                   network recognizer was bigger, and trained on more data.
   In order to be able to reject checks which are the most          Second, because of the GTN architecture, the new system
likely to carry erroneous Viterbi answers, we must rate             could take advantage of grammatical constraints in a much
them with a condence, and reject the check if this con-            more ecient way than the previous system. Third, the
dence is below a given threshold. To compare the un-               GTN architecture provided extreme exibility for testing
normalized Viterbi Penalties of two di erent checks would           heuristics, adjusting parameters, and tuning the system.
be meaningless when it comes to decide which answer we              This last point is more important than it seems. The GTN
trust the most.                                                     framework separates the \algorithmic" part of the system
   The optimal measure of condence is the probability of           from the \knowledge-based" part of the system, allowing
the Viterbi answer given the input image. As seen in Sec-           easy adjustments of the latter. The importance of global
tion VI-E, given a target sequence (which, in this case,            training was only minor in this task because the global
would be the Viterbi answer), the discriminative forward            training only concerned a small subset of the parameters.
loss function is an estimate of the logarithm of this prob-            An independent test performed by systems integrators
ability. Therefore, a simple solution to obtain a good esti-        in 1995 showed the superiority of this system over other
mate of the condence is to reuse the interpretation graph          commercial Courtesy amount reading systems. The system
(see Figure 33) to compute the discriminative forward loss          was integrated in NCR's line of check reading systems. It

<!-- page 40 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                          40

has been elded in several banks across the US since June Neural Networks allows to learn appropriate features from
1996, and has been reading millions of checks per day since examples. The success of this approach was demonstrated
then.                                                        in extensive comparative digit recognition experiments on
                                                             the NIST database.
                     XI. Conclusions                         2. Segmentation and recognition of objects in images can-
   During the short history of automatic pattern recogni- not be completely decoupled. Instead of taking hard seg-
tion, increasing the role of learning seems to have invari- mentation decisions too early, we have used Heuristic Over-
ably improved the overall performance of recognition sys- Segmentation to generate and evaluate a large number of
tems. The systems described in this paper are more ev- hypotheses in parallel, postponing any decision until the
idence to this fact. Convolutional Neural Networks have overall criterion is minimized.
been shown to eliminate the need for hand-crafted fea- 3. Hand truthing images to obtain segmented characters
ture extractors. Graph Transformer Networks have been for training a character recognizer is expensive and does
shown to reduce the need for hand-crafted heuristics, man- not take into account the way in which a whole document
ual labeling, and manual parameter tuning in document or sequence of characters will be recognized (in particular
recognition systems. As training data becomes plentiful, as the fact that some segmentation candidates may be wrong,
computers get faster, as our understanding of learning al- even though they may look like true characters). Instead
gorithms improves, recognition systems will rely more and we train multi-module systems to optimize a global mea-
more of learning, and their performance will improve.        sure of performance, which does not require time consum-
   Just as the back-propagation algorithm elegantly solved ing detailed hand-truthing, and yields signicantly better
the credit assignment problem in multi-layer neural net- recognition performance, because it allows to train these
works, the gradient-based learning procedure for Graph modules to cooperate towards a common goal.
Transformer Networks introduced in this paper solves the 4. Ambiguities inherent in the segmentation, character
credit assignment problem in systems whose functional ar- recognition, and linguistic model should be integrated op-
chitecture dynamically changes with each new input. The timally. Instead of using a sequence of task-dependent
learning algorithms presented here are in a sense nothing heuristics to combine these sources of information, we
more than unusual forms of gradient descent in complex, have proposed a unied framework in which generalized
dynamic architectures, with ecient back-propagation al- transduction methods are applied to graphs representing a
gorithms to compute the gradient. The results in this pa- weighted set of hypotheses about the input. The success of
per help establish the usefulness and relevance of gradient- this approach was demonstrated with a commercially de-
based minimization methods as a general organizing prin- ployed check reading system that reads millions of business
ciple for learning in large systems.                         and personal checks per day: the generalized transduction
   It was shown that all the steps of a document analysis engine resides in only a few hundred lines of code.
system can be formulated as graph transformers through 5. Traditional recognition systems rely on many hand-
which gradients can be back-propagated. Even in the crafted heuristics to isolate individually recognizable ob-
non-trainable parts of the system, the design philosophy jects. The promising Space Displacement Neural Network
in terms of graph transformation provides a clear separa- approach draws on the robustness and eciency of Con-
tion between domain-specic heuristics (e.g. segmentation volutional Neural Networks to avoid explicit segmentation
heuristics) and generic, procedural knowledge (the gener- altogether. Simultaneous automatic learning of segmenta-
alized transduction algorithm)                               tion and recognition can be achieved with Gradient-Based
   It is worth pointing out that data generating models Learning methods.
(such as HMMs) and the Maximum Likelihood Principle             This paper presents a small number of examples of graph
were not called upon to justify most of the architectures    transformer    modules, but it is clear that the concept can be
and the training criteria described in this paper. Gradient  applied  to  many   situations where the domain knowledge or
based learning applied to global discriminative loss func-   the state  information   can be represented by graphs. This is
tions guarantees optimal classication and rejection with-   the case  in  many   audio signal recognition tasks, and visual
out the use of \hard to justify" principles that put strong  scene  analysis   applications.  Future work will attempt to
constraints on the system architecture, often at the expense apply  Graph    Transformer   Networks   to such problems, with
of performances.                                             the hope    of allowing  more  reliance on  automatic learning,
   More specically, the methods and architectures pre-      and  less on   detailed engineering.
sented in this paper o er generic solutions to a large num-                            Appendices
ber of problems encountered in pattern recognition sys-
tems:                                                        A. Pre-conditions for faster convergence
1. Feature extraction is traditionally a xed transform,        As seen before, the squashing function used in our Con-
generally derived from some expert prior knowledge about volutional Networks is f (a) = A tanh(Sa). Symmetric
the task. This relies on the probably incorrect assumption functions are believed to yield faster convergence, although
that the human designer is able to capture all the rele- the learning can become extremely slow if the weights are
vant information in the input. We have shown that the too small. The cause of this problem is that in weight space
application of Gradient-Based Learning to Convolutional the origin is a xed point of the learning dynamics, and,

<!-- page 41 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                        41

although it is a saddle point, it is attractive in almost allperforming two complete learning iterations over the small
directions 116]. For our simulations, we use A = 1:7159      subset. This idea can be generalized to training sets where
and S = (see 20], 34]). With this choice of parame-
                                                             there exist no precise repetition of the same pattern but
ters, the equalities f (1) = 1 and f (;1) = ;1 are satised. where some redundancy is present. In fact stochastic up-

The rationale behind this is that the overall gain of the    date must be better when there is redundancy, i.e., when a
squashing transformation is around 1 in normal operat-       certain level of generalization is expected.
ing conditions, and the interpretation of the state of the      Many authors have claimed that second-order meth-
network is simplied. Moreover, the absolute value of the    ods should be used in lieu of gradient descent for neu-
second derivative of f is a maximum at +1 and ;1, which      ral net training. The literature abounds with recom-
improves the convergence towards the end of the learning     mendations 118] for classical second-order methods such
session. This particular choice of parameters is merely a    as the Gauss-Newton or Levenberg-Marquardt algorithms,
convenience, and does not a ect the result.                  for Quasi-Newton methods such as the Broyden-Fletcher-
   Before training, the weights are initialized with random  Goldfarb-Shanno method (BFGS), Limited-storage BFGS,
values using a uniform distribution between ;2:4=Fi and      or for various versions of the Conjugate Gradients (CG)
2:4=Fi where Fi is the number of inputs (fan-in) of the unit method. Unfortunately, all of the above methods are un-
which the connection belongs to. Since several connections   suitable for training large neural networks on large data
share a weight, this rule could be dicult to apply, but in  sets. The Gauss-Newton and Levenberg-Marquardt meth-
our case, all connections sharing a same weight belong to    ods require O(N ) operations per update, where N is

units with identical fan-ins. The reason for dividing by the the number of parameters, which makes them impracti-
fan-in is that we would like the initial standard deviation  cal for even moderate size networks. Quasi-Newton meth-
of the weighted sums to be in the same range for each        ods require \only" O(N ) operations per update, but that

unit, and to fall within the normal operating region of the  still makes them impractical for large networks. Limited-
sigmoid. If the initial weights are too small, the gradients Storage BFGS and Conjugate Gradient require only O(N )
are very small and the learning is slow. If they are too     operations per update so they would appear appropriate.
large, the sigmoids are saturated and the gradient is also   Unfortunately, their convergence speed relies on an accu-
very small. The standard deviation of the weighted sum       rate evaluation of successive \conjugate descent directions"
scales like the square root of the number of inputs when     which only makes sense in \batch" mode. For large data
the inputs are independent, and it scales linearly with the  sets, the speed-up brought by these methods over regular
number of inputs if the inputs are highly correlated. We     batch gradient descent cannot match the enormous speed
chose to assume the second hypothesis since some units       up brought by the use of stochastic gradient. Several au-
receive highly correlated signals.                           thors have attempted to use Conjugate Gradient with small
                                                             batches, or batches of increasing sizes 119], 120], but those
B. Stochastic Gradient vs Batch Gradient                     attempts have not yet been demonstrated to surpass a care-
   Gradient-Based Learning algorithms can use one of two fully tuned stochastic gradient. Our experiments were per-
classes of methods to update the parameters. The rst formed with a stochastic method that scales the parameter
method, dubbed \Batch Gradient", is the classical one: the axes so as to minimize the eccentricity of the error surface.
gradients are accumulated over the entire training set, and C. Stochastic Diagonal Levenberg-Marquardt
the parameters are updated after the exact gradient has
been so computed. In the second method, called \Stochas-        Owing to the reasons given in Appendix B, we prefer to
tic Gradient", a partial, or noisy, gradient is evaluated on update the weights after each presentation of a single pat-
the basis of one single training sample (or a small num- tern in accordance with stochastic update methods. The
ber of samples), and the parameters are updated using patterns are presented in a constant random order, and the
this approximate gradient. The training samples can be training set is typically repeated 20 times.
selected randomly or according to a properly randomized         Our update algorithm is dubbed the Stochastic Diagonal
sequence. In the stochastic version, the gradient estimates Levenberg-Marquardt method where an individual learning
are noisy, but the parameters are updated much more often rate (step size) is computed for each parameter (weight)
than with the batch version. An empirical result of con- before each pass through the training set 20], 121], 34].
siderable practical importance is that on tasks with large, These learning rates are computed using the diagonal terms
redundant data sets, the stochastic version is considerably of an estimate of the Gauss-Newton approximation to the
faster than the batch version, sometimes by orders of mag- Hessian (second derivative) matrix. This algorithm is not
nitude 117]. Although the reasons for this are not totally believed to bring a tremendous increase in learning speed
understood theoretically, an intuitive explanation can be but it converges reliably without requiring extensive ad-
found in the following extreme example. Let us take an justments of the learning parameters. It corrects major ill-
example where the training database is composed of two conditioning of the loss function that are due to the pecu-
copies of the same subset. Then accumulating the gradient liarities of the network architecture and the training data.
over the whole set would cause redundant computations The additional cost of using this procedure over standard
to be performed. On the other hand, running Stochas- stochastic gradient descent is negligible.
tic Gradient once on this training set would amount to          At each learning iteration a particular parameter wk is

<!-- page 42 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                  42

updated according to the following stochastic update rule the total input to unit i (denoted ai ). Interestingly, there is
                                             p                         an ecient algorithm to compute those second derivatives
                      wk  wk ; k @E    @wk   :                 (18)  which is very similar to the back-propagation procedure
                                                                       used to compute the rst derivatives 20], 121]:
          p
where E is the instantaneous loss function for pattern p.                   2
                                                                              @ E p = f 0 (a ) X u @ E p + f 00 (a ) @E p

                                                                                                                                 (26)
In Convolutional Neural Networks,            because of the weight             @ai           i2   2
                                                                                                     ki @a          i @x
                                                                                                                         i
sharing, the partial derivative @E                                                                         k
                                                                                2                         2
                                       p
                                    @wk   is the   sum  of the partial                          k
derivatives with respect to the connections that share the Unfortunately, using those derivatives leads to well-known
parameter wk :
                      @E p = X @E p
                                                                       problems associated with every Newton-like algorithm:
                      @wk ij 2Vk @uij                           (19) these terms can be negative, and can cause the gradient
                                                                       algorithm to move uphill instead of downhill. Therefore,
                                                                       our second approximation is a well-known trick, called the
                                       (       )

where uij is the connection weight from unit j to unit i, Vk Gauss-Newton approximation, which guarantees that the
is the set of unit index pairs (i j ) such that the connection second derivative estimates are non-negative. The Gauss-
between i and j share the parameter wk , i.e.:                         Newton approximation essentially ignores the non-linearity
                    uij = wk 8(i j ) 2 Vk                       (20) of the estimated function (the Neural Network in our case),
                                                                       but not that of the loss function. The back-propagation
As stated previously, the step sizes k are not constant but equation for Gauss-Newton approximations of the second
are function of the second derivative of the loss function derivatives is:
along the axis wk :                                                                   2
                                                                                        @ E p = f 0 (a ) X u @ E p2

                                                                                                                                 (27)
                                                                                          @ai         i
                                                                                                             ki @a

                                                                                                                   k
                                                                                          2                           2

                          k = + h                               (21)                                     k
                                        kk                             This is very similar to the formula for back-propagating the
where is a hand-picked constant and hkk is an estimate rst derivatives, except that the sigmoid's derivative and
of the second derivative of the loss function E with re- the weight values are squared. The right-hand side is a sum
spect to wk . The larger hkk , the smaller the weight update. of products of non-negative terms, therefore the left-hand
The parameter prevents the step size from becoming too side term is non-negative.
large when the second derivative is small, very much like                 The third approximation we make is that we do not run
the \model-trust" methods, and the Levenberg-Marquardt the average in Equation 24 over the entire training set, but
methods in non-linear optimization 8]. The exact formula run it on a small subset of the training set instead. In
to compute hkk from the second derivatives with respect addition the re-estimation does not need to be done of-
to the connection weights is:                                          ten since the second order properties of the error surface
                hkk =
                          X         X         @   E
                                                                       change rather slowly. In the experiments described in this
                                                                 (22) paper, we re-estimate the hkk on 500 patterns before each

                        ij 2Vk kl 2Vk    @u  ij @u kl                training pass through the training set. Since the size of the
                                                                       training set is 60,000, the additional cost of re-estimating
                      (        )           (       )

However, we make three approximations. The rst approx- the hkk is negligible. The estimates are not particularly
imation is to drop the o -diagonal terms of the Hessian sensitive to the particular subset of the training set used in
with respect to the connection weights in the above equa- the averaging. This seems to suggest that the second-order
tion:
                      hkk =
                                  X @E                      2

                                                                 (23)
                                                                       properties of the error surface are mainly determined by
                                                                       the structure of the network, rather than by the detailed
                                ij 2Vk @uij                           statistics of the samples. This algorithm is particularly use-
                                       (       )
                                                                       ful for shared-weight networks because the weight sharing
                        @ 2 E2 are the average over the training creates ill-conditionning of the error surface. Because of
Naturally, the terms @u      ij
set of the local second derivatives:                                   the sharing, one single parameter in the rst few layers can
                                                                       have an enormous inuence on the output. Consequently,
                      @E=1X
                       2             P @ Ep                 2

                                                                 (24) the   second derivative of the error with respect to this pa-
                      @uij P p @uij
                           2                                    2
                                                                       rameter may be very large, while it can be quite small for
                                                   =1
                                                                       other parameters elsewhere in the network. The above al-
Those local second derivatives with respect to connection gorithm compensates for that phenomenon.
weights can be computed from local second derivatives with                Unlike most other second-order acceleration methods for
respect to the total input of the downstream unit:                     back-propagation, the above method works in stochastic
                                                                       mode. It uses a diagonal approximation of the Hessian.
                        @ Ep = @ Ep x
                           2                       2

                                                                 (25) Like the classical Levenberg-Marquardt algorithm, it uses a
                                     @ai j

                        @uij       2                    2
                                                                       \safety" factor to prevent the step sizes from getting too
                                                  2 p                  large if the second derivative estimates are small. Hence
where xj is the state of unit j and @@aE2i is the second the method is called the Stochastic Diagonal Levenberg-
derivative of the instantaneous loss function with respect to Marquardt method.

<!-- page 43 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                43

                      Acknowledgments                                            E. Bienenstock, F. Fogelman-Soulie, and G. Weisbuch, Eds.,
                                                                                 Les Houches, France, 1986, pp. 233{240, Springer-Verlag.
   Some of the systems described in this paper is the work                19]   D. B. Parker, \Learning-logic," Tech. Rep., TR-47, Sloan
of many researchers now at AT&T, and Lucent Technolo-                            School of Management, MIT, Cambridge, Mass., April 1985.
gies. In particular, Christopher Burges, Craig Nohl, Troy                 20]   Y. LeCun, Modeles connexionnistes de l'apprentissage (con-
                                                                                 nectionist learning models), Ph.D. thesis, Universite P. et M.
Cauble and Jane Bromley contributed much to the check                            Curie (Paris 6), June 1987.
reading system. Experimental results described in sec-                    21]   Y. LeCun, \A theoretical framework for back-propagation," in
tion III include contributions by Chris Burges, Aymeric                          Proceedings of the 1988 Connectionist Models Summer School,
                                                                                 D. Touretzky, G. Hinton, and T. Sejnowski, Eds., CMU, Pitts-
Brunot, Corinna Cortes, Harris Drucker, Larry Jackel, Urs                        burgh, Pa, 1988, pp. 21{28, Morgan Kaufmann.
M"uller, Bernhard Sch"olkopf, and Patrice Simard. The au-                 22]   L. Bottou and P. Gallinari, \A framework for the cooperation of
thors wish to thank Fernando Pereira, Vladimir Vapnik,                           learning algorithms," in Advances in Neural Information Pro-
                                                                                 cessing Systems, D. Touretzky and R. Lippmann, Eds., Denver,
John Denker, and Isabelle Guyon for helpful discussions,                         1991, vol. 3, Morgan Kaufmann.
Charles Stenard and Ray Higgins for providing the appli-                  23]   C. Y. Suen, C. Nadal, R. Legault, T. A. Mai, and L. Lam,
cations that motivated some of this work, and Lawrence R.                        \Computer recognition of unconstrained handwritten numer-
                                                                                 als," Proceedings of the IEEE, Special issue on Optical Char-
Rabiner and Lawrence D. Jackel for relentless support and                        acter Recognition, vol. 80, no. 7, pp. 1162{1180, July 1992.
encouragements.                                                           24]   S. N. Srihari, \High-performance reading machines," Proceed-
                                                                                 ings of the IEEE, Special issue on Optical Character Recogni-
                                                                                 tion, vol. 80, no. 7, pp. 1120{1132, July 1992.
                             References                                   25]   Y. LeCun, L. D. Jackel, B. Boser, J. S. Denker, H. P. Graf,
1]    R. O. Duda and P. E. Hart, Pattern Classication And Scene                I. Guyon, D. Henderson, R. E. Howard, and W. Hubbard,
       Analysis, Wiley and Son, 1973.                                            \Handwritten digit recognition: Applications of neural net
2]    Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard,             chips and automatic learning," IEEE Communication, pp. 41{
       W. Hubbard, and L. D. Jackel, \Backpropagation applied to                 46, November 1989, invited paper.
       handwritten zip code recognition," Neural Computation, vol.        26]   J. Keeler, D. Rumelhart, and W. K. Leow, \Integrated seg-
       1, no. 4, pp. 541{551, Winter 1989.                                       mentation and recognition of hand-printed numerals," in Neu-
3]    S. Seung, H. Sompolinsky, and N. Tishby, \Statistical mechan-             ral Information Processing Systems, R. P. Lippmann, J. M.
       ics of learning from examples," Physical Review A, vol. 45, pp.           Moody, and D. S. Touretzky, Eds., vol. 3, pp. 557{563. Morgan
       6056{6091, 1992.                                                          Kaufmann Publishers, San Mateo, CA, 1991.
4]    V. N. Vapnik, E. Levin, and Y. LeCun, \Measuring the vc-           27]   Ofer Matan, Christopher J. C. Burges, Yann LeCun, and
       dimension of a learning machine," Neural Computation, vol. 6,             John S. Denker, \Multi-digit recognition using a space dis-
       no. 5, pp. 851{876, 1994.                                                 placement neural network," in Neural Information Processing
5]    C. Cortes, L. Jackel, S. Solla, V. N. Vapnik, and J. Denker,              Systems, J. M. Moody, S. J. Hanson, and R. P. Lippman, Eds.
       \Learning curves: asymptotic values and rate of convergence,"             1992, vol. 4, Morgan Kaufmann Publishers, San Mateo, CA.
       in Advances in Neural Information Processing Systems 6, J. D.      28]   L. R. Rabiner, \A tutorial on hidden Markov models and se-
       Cowan, G. Tesauro, and J. Alspector, Eds., San Mateo, CA,                 lected applications in speech recognition," Proceedings of the
       1994, pp. 327{334, Morgan Kaufmann.                                       IEEE, vol. 77, no. 2, pp. 257{286, February 1989.
6]    V. N. Vapnik, The Nature of Statistical Learning Theory,           29]   H. A. Bourlard and N. Morgan, CONNECTIONIST SPEECH
       Springer, New-York, 1995.                                                 RECOGNITION: A Hybrid Approach, Kluwer Academic Pub-
7]    V. N. Vapnik, Statistical Learning Theory, John Wiley & Sons,             lisher, Boston, 1994.
       New-York, 1998.                                                    30]   D. H. Hubel and T. N. Wiesel, \Receptive elds, binocular
8]    W. H. Press, B. P. Flannery, S. A. Teukolsky, and W. T. Vet-              interaction, and functional architecture in the cat's visual cor-
       terling, Numerical Recipes: The Art of Scientic Computing,               tex," Journal of Physiology (London), vol. 160, pp. 106{154,
       Cambridge University Press, Cambridge, 1986.                              1962.
9]    S. I. Amari, \A theory of adaptive pattern classiers," IEEE       31]   K. Fukushima, \Cognitron: A self-organizing multilayered neu-
       Transactions on Electronic Computers, vol. EC-16, pp. 299{                ral network," Biological Cybernetics, vol. 20, no. 6, pp. 121{136,
       307, 1967.                                                                November 1975.
10]   Ya. Tsypkin, Adaptation and Learning in automatic systems,         32]   K. Fukushima and S. Miyake, \Neocognitron: A new algorithm
       Academic Press, 1971.                                                     for pattern recognition tolerant of deformations and shifts in
11]   Ya. Tsypkin, Foundations of the theory of learning systems,               position," Pattern Recognition, vol. 15, pp. 455{469, 1982.
       Academic Press, 1973.                                              33]   M. C. Mozer, The perception of multiple objects: A connec-
12]   M. Minsky and O. Selfridge, \Learning in random nets," in                 tionist approach, MIT Press-Bradford Books, Cambridge, MA,
       4th London symposium on Information Theory, London, 1961,                 1991.
       pp. 335{347.                                                       34]   Y. LeCun, \Generalization and network design strategies," in
13]   D. H. Ackley, G. E. Hinton, and T. J. Sejnowski, \A learning              Connectionism in Perspective, R. Pfeifer, Z. Schreter, F. Fogel-
       algorithm for boltzmann machines," Cognitive Science, vol. 9,             man, and L. Steels, Eds., Zurich, Switzerland, 1989, Elsevier,
       pp. 147{169, 1985.                                                        an extended version was published as a technical report of the
14]   G. E. Hinton and T. J. Sejnowski, \Learning and relearning                University of Toronto.
       in Boltzmann machines," in Parallel Distributed Processing:        35]   Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard,
       Explorations in the Microstructure of Cognition. Volume 1:                W. Hubbard, and L. D. Jackel, \Handwritten digit recognition
       Foundations, D. E. Rumelhart and J. L. McClelland, Eds. MIT               with a back-propagation network," in Advances in Neural In-
       Press, Cambridge, MA, 1986.                                               formation Processing Systems 2 (NIPS*89), David Touretzky,
15]   D. E. Rumelhart, G. E. Hinton, and R. J. Williams, \Learning              Ed., Denver, CO, 1990, Morgan Kaufmann.
       internal representations by error propagation," in Parallel dis-   36]   G. L. Martin, \Centered-object integrated segmentation and
       tributed processing: Explorations in the microstructure of cog-           recognition of overlapping hand-printed characters," Neural
       nition, vol. I, pp. 318{362. Bradford Books, Cambridge, MA,               Computation, vol. 5, no. 3, pp. 419{429, 1993.
       1986.                                                              37]   J. Wang and J Jean, \Multi-resolution neural networks for om-
16]   A. E. Jr. Bryson and Yu-Chi Ho, Applied Optimal Control,                  nifont character recognition," in Proceedings of International
       Blaisdell Publishing Co., 1969.                                           Conference on Neural Networks, 1993, vol. III, pp. 1588{1593.
17]   Y. LeCun, \A learning scheme for asymmetric threshold net-         38]   Y. Bengio, Y. LeCun, C. Nohl, and C. Burges, \Lerec: A
       works," in Proceedings of Cognitiva 85, Paris, France, 1985,              NN/HMM hybrid for on-line handwriting recognition," Neural
       pp. 599{604.                                                              Computation, vol. 7, no. 5, 1995.
18]   Y. LeCun, \Learning processes in an asymmetric threshold           39]   S. Lawrence, C. Lee Giles, A. C. Tsoi, and A. D. Back, \Face
       network," in Disordered systems and biological organization,              recognition: A convolutional neural network approach," IEEE

<!-- page 44 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                                44

       Transactions on Neural Networks, vol. 8, no. 1, pp. 98{113,              son, J. D. Cowan, and C. L. Giles, Eds., San Mateo, CA, 1993,
       1997.                                                                    pp. 42{49, Morgan Kaufmann.
40]   K. J. Lang and G. E. Hinton, \A time delay neural network         61]   P. Simard, Y. LeCun, and Denker J., \E!cient pattern recog-
       architecture for speech recognition," Tech. Rep. CMU-CS-88-              nition using a new transformation distance," in Advances in
       152, Carnegie-Mellon University, Pittsburgh PA, 1988.                    Neural Information Processing Systems, S. Hanson, J. Cowan,
41]   A. H. Waibel, T. Hanazawa, G. Hinton, K. Shikano, and                    and L. Giles, Eds., vol. 5. Morgan Kaufmann, 1993.
       K. Lang, \Phoneme recognition using time-delay neural net-        62]   B. Boser, I. Guyon, and V. Vapnik, \A training algorithm for
       works," IEEE Transactions on Acoustics, Speech and Signal                optimal margin classiers," in Proceedings of the Fifth Annual
       Processing, vol. 37, pp. 328{339, March 1989.                            Workshop on Computational Learning Theory, 1992, vol. 5, pp.
42]   L. Bottou, F. Fogelman, P. Blanchet, and J. S. Lienard,                  144{152.
       \Speaker independent isolated digit recognition: Multilayer       63]   C. J. C. Burges and B. Schoelkopf, \Improving the accuracy
       perceptron vs dynamic time warping," Neural Networks, vol.               and speed of support vector machines," in Advances in Neural
       3, pp. 453{465, 1990.                                                    Information Processing Systems 9, M. Jordan M. Mozer and
43]   P. Ha ner and A. H. Waibel, \Time-delay neural networks                  T. Petsche, Eds. 1997, The MIT Press, Cambridge.
       embedding time alignment: a performance analysis," in EU-         64]   Eduard Sackinger, Bernhard Boser, Jane Bromley, Yann Le-
       ROSPEECH'91, 2nd European Conference on Speech Commu-                    Cun, and Lawrence D. Jackel, \Application of the ANNA neu-
       nication and Technology, Genova, Italy, Sept. 1991.                      ral network chip to high-speed character recognition," IEEE
44]   I. Guyon, P. Albrecht, Y. LeCun, J. S. Denker, and W. Hub-               Transaction on Neural Networks, vol. 3, no. 2, pp. 498{505,
       bard, \Design of a neural network character recognizer for a             March 1992.
       touch terminal," Pattern Recognition, vol. 24, no. 2, pp. 105{    65]   J. S. Bridle, \Probabilistic interpretation of feedforward classi-
       119, 1991.                                                               cation networks outputs, with relationship to statistical pattern
45]   J. Bromley, J. W. Bentz, L. Bottou, I. Guyon, Y. LeCun,                  recognition," in Neurocomputing, Algorithms, Architectures
       C. Moore, E. Sackinger, and R. Shah, \Signature verica-                 and Applications, F. Fogelman, J. Herault, and Y. Burnod,
       tion using a siamese time delay neural network," International           Eds., Les Arcs, France, 1989, Springer.
       Journal of Pattern Recognition and Articial Intelligence, vol.   66]   Y. LeCun, L. Bottou, and Y. Bengio, \Reading checks with
       7, no. 4, pp. 669{687, August 1993.                                      graph transformer networks," in International Conference on
46]   Y. LeCun, I. Kanter, and S. Solla, \Eigenvalues of covariance            Acoustics, Speech, and Signal Processing, Munich, 1997, vol. 1,
       matrices: application to neural-network learning," Physical              pp. 151{154, IEEE.
       Review Letters, vol. 66, no. 18, pp. 2396{2399, May 1991.         67]   Y. Bengio, Neural Networks for Speech and Sequence Recogni-
47]   T. G. Dietterich and G. Bakiri, \Solving multiclass learning             tion, International Thompson Computer Press, London, UK,
       problems via error-correcting output codes.," Journal of Arti-           1996.
       cial Intelligence Research, vol. 2, pp. 263{286, 1995.           68]   C. Burges, O. Matan, Y. LeCun, J. Denker, L. Jackel, C. Ste-
48]   L. R. Bahl, P. F. Brown, P. V. de Souza, and R. L. Mercer,               nard, C. Nohl, and J. Ben, \Shortest path segmentation: A
       \Maximum mutual information of hidden Markov model pa-                   method for training a neural network to recognize character
       rameters for speech recognition," in Proc. Int. Conf. Acoust.,           strings," in International Joint Conference on Neural Net-
       Speech, Signal Processing, 1986, pp. 49{52.                              works, Baltimore, 1992, vol. 3, pp. 165{172.
49]   L. R. Bahl, P. F. Brown, P. V. de Souza, and R. L. Mercer,        69]   T. M. Breuel, \A system for the o -line recognition of hand-
       \Speech recognition with continuous-parameter hidden Markov              written text," in ICPR'94, IEEE, Ed., Jerusalem 1994, 1994,
       models," Computer, Speech and Language, vol. 2, pp. 219{234,             pp. 129{134.
       1987.                                                             70]   A. Viterbi, \Error bounds for convolutional codes and an
50]   B. H. Juang and S. Katagiri, \Discriminative learning for min-           asymptotically optimum decoding algorithm," IEEE Trans-
       imum error classication," IEEE Trans. on Acoustics, Speech,             actions on Information Theory, pp. 260{269, April 1967.
       and Signal Processing, vol. 40, no. 12, pp. 3043{3054, December   71]   Lippmann R. P. and Gold B., \Neural-net classiers useful for
       1992.                                                                    speech recognition," in Proceedings of the IEEE First Interna-
51]   Y. LeCun, L. D. Jackel, L. Bottou, A. Brunot, C. Cortes, J. S.           tional Conference on Neural Networks, San Diego, June 1987,
       Denker, H. Drucker, I. Guyon, U. A. Muller, E. Sackinger,                pp. 417{422.
       P. Simard, and V. N. Vapnik, \Comparison of learning al-          72]   H. Sakoe, R. Isotani, K. Yoshida, K. Iso, and T. Watan-
       gorithms for handwritten digit recognition," in International            abe, \Speaker-independent word recognition using dynamic
       Conference on Articial Neural Networks, F. Fogelman and                 programming neural networks," in International Conference
       P. Gallinari, Eds., Paris, 1995, pp. 53{60, EC2 & Cie.                   on Acoustics, Speech, and Signal Processing, Glasgow, 1989,
52]   I Guyon, I. Poujaud, L. Personnaz, G. Dreyfus, J. Denker, and            pp. 29{32.
       Y. LeCun, \Comparing di erent neural net architectures for        73]   J. S. Bridle, \Alphanets: a recurrent `neural' network archi-
       classifying handwritten digits," in Proc. of IJCNN, Washing-             tecture with a hidden markov model interpretation," Speech
       ton DC. 1989, vol. II, pp. 127{132, IEEE.                                Communication, vol. 9, no. 1, pp. 815{819, 1990.
53]   R. Ott, \construction of quadratic polynomial classiers,"        74]   M. A. Franzini, K. F. Lee, and A. H. Waibel, \Connectionist
       in Proc. of International Conference on Pattern Recognition.             viterbi training: a new hybrid method for continuous speech
       1976, pp. 161{165, IEEE.                                                 recognition," in International Conference on Acoustics, Speech,
54]   J. Schurmann, \A multi-font word recognition system for postal           and Signal Processing, Albuquerque, NM, 1990, pp. 425{428.
       address reading," IEEE Transactions on Computers, vol. C-27,      75]   L. T. Niles and H. F. Silverman, \Combining hidden markov
       no. 8, pp. 721{732, August 1978.                                         models and neural network classiers," in International Con-
55]   Y. Lee, \Handwritten digit recognition using k-nearest neigh-            ference on Acoustics, Speech, and Signal Processing, Albu-
       bor, radial-basis functions, and backpropagation neural net-             querque, NM, 1990, pp. 417{420.
       works," Neural Computation, vol. 3, no. 3, pp. 440{449, 1991.     76]   X. Driancourt and L. Bottou, \MLP, LVQ and DP: Compari-
56]   D. Saad and S. A. Solla, \Dynamics of on-line gradient de-               son & cooperation," in Proceedings of the International Joint
       scent learning for multilayer neural networks," in Advances in           Conference on Neural Networks, Seattle, 1991, vol. 2, pp. 815{
       Neural Information Processing Systems, David S. Touretzky,               819.
       Michael C. Mozer, and Michael E. Hasselmo, Eds. 1996, vol. 8,     77]   Y. Bengio, R. De Mori, G. Flammia, and R. Kompe, \Global
       pp. 302{308, The MIT Press, Cambridge.                                   optimization of a neural network-hidden Markov model hy-
57]   G. Cybenko, \Approximation by superpositions of sigmoidal                brid," IEEE Transactions on Neural Networks, vol. 3, no. 2,
       functions," Mathematics of Control, Signals, and Systems, vol.           pp. 252{259, 1992.
       2, no. 4, pp. 303{314, 1989.                                      78]   P. Ha ner and A. H. Waibel, \Multi-state time-delay neural
58]   L. Bottou and V. N. Vapnik, \Local learning algorithms," Neu-            networks for continuous speech recognition," in Advances in
       ral Computation, vol. 4, no. 6, pp. 888{900, 1992.                       Neural Information Processing Systems. 1992, vol. 4, pp. 579{
59]   R. E. Schapire, \The strength of weak learnability," Machine             588, Morgan Kaufmann, San Mateo.
       Learning, vol. 5, no. 2, pp. 197{227, 1990.                       79]   Y. Bengio, , P. Simard, and P. Frasconi, \Learning long-term
60]   H. Drucker, R. Schapire, and P. Simard, \Improving perfor-               dependencies with gradient descent is di!cult," IEEE Trans-
       mance in neural networks using a boosting algorithm," in Ad-             actions on Neural Networks, vol. 5, no. 2, pp. 157{166, March
       vances in Neural Information Processing Systems 5, S. J. Han-            1994, Special Issue on Recurrent Neural Network.

<!-- page 45 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                            45

80] T. Kohonen, G. Barna, and R. Chrisley, \Statistical pattern              Lippmann, Eds., Denver, CO, 1992, pp. 175{182, Morgan Kauf-
     recognition with neural network: Benchmarking studies," in               mann.
     Proceedings of the IEEE Second International Conference on         100] F. C. N. Pereira and M. Riley, \Speech recognition by compo-
     Neural Networks, San Diego, 1988, vol. 1, pp. 61{68.                     sition of weighted nite automata," in Finite-State Devices for
81] P. Ha ner, \Connectionist speech recognition with a global               Natural Langue Processing, Cambridge, Massachusetts, 1997,
     MMI algorithm," in EUROSPEECH'93, 3rd European Confer-                   MIT Press.
     ence on Speech Communication and Technology, Berlin, Sept.         101] M. Mohri, \Finite-state transducers in language and speech
     1993.                                                                    processing," Computational Linguistics, vol. 23, no. 2, pp. 269{
82] J. S. Denker and C. J. Burges, \Image segmentation and recog-            311, 1997.
     nition," in The Mathematics of Induction. 1995, Addison Wes-       102] I. Guyon, M. Schenkel, and J. Denker, \Overview and syn-
     ley.                                                                     thesis of on-line cursive handwriting recognition techniques,"
83] L. Bottou, Une Approche theorique de l'Apprentissage Connex-            in Handbook on Optical Character Recognition and Document
     ionniste: Applications a la Reconnaissance de la Parole, Ph.D.           Image Analysis, P. S. P. Wang and Bunke H., Eds. 1996, World
     thesis, Universite de Paris XI, 91405 Orsay cedex, France, 1991.         Scientic.
84] M. Rahim, Y. Bengio, and Y. LeCun, \Discriminative feature         103] M. Mohri and M. Riley, \Weighted determinization and min-
     and model design for automatic speech recognition," in Proc.             imization for large vocabulary recognition," in Proceedings of
     of Eurospeech, Rhodes, Greece, 1997.                                     Eurospeech '97, Rhodes, Greece, September 1997, pp. 131{134.
85] U. Bodenhausen, S. Manke, and A. Waibel, \Connectionist ar-        104] Y. Bengio and P. Frasconi, \An input/output HMM architec-
     chitectural learning for high performance character and speech           ture," in Advances in Neural Information Processing Systems,
     recognition," in International Conference on Acoustics, Speech,          G. Tesauro, D Touretzky, and T. Leen, Eds. 1996, vol. 7, pp.
     and Signal Processing, Minneapolis, 1993, vol. 1, pp. 625{628.           427{434, MIT Press, Cambridge, MA.
86] F. Pereira, M. Riley, and R. Sproat, \Weighted rational trans-     105] Y. Bengio and P. Frasconi, \Input/Output HMMs for sequence
     ductions and their application to human language processing,"            processing," IEEE Transactions on Neural Networks, vol. 7,
     in ARPA Natural Language Processing workshop, 1994.                      no. 5, pp. 1231{1249, 1996.
87] M. Lades, J. C. Vorbruggen, J. Buhmann, and C. von der Mals-       106] M. Mohri, F. C. N. Pereira, and M. Riley, A rational design
     burg, \Distortion invariant object recognition in the dynamic            for a weighted nite-state transducer library, Lecture Notes in
     link architecture," IEEE Trans. Comp., vol. 42, no. 3, pp.               Computer Science. Springer Verlag, 1997.
     300{311, 1993.                                                     107] M. Rahim, C. H. Lee, and B. H. Juang, \Discriminative ut-
88] B. Boser, E. Sackinger, J. Bromley, Y. LeCun, and L. Jackel,             terance verication for connected digits recognition," IEEE
     \An analog neural network processor with programmable topol-             Trans. on Speech & Audio Proc., vol. 5, pp. 266{277, 1997.
     ogy," IEEE Journal of Solid-State Circuits, vol. 26, no. 12, pp.   108] M. Rahim, Y. Bengio, and Y. LeCun, \Discriminative feature
     2017{2025, December 1991.                                                and model design for automatic speech recognition," in Eu-
89] M. Schenkel, H. Weissman, I. Guyon, C. Nohl, and D. Hender-              rospeech '97, Rhodes, Greece, 1997, pp. 75{78.
     son, \Recognition-based segmentation of on-line hand-printed       109] S. Bengio and Y. Bengio, \An EM algorithm for asynchronous
     words," in Advances in Neural Information Processing Systems             input/output hidden Markov models," in International Con-
     5, S. J. Hanson, J. D. Cowan, and C. L. Giles, Eds., Denver,             ference On Neural Information Processing, L. Xu, Ed., Hong-
     CO, 1993, pp. 723{730.                                                   Kong, 1996, pp. 328{334.
90] C. Dugast, L. Devillers, and X. Aubert, \Combining TDNN            110] C. Tappert, C. Suen, and T. Wakahara, \The state of the
     and HMM in a hybrid system for improved continuous-speech                art in on-line handwriting recognition," IEEE Transactions on
     recognition," IEEE Transactions on Speech and Audio Pro-                 Pattern Analysis and Machine Intelligence, vol. 8, no. 12, pp.
     cessing, vol. 2, no. 1, pp. 217{224, 1994.                               787{808, 1990.
91] Ofer Matan, Henry S. Baird, Jane Bromley, Christopher J. C.        111] S. Manke and U. Bodenhausen, \A connectionist recognizer for
     Burges, John S. Denker, Lawrence D. Jackel, Yann Le Cun, Ed-             on-line cursive handwriting recognition," in International Con-
     win P. D. Pednault, William D. Sattereld, Charles E. Stenard,           ference on Acoustics, Speech, and Signal Processing, Adelaide,
     and Timothy J. Thompson, \Reading handwritten digits: A                  1994, vol. 2, pp. 633{636.
     ZIP code recognition system," Computer, vol. 25, no. 7, pp.        112] M. Gilloux and M. Leroux, \Recognition of cursive script
     59{62, July 1992.                                                        amounts on postal checks," in European Conference dedicated
92] Y. Bengio and Y. Le Cun, \Word normalization for on-line                 to Postal Technologies, Nantes, France, June 1993, pp. 705{
     handwritten word recognition," in Proc. of the International             712.
     Conference on Pattern Recognition, IAPR, Ed., Jerusalem,           113] D. Guillevic and C. Y. Suen, \Cursive script recognition applied
     1994, IEEE.                                                              to the processing of bank checks," in Int. Conf. on Document
93] R. Vaillant, C. Monrocq, and Y. LeCun, \Original approach                Analysis and Recognition, Montreal, Canada, August 1995, pp.
     for the localization of objects in images," IEE Proc on Vision,          11{14.
     Image, and Signal Processing, vol. 141, no. 4, pp. 245{250,        114] L. Lam, C. Y. Suen, D. Guillevic, N. W. Strathy, M. Cheriet,
     August 1994.                                                             K. Liu, and J. N. Said, \Automatic processing of information
94] R. Wolf and J. Platt, \Postal address block location using a             on checks," in Int. Conf. on Systems, Man & Cybernetics,
     convolutional locator network," in Advances in Neural Infor-             Vancouver, Canada, October 1995, pp. 2353{2358.
     mation Processing Systems 6, J. D. Cowan, G. Tesauro, and          115] C. J. C. Burges, J. I. Ben, J. S. Denker, Y. LeCun, and C. R.
     J. Alspector, Eds. 1994, pp. 745{752, Morgan Kaufmann Pub-               Nohl, \O line recognition of handwritten postal words using
     lishers, San Mateo, CA.                                                  neural networks," Int. Journal of Pattern Recognition and Ar-
95] S. Nowlan and J. Platt, \A convolutional neural network hand             ticial Intelligence, vol. 7, no. 4, pp. 689, 1993, Special Issue
     tracker," in Advances in Neural Information Processing Sys-              on Applications of Neural Networks to Pattern Recognition (I.
     tems 7, G. Tesauro, D. Touretzky, and T. Leen, Eds., San Ma-             Guyon Ed.).
     teo, CA, 1995, pp. 901{908, Morgan Kaufmann.                       116] Y. LeCun, Y. Bengio, D. Henderson, A. Weisbuch, H. Weiss-
96] H. A. Rowley, S. Baluja, and T. Kanade, \Neural network-                 man, and Jackel. L., \On-line handwriting recognition with
     based face detection," in Proceedings of CVPR'96. 1996, pp.              neural networks: spatial representation versus temporal repre-
     203{208, IEEE Computer Society Press.                                    sentation.," in Proc. International Conference on handwriting
97] E. Osuna, R. Freund, and F. Girosi, \Training support vector             and drawing. 1993, Ecole Nationale Superieure des Telecommu-
     machines: an application to face detection," in Proceedings of           nications.
     CVPR'96. 1997, pp. 130{136, IEEE Computer Society Press.           117] U. Muller, A. Gunzinger, and W. Guggenbuhl, \Fast neural
98] H. Bourlard and C. J. Wellekens, \Links between Markov mod-              net simulation with a DSP processor array," IEEE Trans. on
     els and multilayer perceptrons," in Advances in Neural Infor-            Neural Networks, vol. 6, no. 1, pp. 203{213, 1995.
     mation Processing Systems, D. Touretzky, Ed., Denver, 1989,        118] R. Battiti, \First- and second-order methods for learning: Be-
     vol. 1, pp. 186{187, Morgan-Kaufmann.                                    tween steepest descent and newton's method.," Neural Com-
99] Y. Bengio, R. De Mori, G. Flammia, and R. Kompe, \Neu-                   putation, vol. 4, no. 2, pp. 141{166, 1992.
     ral network - gaussian mixture hybrid for speech recognition       119] A. H. Kramer and A. Sangiovanni-Vincentelli, \E!cient par-
     or density estimation," in Advances in Neural Information                allel learning algorithms for neural networks," in Advances in
     Processing Systems 4, J. E. Moody, S. J. Hanson, and R. P.               Neural Information Processing Systems, D.S. Touretzky, Ed.,

<!-- page 46 -->
PROC. OF THE IEEE, NOVEMBER 1998                                                                                                              46

      Denver 1988, 1989, vol. 1, pp. 40{48, Morgan Kaufmann, San                                Yoshua Bengio Yoshua Bengio received his
      Mateo.                                                                                     B.Eng. in electrical engineering in 1986 from
120] M. Moller, Ecient Training of Feed-Forward Neural Net-                                    McGill University. He also received a M.Sc.
      works, Ph.D. thesis, Aarhus University, Aarhus, Denmark,                                   and a Ph.D. in computer science from McGill
      1993.                                                                                      University in 1988 and 1991 respectively. In
121] S. Becker and Y. LeCun, \Improving the convergence of back-                                1991-1992 he was a post-doctoral fellow at the
      propagation learning with second-order methods," Tech. Rep.                                Massachusetts Institute of Technology. In 1992
      CRG-TR-88-5, University of Toronto Connectionist Research                                  he joined AT&T Bell Laboratories, which later
      Group, September 1988.                                                                     became AT&T Labs-Research. In 1993 he
                                                                                                 joined the faculty of the computer science de-
                                                                                                 partment of the Universite de Montreal where
                                                                          he is now an associate professor. Since his rst work on neural net-
                                                                          works in 1986, his research interests have been centered around learn-
                                                                          ing algorithms especially for data with a sequential or spatial nature,
                                                                          such as speech, handwriting, and time-series.

                       Yann LeCun Yann LeCun received a                                         Patrick Ha ner Patrick Ha ner graduated
                        Dipl^ome d'Ingenieur from the Ecole Superieure                           from Ecole Polytechnique, Paris, France in
                        d'Ingenieur en Electrotechnique et Electron-                             1987 and from Ecole Nationale Superieure des
                        ique, Paris in 1983, and a PhD in Computer                               Telecommunications (ENST), Paris, France in
                        Science from the Universite Pierre et Marie                              1989. He received his Ph.D in speech and sig-
                        Curie, Paris, in 1987, during which he proposed                          nal processing from ENST in 1994. In 1988
                        an early version of the back-propagation learn-                          and 1990, he worked with Alex Waibel on the
                        ing algorithm for neural networks. He then                               design of the TDNN and the MS-TDNN ar-
                        joined the Department of Computer Science at                             chitectures at ATR (Japan) and Carnegie Mel-
                        the University of Toronto as a research asso-                            lon University. From 1989 to 1995, as a re-
                        ciate. In 1988, he joined the Adaptive Systems                           search scientist for CNET/France-Telecom in
Research Department at AT&T Bell Laboratories in Holmdel, NJ,             Lannion, France, he developed connectionist learning algorithms for
where he worked among other thing on neural networks, machine             telephone speech recognition. In 1995, he joined AT&T Bell Labora-
learning, and handwriting recognition. Following AT&T's second            tories and worked on the application of Optical Character Recognition
breakup in 1996, he became head of the Image Processing Services          and transducers to the processing of nancial documents. In 1997, he
Research Department at AT&T Labs-Research.                                joined the Image Processing Services Research Department at AT&T
   He is serving on the board of the Machine Learning Journal, and        Labs-Research. His research interests include statistical and connec-
has served as associate editor of the IEEE Trans. on Neural Networks.     tionist models for sequence recognition, machine learning, speech and
He is general chair of the "Machines that Learn" workshop held every      image recognition, and information theory.
year since 1986 in Snowbird, Utah. He has served as program co-chair
of IJCNN 89, INNC 90, NIPS 90,94, and 95. He is a member of the
IEEE Neural Network for Signal Processing Technical Committee.
   He has published over 70 technical papers and book chapters on
neural networks, machine learning, pattern recognition, handwriting
recognition, document understanding, image processing, VLSI design,
and information theory. In addition to the above topics, his current
interests include video-based user interfaces, image compression, and
content-based indexing of multimedia material.

                       Leon Bottou Leon Bottou received a Dipl^ome
                       from Ecole Polytechnique, Paris in 1987, a
                       Magist$ere en Mathematiques Fondamentales et
                       Appliquees et Informatiques from Ecole Nor-
                       male Superieure, Paris in 1988, and a PhD
                       in Computer Science from Universite de Paris-
                       Sud in 1991, during which he worked on speech
                       recognition and proposed a framework for
                       stochastic gradient learning and global train-
                       ing. He then joined the Adaptive Systems Re-
                       search Department at AT&T Bell Laboratories
where he worked on neural network, statistical learning theory and
local learning algorithms. He returned to France in 1992 as a research
engineer at ONERA. He then became chairman of Neuristique S.A.,
a company making neural network simulators and tra!c forecast-
ing software. He eventually came back to AT&T Bell Laboratories
in 1995 where he worked on graph transformer networks for optical
character recognition. He is now a member of the Image Process-
ing Services Research Department at AT&T Labs-Research. Besides
learning algorithms, his current interests include arithmetic coding,
image compression and indexing.

