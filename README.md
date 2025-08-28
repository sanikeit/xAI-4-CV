# xAI-4-CV

# Explainable AI for Computer Vision (XAI4CV)
## Complete Course Notes and Exam Preparation

---

## Table of Contents
1. [Lecture 1: Introduction to XAI](#lecture-1-introduction-to-xai)
2. [Lecture 2: The Many Faces of Explainability](#lecture-2-the-many-faces-of-explainability)
3. [Lecture 3: Backpropagation-Based Pixel Attributions](#lecture-3-backpropagation-based-pixel-attributions)
4. [Lecture 4: Interpretable Models](#lecture-4-interpretable-models)
5. [Lecture 5: Model-Agnostic Methods](#lecture-5-model-agnostic-methods)
6. [Detailed Q&A Section](#detailed-qa-section)
7. [Exam Guide](#xai4cv-exam-preparation-guide)

---

## Lecture 1: Introduction to XAI

### 1.1 Course Overview

**What is Explainable AI (XAI)?**
- XAI refers to AI systems that are transparent in their operations, allowing human users to understand and trust decisions
- Goal: Create models where humans can understand the cause of a decision
- Key principle: Explanations must be faithful to the model AND understandable by humans

**Definition by Tim Miller (2017):**
> "Interpretability is the degree to which a human can understand the cause of a decision."

### 1.2 Why Do We Need Explainability?

#### 1.2.1 Bias Detection
- AI systems can suffer from dataset bias
- Example: Medical imaging models focusing on skin markers instead of actual pathology
- Example: Autonomous driving systems making decisions based on incorrect contextual cues

#### 1.2.2 Legal and Regulatory Requirements
- **GDPR**: Demands transparent data processing
- **EU AI Act**: Requirements for high-risk AI systems
- **Algorithmic Accountability Act (USA)**: Assessment of AI system impacts

#### 1.2.3 Four Main Arguments for XAI

**1. Explain to Justify**
- Ensure auditable and provable defense of outputs
- Compliance with regulations and fairness
- Increase trust and accountability

**2. Explain to Control**
- Identify and override erroneous predictions
- Bias identification and risk control
- Understanding model behavior on new/unseen data

**3. Explain to Improve**
- Enable ML model improvement
- Error detection and correction
- Build more generalizable models

**4. Explain to Discover**
- Recognize previously unknown patterns
- Learn from AI insights in big data structures

### 1.3 Goals of XAI

- **Fairness**: Unbiased predictions without discrimination
- **Privacy**: Protection of sensitive information
- **Reliability/Robustness**: Consistent behavior with input variations
- **Causality**: Capturing only causal relationships
- **Trust**: Building human confidence in AI decisions

### 1.4 Historical Context

#### 1.4.1 Expert Systems (1970s-1980s)
- Rule-based systems with explicit knowledge bases
- **Advantages**: Explanations by design
- **Limitations**: Cannot generalize, expensive knowledge acquisition

#### 1.4.2 Evolution of Explanation Systems
- **First Generation**: Simple rule verbalization
- **Second Generation**: Context-sensitive, user-tailored explanations
- **Third Generation**: Modern deep learning explanations focusing on contrastive explanations

### 1.5 Current Challenges

#### 1.5.1 Accuracy vs. Explainability Trade-off
- Simple models (linear, decision trees) are interpretable but may lack accuracy
- Complex models (deep neural networks) achieve high accuracy but lack transparency

#### 1.5.2 Complexity Issues
- Neural networks have millions/billions of parameters
- Mathematical function is faithful but too complex for human understanding
- Need for approximation methods that maintain fidelity

---

## Lecture 2: The Many Faces of Explainability

### 2.1 Different Types of Explanations

#### 2.1.1 Explanation vs. Explanation vs. Explanation

**Human Prior Knowledge Explanations**
- Align with human understanding of concepts
- Example: "This is a dog because it has four legs, fur, etc."

**Model-Faithful Explanations**
- Represent actual model decision-making process
- Example: "This is a dog because pixel (X,Y) has value Z"
- May not align with human intuition but accurately represents model behavior

### 2.2 Machine Learning Ecosystem Roles

#### 2.2.1 Agent Types (Tomsett et al. Framework)

**Creators**
- Design and build ML systems
- Goal: Improve system performance and optimization

**Operators**
- Directly interact with ML systems
- Provide inputs and receive outputs
- Goal: Ensure correct data input and information relay

**Executors**
- Make decisions based on ML system information
- Goal: Make good decisions (definition of "good" varies by context)

**Decision-subjects**
- Affected by executor decisions
- Goal: Understand why specific decisions were made

**Data-subjects**
- Personal data used in training
- Goal: Privacy protection and data rights

**Examiners**
- Investigate and audit ML systems
- Goal: Safety testing, auditing, forensic investigation

#### 2.2.2 Role-Based Interpretability Requirements
- Different agents require different types of explanations
- Creators need performance insights
- Executors need decision support
- Decision-subjects need justification
- Examiners need comprehensive system understanding

### 2.3 Properties of Explanation Methods

#### 2.3.1 Method Properties

**Expressive Power**
- "Language" or structure of explanations
- Examples: IF-THEN rules, decision trees, weighted sums, attribution maps

**Translucency**
- Level of access to model internals
- High: Full access to parameters and gradients
- Low: Only input-output access (black-box)

**Portability**
- Range of ML models the method can explain
- Low translucency → High portability
- Model-specific methods have low portability

**Algorithmic Complexity**
- Computational cost of explanation method
- Important for real-time applications

#### 2.3.2 Individual Explanation Properties

**Fidelity**
- How accurately the explanation represents the model
- Most important property for trustworthy explanations
- Can be local (single instance) or global (entire model)

**Stability**
- Consistency of explanations for similar instances
- High stability: similar inputs → similar explanations
- Low stability may indicate method variance or model non-determinism

**Comprehensibility**
- How well humans understand the explanations
- Difficult to define and measure objectively
- Critical for practical adoption

**Certainty**
- Whether explanation reflects model confidence
- Should indicate when model is uncertain

**Novelty**
- Explanation's ability to handle out-of-distribution data
- Related to certainty (high novelty ↔ low certainty)

**Representativeness**
- Scope of explanation coverage
- Global: explains entire model behavior
- Local: explains individual predictions

### 2.4 What Makes a Good Explanation?

#### 2.4.1 Human-Centered Principles

**Contrastive Nature**
- Humans ask "Why this instead of that?"
- Explanations should highlight differences between alternatives

**Selectivity**
- Humans don't want complete reasoning
- Good explanations are concise and focused

**Social Interaction**
- Explanations are communication between explainer and recipient
- Must be tailored to the audience

**Focus on Abnormal**
- Humans focus on unusual causes
- Abnormal features should be highlighted in explanations

**Truthfulness**
- Explanations must be faithful to the model
- Trade-off between fidelity and comprehensibility

**Prior Belief Consistency**
- Good explanations align with recipient's prior knowledge
- Conflicting explanations may be devalued due to confirmation bias

**Generality and Probability**
- General explanations that cover many cases are preferred
- In absence of abnormal events, general explanations are better

---

## Lecture 3: Backpropagation-Based Pixel Attributions

### 3.1 Pixel Attribution Definition

**Goal**: Highlight how relevant each input pixel is for the final prediction

**Applications**:
- Detecting biases and model misbehavior
- Understanding model decision mechanisms
- Building trust through transparency

### 3.2 Gradient-Based Methods

#### 3.2.1 Vanilla Gradient (Saliency Maps)

**Motivation**: For linear models f_c(x) = w^T_c x + b_c, the weight magnitude |w_i| defines pixel importance

**Extension to DNNs**: Use first-order Taylor expansion around input x_0:
f_c(x_0) ≈ w^T_c x_0 + b_c, where w = ∂f_c(x_0)/∂x_0

**Computation**:
1. Forward pass through network
2. Compute gradient: ∂f_c(x_0)/∂x_0
3. Optional: Take maximum magnitude across color channels

**Intuition**: If small pixel changes strongly impact output, that pixel is important

#### 3.2.2 Input × Gradient

**Formula**: IxG = x_0 ⊙ (∂f_c(x_0)/∂x_0)

**Advantages**:
- Leverages both input strength and gradient information
- Often produces cleaner visualizations than vanilla gradient
- Simple computation with element-wise multiplication

#### 3.2.3 SmoothGrad

**Problem**: Gradient-based methods are noisy due to derivative fluctuations

**Solution**: Add noise and average over multiple noisy versions

**Formula**: Â_c(x,f) = (1/n) Σ A_c(x + N(0,σ²), f)

**Parameters**:
- σ: Noise standard deviation (10-20% of input range)
- n: Number of samples (recommended: 50)

**Benefits**: Reduces noise while preserving important features

### 3.3 Class Activation Mapping (CAM)

#### 3.3.1 CAM Requirements
- Network architecture: Conv layers → Global Average Pooling → Linear layer
- Examples: ResNet-50, GoogLeNet

#### 3.3.2 CAM Computation
- Maps predicted class score back to convolutional feature maps
- Uses linear layer weights to weight feature maps
- Results in class-specific activation maps
- Requires upsampling to match input resolution

### 3.4 Grad-CAM

#### 3.4.1 Motivation
- Generalizes CAM to any CNN architecture
- Doesn't require specific architectural constraints

#### 3.4.2 Computation Steps
1. Forward pass to get class score f_c(x)
2. Compute gradients: ∂f_c(x)/∂f^(1→l)_k(x)
3. Global average pooling: α^c_k = (1/Z) Σ_i Σ_j (∂f_c(x)/∂f^(1→l)_k(x)_{i,j})
4. Linear combination: Grad-CAM_c = ReLU(Σ_k α^c_k f^(1→l)_k(x))

#### 3.4.3 Guided Grad-CAM
- Combines coarse Grad-CAM with high-resolution gradient methods
- Formula: Guided Grad-CAM_c = Grad-CAM_c ⊙ Vanilla Gradient_c
- Provides both localization and fine-grained details

### 3.5 Layer-wise Relevance Propagation (LRP)

**Key Principles**:
- Backward propagation of relevance through layers
- Conservation principle: relevance is preserved
- Uses specific decomposition rules for different layer types
- Provides pixel-wise explanations

### 3.6 Integrated Gradients (IG)

#### 3.6.1 Motivation: Axiomatic Attribution

**Problem**: Gradient saturation in neural networks
- Gradients can be near zero even when features are important
- Violates sensitivity axiom

**Axioms**:
- **Sensitivity**: Relevant features should have non-zero attribution
- **Implementation Invariance**: Functionally equivalent models should have identical attributions

#### 3.6.2 IG Formula
IG(x, b, f) = (x - b) ∫₀¹ ∇_x f(α·x + (1-α)·b) dα

**Components**:
- x: Input image
- b: Baseline (e.g., black image)
- α: Interpolation constant [0,1]
- Integration approximated using Riemann sum

#### 3.6.3 Baseline Selection
- **Black image**: Common but contains "black" information
- **Blurred image**: Removes high-frequency details
- **Uniform noise**: Random baseline
- **Training distribution**: Multiple baselines for robustness

### 3.7 Fast Integrated Gradients (𝒳-Gradients)

#### 3.7.1 Problem
- IG requires 50-200 forward passes
- Computationally expensive for real-time applications

#### 3.7.2 Solution: Nonnegative Homogeneity
- Remove bias terms from network: α(Wx) = W(αx)
- For nonnegatively homogeneous networks: IG = Input × Gradient
- Significant speedup with minimal accuracy loss

#### 3.7.3 𝒳-Network Properties
- Equivariant to contrast changes
- Fast axiomatic attribution
- Small accuracy drop (typically <1% on ImageNet)

---

## Lecture 4: Interpretable Models

### 4.1 Introduction to Interpretable Models

**Philosophy**: Design models with explainability in mind rather than explaining complex models post-hoc

**Advantage**: Inherent interpretability vs. approximation-based explanations

### 4.2 Linear Models

#### 4.2.1 Linear Regression

**Formula**: y = w₀ + w₁x₁ + ... + wₙxₙ + ε

**Properties**:
- **Expressive Power**: Weighted sum
- **Translucency**: High (full access to weights)
- **Portability**: Low (model-specific)
- **Fidelity**: High (exact representation)
- **Comprehensibility**: High (simple interpretation)

**Interpretation Template**: "An increase of feature x_k by one unit increases the prediction by w_k units when all other features remain fixed"

**Advantages**:
- Easy to interpret and implement
- Monotonic relationships
- Fast computation

**Disadvantages**:
- Cannot handle interactions by default
- Strong assumptions (linearity, Gaussian errors, homoscedasticity)
- May be too simple for complex problems

#### 4.2.2 Logistic Regression

**Formula**: P(y=1|x) = 1/(1 + exp(-(w₀ + w₁x₁ + ... + wₙxₙ)))

**Differences from Linear Regression**:
- Outputs probabilities [0,1]
- Uses sigmoid/softmax functions
- More complex (multiplicative) weight interpretation
- Better suited for classification

### 4.3 Decision Trees

#### 4.3.1 CART Algorithm

**Process**:
1. Select splitting dimension and threshold (e.g., using Gini index)
2. Create subsets as different as possible
3. Repeat until stopping criteria met

**Properties**:
- **Expressive Power**: Decision rules/trees
- **Fidelity**: High (exact representation)
- **Comprehensibility**: Depends on tree depth
- **Stability**: Low (sensitive to data changes)

**Advantages**:
- Captures feature interactions
- No assumptions about data distribution
- Easy to interpret (if small)

**Disadvantages**:
- Approximates linear relationships with steps
- Unstable (small data changes → different trees)
- Exponential growth in complexity

### 4.4 BagNet

#### 4.4.1 Bag-of-Words Analogy

**Inspiration**: Document classification using word frequency
**Adaptation**: Image classification using patch evidence

#### 4.4.2 Architecture

**Process**:
1. Split image into q×q patches
2. Pass patches through DNN (ResNet-like)
3. Apply linear classifier to each patch
4. Average class evidence across all patches

**Variants**: BagNet-q where q ∈ {9, 17, 33}

**Benefits**:
- Interpretable patch-based reasoning
- Good accuracy with small patches
- Clear attribution maps

### 4.5 Prototypical Part Networks

#### 4.5.1 ProtoPNet

**Motivation**: Human reasoning based on parts ("pointy ears", "serrated tail")

**Architecture Components**:
1. **Convolutional backbone f**: Maps input to feature space
2. **Prototypes P = {p_j}**: Represent prototypical activation patterns
3. **Prototype units g_p_j**: Compute similarity between prototypes and features
4. **Fully connected layer h**: Uses similarity scores for classification

**Training Stages**:
1. **SGD**: Optimize convolutional layers and prototypes
2. **Projection**: Map prototypes to actual training patches
3. **Convex optimization**: Sparsify final layer for positive reasoning

**Loss Function**:
minimize: CE + λ₁·Clst + λ₂·Sep
- CE: Standard cross-entropy loss
- Clst: Cluster cost (encourages prototype-patch similarity)
- Sep: Separation cost (pushes away different-class prototypes)

#### 4.5.2 PIP-Net Improvements

**Problem with ProtoPNet**: Learned similarities don't align with human notions
**Solution**: Self-supervised learning with data augmentation
**Key Insight**: Augmented versions should have similar latent representations

---

## Lecture 5: Model-Agnostic Methods

### 5.1 B-cos Networks

#### 5.1.1 Dynamic Linearity Concept

**Core Idea**: "Optimize model for fixed explanation" instead of "optimize explanation for fixed model"

**Mathematical Framework**:
- Standard DNN: Complex nonlinear function
- B-cos Network: y(x) = W(x)x (dynamic linear)
- W(x): Input-dependent weight matrix

#### 5.1.2 B-cos Transformation

**Formula**: B-cos(x; w) = |cos(x,w)|^(B-1) × ŵᵀx = w^T(x)x

**Properties**:
1. **Dynamic Linear**: Exact summary through matrix multiplication
2. **Alignment Pressure**: Encourages weight-input alignment  
3. **High Compatibility**: Works with existing architectures

**Benefits**:
- Model-inherent explanations
- Faithful by design
- Competitive accuracy with standard networks

### 5.2 Global Model-Agnostic Methods

#### 5.2.1 Surrogate Models

**Definition**: Interpretable model trained to approximate black-box model predictions

**Procedure**:
1. Select dataset X
2. Get black-box predictions for X
3. Choose interpretable model type (linear, tree, etc.)
4. Train surrogate on (X, predictions)
5. Interpret surrogate model

**Advantages**:
- Flexible and intuitive
- Multiple explanation types possible
- Model-agnostic approach

**Disadvantages**:
- Explains model, not data
- Unclear when surrogate is "good enough"
- May not generalize to all data regions

### 5.3 Local Model-Agnostic Methods

#### 5.3.1 LIME (Local Interpretable Model-agnostic Explanations)

**Core Assumption**: Complex models are locally linear

**Process for Images**:
1. Segment image into superpixels
2. Create perturbations by masking superpixels
3. Get model predictions for perturbations
4. Train local linear model weighted by proximity
5. Use linear coefficients as explanation

**Mathematical Formulation**:
minimize: L(f, g, π_x) + Ω(g)
- L: Fidelity loss (weighted by proximity π_x)
- Ω: Complexity penalty (e.g., sparsity)
- g: Local interpretable model

**Advantages**:
- Model-agnostic
- Works for multiple data types
- Sparse, human-friendly explanations

**Disadvantages**:
- Hyperparameter sensitive
- Local linearity assumption may not hold
- Computationally expensive
- Instability issues

#### 5.3.2 RISE (Randomized Input Sampling for Explanation)

**Motivation**: Address LIME limitations
- No linearity assumption
- Continuous importance values
- No superpixel dependency

**Algorithm**:
1. Generate N random masks M_i
2. Compute masked inputs: I ⊙ M_i
3. Get model outputs: f(I ⊙ M_i)
4. Compute importance: S_{I,f}(λ) ≈ (1/E[M]) Σ f(I ⊙ M_i) · M_i(λ)

**Mask Generation**:
1. Sample small binary masks (h×w)
2. Upsample with bilinear interpolation
3. Random cropping for spatial variation
4. Results in smooth, diverse masks

**Advantages over LIME**:
- No linearity assumption
- Continuous attribution values
- No segmentation dependency
- More stable results

**Comparison Properties**:
- LIME: Higher comprehensibility (binary superpixel explanations)
- RISE: Higher fidelity (no linearity assumption)
- RISE: Lower algorithmic complexity (no local model training)

---

## Detailed Q&A Section

### Chapter 1: Introduction to XAI

**Q1: What is the fundamental definition of explainable AI and why is it important?**

**A1**: Explainable AI refers to artificial intelligence systems that are transparent in their operations, allowing human users to understand and trust the decisions made by these systems. According to Tim Miller (2017), "Interpretability is the degree to which a human can understand the cause of a decision."

XAI is important for several critical reasons:
1. **Bias Detection**: AI systems can learn spurious correlations from biased datasets
2. **Legal Compliance**: Regulations like GDPR and EU AI Act require transparency
3. **Trust Building**: Users need to understand AI decisions to trust them
4. **Error Correction**: Understanding failures helps improve models
5. **Risk Management**: Critical applications require explainable decisions

**Q2: Explain the four main arguments for XAI with examples.**

**A2**: The four main arguments for XAI are:

1. **Explain to Justify**: 
   - Purpose: Provide auditable defense of outputs
   - Example: Credit scoring - explaining why someone was denied a loan
   - Importance: Legal compliance and fairness assurance

2. **Explain to Control**:
   - Purpose: Identify and override erroneous predictions
   - Example: Medical diagnosis - detecting when model focuses on wrong features
   - Importance: Risk mitigation and bias identification

3. **Explain to Improve**:
   - Purpose: Enable model enhancement through understanding failures
   - Example: Autonomous driving - understanding why model fails in specific scenarios
   - Importance: Building more robust and generalizable models

4. **Explain to Discover**:
   - Purpose: Uncover previously unknown patterns
   - Example: Drug discovery - finding new molecular relationships
   - Importance: Scientific advancement and knowledge generation

**Q3: What are the main challenges in achieving explainability for deep neural networks?**

**A3**: The main challenges include:

1. **Model Complexity**: DNNs have millions/billions of parameters making direct interpretation impossible

2. **Accuracy vs. Explainability Trade-off**: Simple interpretable models often lack the accuracy of complex models

3. **Approximation Requirements**: True mathematical functions are too complex, requiring approximations that may lose fidelity

4. **Feature Representation**: Unlike tabular data, image features are not inherently meaningful to humans

5. **Evaluation Difficulty**: No ground truth for "correct" explanations makes evaluation challenging

6. **Context Dependency**: Different stakeholders need different types of explanations

### Chapter 2: The Many Faces of Explainability

**Q4: Explain the different types of explanations and their purposes.**

**A4**: There are three main types of explanations:

1. **Human Prior Knowledge Explanations**:
   - Align with human understanding
   - Example: "This is a dog because it has four legs and fur"
   - Purpose: Match human intuition and reasoning

2. **Model-Faithful Explanations**:
   - Represent actual model decision process
   - Example: "This is a dog because pixel (124,124) has high activation"
   - Purpose: Accurately describe model behavior

3. **Task-Specific Explanations**:
   - Answer specific questions about the prediction
   - Example: Visual Question Answering explanations
   - Purpose: Provide contextual understanding

The key insight is that faithful model explanations may not align with human explanations, and this disconnect must be carefully considered.

**Q5: Describe the six agent types in the ML ecosystem and their explanation needs.**

**A5**: The six agent types are:

1. **Creators**: Build ML systems
   - Need: Performance insights and optimization guidance
   - Explanation type: Technical debugging information

2. **Operators**: Directly interact with systems
   - Need: Input validation and output interpretation
   - Explanation type: Operational guidance and uncertainty indicators

3. **Executors**: Make decisions based on ML outputs
   - Need: Decision support and confidence information
   - Explanation type: Actionable insights and alternatives

4. **Decision-subjects**: Affected by ML-informed decisions
   - Need: Justification and fairness assurance
   - Explanation type: Clear reasoning and appeal mechanisms

5. **Data-subjects**: Data used in training
   - Need: Privacy protection and data usage transparency
   - Explanation type: Data influence and privacy impact

6. **Examiners**: Investigate and audit systems
   - Need: Comprehensive system understanding
   - Explanation type: Complete model behavior analysis

**Q6: What are the key properties for evaluating explanation methods and individual explanations?**

**A6**: 

**Method Properties**:
1. **Expressive Power**: The "language" of explanations (rules, attribution maps, etc.)
2. **Translucency**: Level of model access required (high = white-box, low = black-box)
3. **Portability**: Range of models the method can explain
4. **Algorithmic Complexity**: Computational cost of generating explanations

**Individual Explanation Properties**:
1. **Fidelity**: How accurately the explanation represents the model
2. **Stability**: Consistency of explanations for similar instances
3. **Comprehensibility**: How well humans understand the explanations
4. **Certainty**: Whether explanations reflect model confidence
5. **Novelty**: Ability to handle out-of-distribution data
6. **Representativeness**: Scope of coverage (local vs. global)

### Chapter 3: Backpropagation-Based Pixel Attributions

**Q7: Explain the gradient-based attribution methods and their relative advantages.**

**A7**: 

**Vanilla Gradient (Saliency Maps)**:
- Method: Compute ∂f_c(x)/∂x
- Advantage: Simple, fast computation
- Disadvantage: Noisy, gradient saturation issues

**Input × Gradient**:
- Method: x ⊙ (∂f_c(x)/∂x)
- Advantage: Cleaner visualizations, incorporates input magnitude
- Disadvantage: Still suffers from gradient noise

**SmoothGrad**:
- Method: Average gradients over noisy inputs
- Advantage: Reduces noise significantly
- Disadvantage: Increased computational cost (50× more evaluations)

**Integrated Gradients**:
- Method: Integrate gradients along path from baseline to input
- Advantage: Satisfies axioms (sensitivity, implementation invariance)
- Disadvantage: Very expensive (50-200× vanilla gradient)

**𝒳-Gradients (Fast IG)**:
- Method: Use nonnegatively homogeneous networks
- Advantage: Axiomatic properties with Input×Gradient speed
- Disadvantage: Requires model modification (bias removal)

**Q8: Compare CAM and Grad-CAM in terms of applicability and computation.**

**A8**: 

**CAM (Class Activation Mapping)**:
- Requirements: Specific architecture (Conv → GAP → Linear)
- Method: Use final layer weights to weight feature maps
- Applicability: Limited to compatible architectures
- Advantage: Direct, interpretable computation
- Disadvantage: Architecture constraint limits usage

**Grad-CAM**:
- Requirements: Any differentiable CNN
- Method: Use gradients to weight feature maps
- Computation: α^c_k = GAP(∂f_c/∂A^k), then Grad-CAM = ReLU(Σ α^c_k A^k)
- Applicability: Universal for CNNs
- Advantage: Works with any architecture
- Disadvantage: Coarse resolution (requires Guided Grad-CAM for details)

**Key Improvement**: Grad-CAM generalizes CAM by replacing fixed weights with gradient-derived weights, significantly increasing portability.

### Chapter 4: Interpretable Models

**Q9: Compare linear regression and decision trees across all explanation properties.**

**A9**: 

| Property | Linear Regression | Decision Trees |
|----------|------------------|----------------|
| **Expressive Power** | Weighted sum | Decision rules/trees |
| **Translucency** | High | High |
| **Portability** | Low | Low |
| **Algorithmic Complexity** | Low | Low |
| **Fidelity** | High | High |
| **Stability** | High | Low (very sensitive) |
| **Comprehensibility** | High (with few features) | Depends on depth |
| **Certainty** | Low | Low |
| **Novelty** | Low | Low |
| **Representativeness** | Both (global weights, local application) | Both |

**Key Differences**:
- **Interactions**: Linear regression cannot handle feature interactions by default; decision trees naturally capture them
- **Assumptions**: Linear regression has strong distributional assumptions; decision trees are assumption-free
- **Stability**: Linear regression is stable; decision trees are highly unstable to data changes
- **Smoothness**: Linear regression provides smooth predictions; decision trees create step functions

**Q10: Explain the ProtoPNet architecture and training process in detail.**

**A10**: 

**Architecture**:
1. **Convolutional backbone f**: Standard CNN (VGG, ResNet) mapping input to H×W×D features
2. **Prototype layer**: m prototypes P = {p_j} of size 1×1×D representing prototypical patterns
3. **Prototype units g_p_j**: Compute similarity between prototypes and all spatial locations
4. **Global max pooling**: Reduces similarity maps to single prototype activation scores
5. **Fully connected layer h**: Uses prototype activations for final classification

**Training Process**:

**Stage 1 - SGD of early layers**:
- Optimize: CE + λ₁·Clst + λ₂·Sep
- CE: Standard classification loss
- Clst: Encourages each image to have patches close to own-class prototypes
- Sep: Pushes patches away from other-class prototypes
- Result: Meaningful latent space with clustered prototypes

**Stage 2 - Prototype Projection**:
- Project each prototype to nearest real training patch of same class
- Ensures prototypes correspond to actual image patches
- Enables visualization of what each prototype represents

**Stage 3 - Last Layer Optimization**:
- Sparsify final layer connections
- Encourage positive reasoning: "class k because of prototype j"
- Discourage negative reasoning: "class k because NOT prototype j"

**Visualization**: 
- Upsample prototype activation maps to image size
- Show 95% activation region as explanation
- Display corresponding training patch that prototype represents

### Chapter 5: Model-Agnostic Methods

**Q11: Explain the LIME algorithm for image classification step by step.**

**A11**: 

**Step 1: Problem Setup**
- Input: Image I to explain, black-box model f, class c
- Goal: Local linear approximation of f around I

**Step 2: Interpretable Representation**
- Segment image into superpixels using algorithms like SLIC
- Create binary vector z where z_i = 1 if superpixel i is present

**Step 3: Perturbation Generation**
- Generate N random binary vectors z'
- Create perturbed images I' by masking corresponding superpixels
- Typically N = 1000-5000 samples

**Step 4: Model Evaluation**
- Get predictions f(I') for all perturbed images
- Compute proximity weights π_x(z') = exp(-D(z,z')²/σ²)
- D measures distance between original and perturbed instances

**Step 5: Local Model Training**
- Fit sparse linear model: minimize Σ π_x(z')[f(z') - g(z')]² + λ||w||₁
- g(z') = w^T z' (linear model in interpretable space)
- λ controls sparsity (typically want K ≈ 5-10 active features)

**Step 6: Explanation Generation**
- Use linear coefficients w as importance scores
- Positive weights: superpixels supporting the prediction
- Negative weights: superpixels opposing the prediction
- Visualize by highlighting important superpixels

**Q12: Compare LIME and RISE across multiple dimensions.**

**A12**: 

| Dimension | LIME | RISE |
|-----------|------|------|
| **Core Assumption** | Local linearity | None (model-agnostic sampling) |
| **Perturbation Method** | Superpixel masking | Random smooth masks |
| **Output Type** | Binary importance map | Continuous attribution map |
| **Explanation Model** | Local linear regression | Weighted mask averaging |
| **Computational Cost** | High (local model training) | Medium (many forward passes) |
| **Stability** | Low (sensitive to perturbations) | Higher (averaging effect) |
| **Fidelity** | Medium (linearity assumption) | Higher (no model assumptions) |
| **Comprehensibility** | High (binary superpixel explanations) | Medium (continuous heatmaps) |
| **Hyperparameters** | Many (kernel, complexity, segmentation) | Few (mask size, number of samples) |

**Key Advantages**:
- **LIME**: More interpretable binary explanations, works well when linearity holds
- **RISE**: More faithful (no linearity assumption), more stable, fewer hyperparameters

**When to Use**:
- **LIME**: When you need clearly interpretable binary explanations and suspect local linearity
- **RISE**: When you need faithful explanations and don't want to assume local linearity

**Q13: What are the benefits and limitations of B-cos networks?**

**A13**: 

**Benefits**:

1. **Inherent Interpretability**:
   - Built-in explanation through dynamic linearity y(x) = W(x)x
   - No post-hoc approximation needed
   - Explanations are exact model summaries

2. **Alignment Pressure**:
   - Encourages meaningful weight-input alignment
   - Results in more human-interpretable features
   - Class-specific explanations naturally emerge

3. **Competitive Performance**:
   - Minimal accuracy loss compared to standard networks
   - Compatible with existing architectures (CNNs, Transformers)
   - Easy to implement with existing frameworks

4. **Evaluation Advantages**:
   - Performs well on quantitative interpretability metrics
   - More stable explanations across similar inputs
   - Clear mathematical foundation

**Limitations**:

1. **Architecture Modification**:
   - Requires replacing standard operations with B-cos transformations
   - May need retraining existing models
   - Not directly applicable to pre-trained models

2. **Potential Failure Cases**:
   - May learn bias terms through spatial features
   - Explanations might be faithful but not intuitive
   - Requires careful loss function design

3. **Limited Evaluation**:
   - Relatively new approach with limited long-term studies
   - May not work equally well across all domains
   - Needs more comprehensive evaluation

4. **Design Constraints**:
   - Optimization objective influences explanation quality
   - "You get what you ask for" - explanations reflect training objectives
   - May require domain-specific adjustments

**Overall Assessment**: B-cos networks represent a promising approach to inherent interpretability, offering a middle ground between post-hoc explanations and fully interpretable models, but require careful implementation and evaluation.

---
# XAI4CV Exam Preparation Guide
## Practice Questions and Detailed Answers

---

## Section A: Conceptual Questions

### Question 1: Fundamental Concepts (10 points)
**a) Define explainable AI and explain why it has become increasingly important in modern machine learning applications. (5 points)**

**Answer:**
Explainable AI (XAI) refers to artificial intelligence systems that are transparent in their operations, enabling human users to understand and trust the decisions made by these systems. According to Tim Miller (2017), "Interpretability is the degree to which a human can understand the cause of a decision."

XAI has become increasingly important due to:

1. **Regulatory Requirements**: Laws like GDPR and the EU AI Act mandate transparency in AI decision-making
2. **High-Stakes Applications**: Medical diagnosis, autonomous vehicles, and financial services require explainable decisions
3. **Bias Detection**: AI systems can learn harmful biases from training data that need to be identified
4. **Trust Building**: Users need to understand AI decisions to trust and adopt the technology
5. **Error Correction**: Understanding model failures enables improvement and debugging

**b) Explain the trade-off between accuracy and explainability in machine learning models. (5 points)**

**Answer:**
The accuracy-explainability trade-off is a fundamental challenge in machine learning:

**Simple/Interpretable Models**:
- Linear regression, decision trees, rule-based systems
- High explainability: Clear decision boundaries and reasoning
- Lower accuracy: Limited capacity to capture complex patterns
- Example: Linear classifier achieving 70% accuracy but providing clear weight interpretations

**Complex/Black-box Models**:
- Deep neural networks, ensemble methods
- High accuracy: Can capture intricate non-linear relationships
- Low explainability: Millions of parameters make direct interpretation impossible
- Example: Deep CNN achieving 95% accuracy but requiring post-hoc explanation methods

**Middle Ground Approaches**:
- Inherently interpretable neural networks (ProtoPNet, B-cos)
- Model-agnostic explanation methods (LIME, RISE)
- Attempt to balance both requirements with moderate trade-offs

---

### Question 2: Stakeholder Analysis (15 points)
**Describe the six agent types in the machine learning ecosystem according to Tomsett et al. For each agent type, explain their primary goals and what type of explanations they require. (15 points)**

**Answer:**

**1. Creators (Goals: Performance Optimization)**
- Role: Design and build ML systems
- Explanation Needs: Technical debugging information, feature importance analysis, performance bottlenecks
- Example: Data scientists need to understand why their model performs poorly on certain subgroups

**2. Operators (Goals: Correct System Operation)**
- Role: Directly interact with ML systems, provide inputs and receive outputs
- Explanation Needs: Input validation guidance, uncertainty indicators, operational instructions
- Example: Radiologists using AI assistance need to know when the system is uncertain

**3. Executors (Goals: Good Decision Making)**
- Role: Make decisions informed by ML system outputs
- Explanation Needs: Decision support, confidence levels, alternative options
- Example: Judges using risk assessment tools need to understand the reasoning behind recommendations

**4. Decision-subjects (Goals: Understanding Personal Impact)**
- Role: Affected by decisions made by executors based on ML systems
- Explanation Needs: Clear justification, fairness assurance, appeal mechanisms
- Example: Loan applicants need to understand why they were rejected

**5. Data-subjects (Goals: Privacy Protection)**
- Role: Personal data used in training ML systems
- Explanation Needs: Data usage transparency, privacy impact, influence on decisions
- Example: Patients whose medical records were used in training want to understand data usage

**6. Examiners (Goals: System Validation)**
- Role: Investigate, audit, or forensically analyze ML systems
- Explanation Needs: Comprehensive behavior analysis, bias detection, compliance verification
- Example: Regulatory auditors need complete understanding of model behavior across all scenarios

---

## Section B: Technical Methods

### Question 3: Gradient-Based Attribution Methods (20 points)

**a) Explain the mathematical foundation of Vanilla Gradient (Saliency Maps) and derive how it extends from linear models to deep neural networks. (8 points)**

**Answer:**

**Linear Model Foundation:**
For a linear classifier: f_c(x) = w_c^T x + b_c
The importance of feature x_i is directly given by |w_i|, the magnitude of the corresponding weight.

**Extension to Deep Neural Networks:**
For DNNs, f_c(x) is highly non-linear. We use first-order Taylor expansion around input x_0:

f_c(x) ≈ f_c(x_0) + ∇f_c(x_0)^T(x - x_0)

This can be rewritten as:
f_c(x) ≈ w^T x + b

where w = ∇f_c(x_0) = ∂f_c(x_0)/∂x_0

**Interpretation:**
If a small change in pixel x_i causes a large change in the output (large gradient), then that pixel is important for the decision.

**Computation:**
1. Forward pass: compute f_c(x_0)
2. Backward pass: compute ∂f_c(x_0)/∂x_0
3. Optional: take max across color channels

**b) Compare Integrated Gradients with Vanilla Gradients in terms of axioms satisfied and computational complexity. (7 points)**

**Answer:**

**Axioms Comparison:**

*Vanilla Gradients:*
- **Sensitivity**: ❌ Fails due to gradient saturation
- **Implementation Invariance**: ✅ Satisfies for functionally equivalent models
- **Linearity**: ❌ May not satisfy
- **Completeness**: ❌ Does not satisfy

*Integrated Gradients:*
- **Sensitivity**: ✅ Guarantees non-zero attribution for relevant features
- **Implementation Invariance**: ✅ Satisfies by design
- **Linearity**: ✅ Satisfies additivity
- **Completeness**: ✅ Attributions sum to output difference

**Mathematical Formulation:**
IG(x, b, f) = (x - b) ∫₀¹ ∇f(α·x + (1-α)·b) dα

**Computational Complexity:**
- **Vanilla Gradients**: 1 forward + 1 backward pass = O(1)
- **Integrated Gradients**: 50-200 forward + backward passes = O(50-200)

IG is 50-200× more expensive due to numerical integration along the path from baseline to input.

**c) Explain how 𝒳-Gradients achieve the benefits of Integrated Gradients with the computational cost of Vanilla Gradients. (5 points)**

**Answer:**

**Key Insight**: For nonnegatively homogeneous functions, Integrated Gradients reduces to Input × Gradient.

**Nonnegative Homogeneity**: f(αx) = αf(x) for all α ≥ 0

**Mathematical Derivation:**
For homogeneous f and baseline b = 0:
IG(x, 0, f) = x ∫₀¹ ∇f(αx) dα = x ∇f(x)

**Achieving Homogeneity:**
1. Remove bias terms from linear/convolutional layers
2. Use homogeneous activations (ReLU, pooling)
3. Result: 𝒳-networks with f(αx) = αf(x)

**Benefits:**
- ✅ Satisfies IG axioms (sensitivity, implementation invariance)
- ✅ Computational cost of Input × Gradient
- ✅ Small accuracy drop (<1% on ImageNet)

---

### Question 4: Model-Agnostic Methods (25 points)

**a) Describe the LIME algorithm for image classification step by step, including the mathematical formulation of the optimization problem. (12 points)**

**Answer:**

**Step 1: Interpretable Representation**
- Segment image I into superpixels using SLIC algorithm
- Create binary vector z ∈ {0,1}^m where z_i = 1 if superpixel i is present

**Step 2: Perturbation Generation**
- Sample N binary vectors z' randomly
- Generate perturbed images I'(z') by masking superpixels where z'_i = 0
- Typically N = 1000-5000 samples

**Step 3: Local Dataset Creation**
- Evaluate f(I'(z')) for all perturbations
- Compute proximity weights: π_x(z') = exp(-D(z,z')²/σ²)
- D(z,z') measures distance between original and perturbed instances

**Step 4: Local Model Training**
**Mathematical Formulation:**
minimize_{g∈G} L(f, g, π_x) + Ω(g)

where:
- L(f, g, π_x) = Σ π_x(z')[f(z') - g(z')]² (weighted fidelity loss)
- Ω(g) = λ||w||₁ (sparsity penalty)
- g(z') = w^T z' + b (local linear model)
- G = class of linear models

**Step 5: Explanation Generation**
- Extract coefficients w from trained linear model
- Positive weights: superpixels supporting prediction
- Negative weights: superpixels opposing prediction
- Visualize top-K most important superpixels

**b) Compare LIME and RISE across the following dimensions: assumptions, output type, computational complexity, and stability. (8 points)**

**Answer:**

| Dimension | LIME | RISE |
|-----------|------|------|
| **Core Assumptions** | Local linearity around instance | None (purely empirical) |
| **Output Type** | Binary importance (superpixel on/off) | Continuous attribution heatmap |
| **Computational Complexity** | High: O(N·F + K³) for N samples, F features, K active features | Medium: O(N·F) for N masks |
| **Stability** | Low: Sensitive to segmentation and sampling | Higher: Averaging reduces variance |
| **Hyperparameters** | Many: kernel width, complexity K, segmentation | Few: mask size, number of samples |
| **Interpretability** | High: Clear binary explanations | Medium: Continuous but intuitive |
| **Fidelity** | Limited by linearity assumption | Higher: No model assumptions |

**c) Explain the mask generation process in RISE and why it's superior to naive random pixel masking. (5 points)**

**Answer:**

**RISE Mask Generation Process:**

**Step 1: Small Binary Masks**
- Generate N binary masks of size h×w (smaller than image H×W)
- Each element set to 1 with probability p (typically p=0.5)
- Mask space size: 2^(h×w) << 2^(H×W)

**Step 2: Bilinear Upsampling**
- Upsample each mask to size (h+1)C_h × (w+1)C_w
- C_h = H/h, C_w = W/w (scaling factors)
- Creates smooth transitions between masked regions

**Step 3: Random Cropping**
- Crop H×W regions with random offset (0,0) to (C_h, C_w)
- Introduces spatial variation in mask positions
- Final masks have smooth boundaries and diverse coverage

**Why Superior to Naive Pixel Masking:**

1. **Reduced Adversarial Effects**: Smooth masks prevent sharp discontinuities that can cause artifacts
2. **Smaller Search Space**: 2^(h×w) vs 2^(H×W) requires fewer samples for good coverage
3. **Better Statistical Properties**: Smooth masks provide more meaningful perturbations
4. **Computational Efficiency**: Fewer parameters to sample while maintaining diversity

---

## Section C: Interpretable Models

### Question 5: Prototypical Part Networks (20 points)

**a) Explain the architecture and training process of ProtoPNet in detail, including all three training stages. (15 points)**

**Answer:**

**Architecture Components:**

**1. Convolutional Backbone f:**
- Standard CNN (VGG, ResNet) mapping input to features
- Output shape: H × W × D (spatial features with D dimensions)

**2. Prototype Layer:**
- Learn m prototypes P = {p_j}_{j=1}^m of shape 1×1×D
- Each prototype represents prototypical activation pattern
- Prototypes correspond to semantic parts (e.g., bird head, car wheel)

**3. Prototype Units g_{p_j}:**
- Compute similarity between prototype p_j and all spatial locations
- Similarity = exp(-||z - p_j||²) where z ∈ patches(f(x))
- Output: similarity map indicating prototype presence

**4. Global Max Pooling:**
- Reduce similarity map to scalar: max_{h,w} similarity(h,w)
- Captures strongest prototype activation in image

**5. Fully Connected Layer h:**
- Uses prototype activations for final classification
- Weights w_{j,k} connect prototype j to class k

**Training Process:**

**Stage 1: SGD of Early Layers**
*Objective:* minimize CE + λ₁·Clst + λ₂·Sep

- **CE**: Standard cross-entropy classification loss
- **Clst**: Cluster cost = Σᵢ min_{j:p_j∈P_{y_i}} min_{z∈patches(f(x_i))} ||z-p_j||²
  - Encourages each training image to have patches close to own-class prototypes
- **Sep**: Separation cost = -Σᵢ min_{j:p_j∉P_{y_i}} min_{z∈patches(f(x_i))} ||z-p_j||²
  - Pushes training patches away from other-class prototypes

*Result:* Meaningful latent space with clustered prototypes

**Stage 2: Prototype Projection**
*Objective:* For each prototype p_j of class k:
arg min_{z∈Z_j} ||z - p_j||², where Z_j = {z': z' ∈ patches(f(x_i)) ∀i s.t. y_i = k}

- Project each prototype to nearest training patch of same class
- Ensures prototypes correspond to actual image regions
- Enables visualization of what each prototype represents

**Stage 3: Last Layer Optimization**
*Objective:* minimize CE + λ Σₖ Σ_{j:p_j∉P_k} |w_{h,(k,j)}|

- Sparsify connections between prototypes and non-corresponding classes
- Encourage positive reasoning: "class k BECAUSE prototype j"
- Discourage negative reasoning: "class k because NOT prototype j"

*Result:* Sparse weight matrix enabling positive prototype reasoning

**b) What is the main limitation of ProtoPNet and how does PIP-Net address it? (5 points)**

**Answer:**

**Main Limitation of ProtoPNet:**
The similarities learned by ProtoPNet do not align with human notions of similarity. Prototypes that appear similar to the model may not be semantically meaningful to humans, reducing interpretability.

**PIP-Net Solution:**
PIP-Net addresses this through **self-supervised learning with data augmentation**:

1. **Augmentation Consistency**: Two augmented versions of the same image should produce similar latent representations
2. **Contrastive Learning**: Uses techniques similar to SimCLR to learn meaningful representations
3. **Human-Aligned Similarities**: The learned feature space better matches human perception of similarity
4. **Improved Prototypes**: Resulting prototypes are more semantically meaningful and interpretable

**Key Insight**: By encouraging consistency across augmentations, PIP-Net learns representations that are more robust and semantically meaningful, leading to prototypes that humans can better understand and trust.

---

## Section D: Advanced Topics

### Question 6: B-cos Networks (15 points)

**a) Explain the concept of dynamic linearity in B-cos networks and how it enables inherent interpretability. (8 points)**

**Answer:**

**Dynamic Linearity Concept:**
B-cos networks achieve the property: y(x) = W(x)x

Where:
- y(x): Network output
- W(x): Input-dependent weight matrix  
- x: Input

**Contrast with Standard Networks:**
- Standard DNNs: Complex non-linear function y = f(x)
- B-cos Networks: Input-dependent linear function y = W(x)x

**Mathematical Foundation:**
The B-cos transformation: B-cos(x; w) = |cos(x,w)|^(B-1) × ŵᵀx = w^T(x)x

Where:
- w^T(x) = |cos(x,w)|^(B-1) × ŵᵀ (input-dependent weight)
- B: Hyperparameter controlling suppression strength
- ŵ: Normalized weight vector

**Inherent Interpretability:**
1. **Exact Explanations**: W(x) provides exact model summary, not approximation
2. **Faithful by Design**: No post-hoc explanation needed
3. **Input-Specific**: Different inputs get different explanations W(x)
4. **Mathematically Grounded**: Clear interpretation through linear combination

**Dynamic Linear Property:**
For layered networks: W_{1→L}(x₀) = W_L(x_{L-1}) ∘ ... ∘ W_1(x₀)
This provides exact end-to-end explanation through input-dependent weight matrices.

**b) What are the main advantages and potential limitations of B-cos networks? (7 points)**

**Answer:**

**Advantages:**

1. **Inherent Interpretability**:
   - No post-hoc approximation required
   - Explanations are exact model summaries
   - Built-in mathematical foundation for explanations

2. **Alignment Pressure**:
   - Encourages meaningful weight-input alignment
   - Results in more interpretable learned features
   - Natural class-specific explanation emergence

3. **Competitive Performance**:
   - Minimal accuracy loss (<1% on ImageNet)
   - Compatible with existing architectures (CNNs, Transformers)
   - Easy integration with current frameworks

4. **Quantitative Superior Performance**:
   - Better performance on interpretability metrics
   - More stable explanations across similar inputs

**Limitations:**

1. **Architecture Modification Required**:
   - Cannot directly apply to existing pre-trained models
   - Requires replacing standard operations with B-cos transformations
   - May need complete retraining

2. **Potential Failure Cases**:
   - May learn spatial bias terms instead of semantic features
   - Explanations might be faithful but not intuitive to humans
   - Requires careful loss function design

3. **Optimization Dependency**:
   - "You get what you ask for" - explanations reflect training objectives
   - May require domain-specific optimization strategies
   - Limited long-term evaluation across diverse domains

4. **Implementation Complexity**:
   - More complex than standard networks
   - Requires understanding of alignment principles
   - May need hyperparameter tuning for optimal interpretability

---

## Section E: Evaluation and Comparison

### Question 7: Method Comparison Matrix (10 points)
**Create a comprehensive comparison table for the following methods across all explanation properties: Vanilla Gradients, LIME, CAM, and ProtoPNet. Justify your ratings. (10 points)**

**Answer:**

| Property | Vanilla Gradients | LIME | CAM | ProtoPNet |
|----------|------------------|------|-----|-----------|
| **Expressive Power** | Attribution maps | Superpixel importance | Activation maps | Prototype similarity |
| **Translucency** | High (gradients) | Low (black-box) | High (weights) | High (features) |
| **Portability** | High (any differentiable) | High (any model) | Low (specific arch.) | Low (CNN only) |
| **Algorithmic Complexity** | Low (1 pass) | High (N samples + training) | Low (1 pass) | Medium (3 stages) |
| **Fidelity** | Medium (approximation) | Medium (linearity assumption) | High (exact) | High (exact) |
| **Stability** | Low (gradient noise) | Low (sampling variance) | High (deterministic) | High (deterministic) |
| **Comprehensibility** | Medium (pixel-level) | High (superpixel-level) | Medium (region-level) | High (part-based) |
| **Certainty** | Low | Low | Low | Medium |
| **Novelty** | Low | Low | Low | Medium |
| **Representativeness** | Local | Local | Local | Global (prototypes) |

**Justifications:**

**Vanilla Gradients:**
- High portability: Works with any differentiable model
- Low stability: Susceptible to gradient saturation and noise
- Medium fidelity: Linear approximation of non-linear function

**LIME:**
- High comprehensibility: Superpixel explanations align with human perception
- High complexity: Requires perturbation sampling and local model training
- Medium fidelity: Limited by local linearity assumption

**CAM:**
- Low portability: Requires specific architecture (Conv→GAP→Linear)
- High fidelity: Uses actual model weights, no approximation
- High stability: Deterministic based on learned weights

**ProtoPNet:**
- High comprehensibility: Part-based reasoning matches human cognition
- Global representativeness: Prototypes represent general patterns
- Medium certainty: Can indicate prototype confidence levels

---

## Practice Calculation Problems

### Problem 1: Integrated Gradients Computation
Given a simple 1D function f(x) = x³ and baseline b = 0, compute the Integrated Gradients attribution for input x = 2.

**Solution:**
IG(x=2, b=0, f) = (x-b) ∫₀¹ ∇f(αx + (1-α)b) dα
                = 2 ∫₀¹ ∇f(2α) dα
                = 2 ∫₀¹ 3(2α)² dα
                = 2 ∫₀¹ 12α² dα
                = 2 × 12 × [α³/3]₀¹
                = 2 × 12 × 1/3
                = 8

**Verification:** f(2) - f(0) = 8 - 0 = 8 ✓ (Completeness axiom satisfied)

### Problem 2: LIME Optimization
For LIME with 3 superpixels and the following data:
- Original prediction: f(1,1,1) = 0.8
- Perturbations: f(1,1,0)=0.6, f(1,0,1)=0.7, f(0,1,1)=0.5
- All proximity weights = 1
- Sparsity parameter λ = 0.1

Find the optimal linear coefficients.

**Solution:**
Minimize: Σ[f(z') - (w₀ + w₁z₁ + w₂z₂ + w₃z₃)]² + 0.1(|w₁| + |w₂| + |w₃|)

Setting up equations:
- f(1,1,1) = w₀ + w₁ + w₂ + w₃ = 0.8
- f(1,1,0) = w₀ + w₁ + w₂ = 0.6
- f(1,0,1) = w₀ + w₁ + w₃ = 0.7  
- f(0,1,1) = w₀ + w₂ + w₃ = 0.5

Solving:
- w₃ = 0.8 - 0.6 = 0.2
- w₂ = 0.8 - 0.7 = 0.1
- w₁ = 0.8 - 0.5 = 0.3
- w₀ = 0.6 - 0.3 - 0.1 = 0.2

**Answer:** w₀=0.2, w₁=0.3, w₂=0.1, w₃=0.2
**Interpretation:** Superpixel 1 is most important (0.3), followed by superpixel 3 (0.2).

---

## Exam Strategy Tips

### Time Management
- **Reading time**: 10 minutes to understand all questions
- **Conceptual questions**: 30-40% of time
- **Technical derivations**: 40-50% of time  
- **Comparison tables**: 10-20% of time

### Key Formula Sheet
1. **Vanilla Gradient**: ∇f(x) = ∂f(x)/∂x
2. **Input × Gradient**: x ⊙ ∇f(x)
3. **Integrated Gradients**: (x-b) ∫₀¹ ∇f(αx + (1-α)b) dα
4. **SmoothGrad**: (1/n) Σ ∇f(x + N(0,σ²))
5. **Grad-CAM**: ReLU(Σₖ αₖ Aₖ) where αₖ = GAP(∂f/∂Aₖ)
6. **LIME objective**: L(f,g,π) + Ω(g)
7. **RISE**: (1/E[M]) Σ f(I⊙Mᵢ)·Mᵢ

### Common Mistakes to Avoid
1. **Confusing local vs global explanations**
2. **Mixing up gradient saturation with gradient explosion**
3. **Forgetting baseline selection importance in IG**
4. **Incorrectly stating CAM architectural requirements**
5. **Confusing fidelity with comprehensibility**
6. **Not distinguishing between method properties and explanation properties**

### Expected Question Distribution
- **Definitions and concepts**: 25%
- **Method comparisons**: 25%
- **Technical derivations**: 25%
- **Application scenarios**: 15%
- **Evaluation metrics**: 10%
