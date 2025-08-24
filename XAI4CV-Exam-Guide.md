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