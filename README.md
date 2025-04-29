# Semantic Turning Point Detector: Detect meaningful shifts, structure conversations, and extract insights from dialogue.

The **Semantic Turning Point Detector** is a lightweight but powerful tool for detecting **semantic turning points** in conversations or textual sequences. It recursively analyzes message chains (dialogues, transcripts, chat logs) and identifies where **key shifts in meaning, topic, or insight** occur. These turning points are crucial for:

- **Conversation segmentation** — breaking down long dialogues into meaningful, coherent sections.  
- **Insight extraction** — detecting where significant moments or topic changes happen in natural conversations.  
- **Dialogue modeling** — preparing structured inputs for downstream AI models or analytics pipelines.  
- **AI reasoning pipelines** — providing higher-level structure to conversational data for further analysis, summarization, or reasoning.  
- **Confidence scoring** — every run returns a **0-to-1 confidence score** that rates how “healthy” (coherent vs. chaotic) the detected structure is.  

## Confidence Score and ways to possibly interpret it

The confidence score is based essentially on cosine similarity applied between embeddings of texts. Thus, the confidence score provides a notion of the semantic distance from a turning point to another, aggregated together. This flattened number, thus represents a quicker angle in assessing, from a limited but useful perspective, the health or confidence of the results. 

| Score | What it Usually Means | Actionable Take-away |
|:-----:|-----------------------|----------------------|
| **0.0 – 0.2** | Almost no semantic movement. Flat or repetitive text. | Segmentation likely not useful. |
| **0.2 – 0.3** | Weak shifts. Some structure, but still bland. | May need larger chunks or lower thresholds. |
| **0.3 – 0.4** | **Good** — clear but gentle turning points. | Acceptable for overviews. |
| **0.4 – 0.6** | **Ideal** — strong, natural conversation flow. | Recommended “sweet-spot” range. |
| **0.6 – 1.0** | Too many jumps → chaotic / fragmented input. | Clean the transcript or lower shift threshold. |

> **Why high > 0.6 is “bad”**  
> A very high score means the embedding distances between consecutive messages are huge – the text keeps veering off topic, so the detector sees “turning points” everywhere. That usually signals noisy, disjoint or machine-generated content, not a well-paced human dialogue.

### Example Use Cases
- Automatically segment chat logs into meaningful sections.
- Detect moments of insight or topic change in long-form interviews.
- Prepare structured datasets for dialogue summarization or reasoning tasks.
- Integrate with LLM workflows to improve response context awareness.

This repository provides a **TypeScript implementation** of the **Adaptive Recursive Convergence (ARC) with Cascading Re-Dimensional Attention (CRA)** framework described in our research paper. Unlike traditional summarization which condenses content, this detector identifies moments where conversations shift in topic, tone, insight, or purpose, demonstrating a practical application of multi-dimensional reasoning.

## Background and Architecture

The Semantic Turning Point Detector is a concrete implementation of the ARC/CRA theoretical framework. It demonstrates how conversation analysis can benefit from dimensional expansion and adaptive complexity management. Key features include:

- **Multi-dimensional analysis** that can escalate from dimension n to n+1 when complexity saturates
- **Complexity classification** using the framework's χ function that maps to {1,2,3,4,5} scale
- **Transition operator Ψ** that determines whether to remain in the current dimension or escalate
- **Contraction-based convergence** that mathematically guarantees stable results
- **Bounded dimensional escalation** that prevents infinite recursion
- **Model-agnostic design** that works across different LLM architectures and sizes

## Example Usage 

```typescript
/**
 * Example function demonstrating how to use the SemanticTurningPointDetector
 * Implements an adaptive approach based on conversation complexity
 */
async function runTurningPointDetectorExample() {
  const thresholdForMinDialogueShift = 24;
  
  // Calculate adaptive recursion depth based on conversation length
  // This directly implements the ARC concept of adaptive dimensional analysis
  const determineRecursiveDepth = (messages: Message[]) => {
    return Math.floor(messages.length / thresholdForMinDialogueShift);
  }

  const startTime = new Date().getTime();

  // Create detector with configuration based on the ARC/CRA framework
  const detector = new SemanticTurningPointDetector({
    apiKey: process.env.OPENAI_API_KEY || '',
    
    // Dynamic configuration based on conversation complexity
    semanticShiftThreshold: 0.5 - (0.05 * determineRecursiveDepth(conversation)),
    minTokensPerChunk: 512,
    maxTokensPerChunk: 4096,
    embeddingModel: "text-embedding-3-large",
    
    // ARC framework: dynamic recursion depth based on conversation complexity
    maxRecursionDepth: Math.min(determineRecursiveDepth(conversation), 5),
    
    onlySignificantTurningPoints: true,
    significanceThreshold: 0.75,
    
    // ARC framework: chunk size scales with complexity
    minMessagesPerChunk: Math.ceil(determineRecursiveDepth(conversation) * 3.5),
    
    // ARC framework: number of turning points scales with conversation length
    maxTurningPoints: Math.max(6, Math.round(conversation.length / 7)),
    
    // CRA framework: explicit complexity saturation threshold for dimensional escalation
    complexitySaturationThreshold: 4.5,
    
    // Enable convergence measurement for ARC analysis
    measureConvergence: true,

    
    
    // classificationModel: 'phi-4-mini-Q5_K_M:3.8B',
    classificationModel:'qwen2.5:7b-instruct-q5_k_m',
    debug: true,
    //ollama
    endpoint: 'http://localhost:11434/v1'
  });

  try {
    // Detect turning points using the ARC/CRA framework
    const tokensInConvoFile = await detector.getMessageArrayTokenCount(conversation);
    const turningPoints = await detector.detectTurningPoints(conversation);
    
    const endTime = new Date().getTime();
    const difference = endTime - startTime;
    const formattedTimeDateDiff = new Date(difference).toISOString().slice(11, 19);
    
    console.log(`\nTurning point detection took as MM:SS: ${formattedTimeDateDiff} for ${tokensInConvoFile} tokens in the conversation`);
    
    // Display results with complexity scores from the ARC framework
    console.log('\n=== DETECTED TURNING POINTS (ARC/CRA Framework) ===\n');
    
    turningPoints.forEach((tp, i) => {
      console.log(`${i + 1}. ${tp.label} (${tp.category})`);
      console.log(`   Messages: "${tp.span.startId}" → "${tp.span.endId}"`);
      console.log(`   Dimension: n=${tp.detectionLevel}`);
      console.log(`   Complexity Score: ${tp.complexityScore.toFixed(2)} of 5`);
      console.log(`   Best indicator message ID: "${tp.best_id}"`);
      console.log(`   Emotion: ${tp.emotionalTone || 'unknown'}`);
      console.log(`   Significance: ${tp.significance.toFixed(2)}`);
      console.log(`   Keywords: ${tp.keywords?.join(', ') || 'none'}`);
      
      if (tp.quotes?.length) {
        console.log(`   Notable quotes:\n${tp.quotes.flatMap(q => `- "${q}"`).join('\n')}`);
      }
      console.log();
    });
    
    // Get and display convergence history to demonstrate the ARC framework
    const convergenceHistory = detector.getConvergenceHistory();
    
    console.log('\n=== ARC/CRA FRAMEWORK CONVERGENCE ANALYSIS ===\n');
    convergenceHistory.forEach((state, i) => {
      console.log(`Iteration ${i + 1}:`);
      console.log(`  Dimension: n=${state.dimension}`);
      console.log(`  Convergence Distance: ${state.distanceMeasure.toFixed(3)}`);
      console.log(`  Dimensional Escalation: ${state.didEscalate ? 'Yes' : 'No'}`);
      console.log(`  Turning Points: ${state.currentTurningPoints.length}`);
      console.log();
    });
    
    // Save turning points to file
    fs.writeJSONSync('results/turningPoints.json', turningPoints, { spaces: 2, encoding: 'utf-8' });
    
    // Also save convergence analysis
    fs.writeJSONSync('results/convergence_analysis.json', convergenceHistory, { spaces: 2, encoding: 'utf-8' });
    
    console.log('Results saved to files.');
  } catch (err) {
    console.error('Error detecting turning points:', err);
  }
}
```

### Installation

- `npm i @gaiaverse/semantic-turning-point-detector`



## Relation to the ARC/CRA Framework

### Adaptive Recursive Convergence (ARC)

The ARC framework posits that complex problems can be solved through iterative refinement at various dimensions, with controlled dimensional escalation when local refinements cannot resolve complexity. Our implementation demonstrates:

1. **Atomic Memory**: Implementation of shared memory via caching to avoid redundant calculations
2. **Local Sub-processes**: Partitioning of conversation into manageable chunks for parallel processing
3. **Complexity Function χ**: Mapping significance to a discrete {1,2,3,4,5} complexity scale
4. **Global Transition Operator Ψ**: Logic for dimensional escalation based on complexity saturation
5. **Contraction Mappings**: Ensuring convergence within each dimension

### Cascading Re-Dimensional Attention (CRA)

CRA provides a mechanism for detecting saturation and determining when dimensional expansion is necessary. Our implementation demonstrates:

1. **Attention-Based Detection**: Using semantic distances and embeddings to identify significant shifts
2. **Dimensional Escalation**: Creating higher-dimensional abstractions (meta-messages) when dimension n saturates
3. **Re-labeling in Higher Dimensions**: Reprocessing information at higher dimensions for more abstract patterns
4. **Bounded Recursion**: Ensuring the system doesn't expand dimensions indefinitely

## Repository Layout

```
semantic-turning-point-detector/
├── README.md                         // This file
├── package.json                      // NPM metadata
├── src/
│   ├── semanticTurningPointDetector.ts  // Main implementation of ARC/CRA framework
│   ├── tokensUtil.ts                    // Utility for token counting
│   └── conversation.ts                  // Sample conversation for testing
├── results/
│   ├── turningPoints.json              // Output of turning point detection
│   └── convergence_analysis.json       // Convergence metrics from the ARC process
└── ...
```

## Usage

Here's how to use the Semantic Turning Point Detector with the ARC/CRA framework:

```typescript
import { SemanticTurningPointDetector, Message } from './src/semanticTurningPointDetector';

// Sample conversation
const conversation: Message[] = [
  { id: 'msg-1', author: 'user', message: 'Hello, I need help with my project.' },
  { id: 'msg-2', author: 'assistant', message: 'I\'d be happy to help! What kind of project are you working on?' },
  // ... more messages
];

// Dynamic configuration based on conversation complexity
const thresholdForMinDialogueShift = 24;
const determineRecursiveDepth = (messages: Message[]) => {
  return Math.floor(messages.length / thresholdForMinDialogueShift);
}

// Create detector with ARC/CRA framework parameters
const detector = new SemanticTurningPointDetector({
  apiKey: process.env.OPENAI_API_KEY,
  
  // Dynamic configuration based on conversation complexity
  semanticShiftThreshold: 0.5 - (0.05 * determineRecursiveDepth(conversation)),
  embeddingModel: "text-embedding-3-large",
  
  // ARC framework: dynamic recursion depth based on conversation complexity
  maxRecursionDepth: Math.min(determineRecursiveDepth(conversation), 5),
  
  // ARC framework: chunk size scales with complexity
  minMessagesPerChunk: Math.ceil(determineRecursiveDepth(conversation) * 3.5),
  
  // CRA framework: complexity saturation threshold for dimensional escalation
  complexitySaturationThreshold: 4.5,
  
  // Enable convergence measurement for ARC analysis
  measureConvergence: true,
});

// Detect turning points using the ARC/CRA framework
async function analyzeConversation() {
  const turningPoints = await detector.detectTurningPoints(conversation);
  console.log('Detected Turning Points:', turningPoints);
  
  // Get convergence history to analyze the ARC process
  const convergenceHistory = detector.getConvergenceHistory();
  console.log('ARC Framework Convergence Analysis:', convergenceHistory);
}

analyzeConversation().catch(console.error);
```

## Key Components

### 1. Complexity Function χ

The paper defines a discrete complexity function χ(x) → {1,2,3,4,5} that determines when dimensional escalation is necessary. In our implementation:

```typescript
// Calculate complexity score (chi function) from significance and semantic distance
private calculateComplexityScore(significance: number, semanticShiftMagnitude: number): number {
  // Maps [0,1] significance to [1,5] complexity range
  let complexity = 1 + significance * 4;
  // Adjust based on semantic shift magnitude
  complexity += (semanticShiftMagnitude - 0.5) * 0.5;
  // Ensure complexity is in [1,5] range
  return Math.max(1, Math.min(5, complexity));
}
```

This function maps continuous significance metrics to the discrete complexity scores defined in the paper.

### 2. Transition Operator Ψ

The transition operator Ψ(x,n) determines whether to remain in dimension n or escalate to dimension n+1:

```typescript
// Implement Transition Operator Ψ from the ARC/CRA framework
const maxComplexity = Math.max(...mergedLocalTurningPoints.map(tp => tp.complexityScore));
const needsDimensionalEscalation = maxComplexity >= this.config.complexitySaturationThreshold;

if (needsDimensionalEscalation) {
  // Create meta-messages from turning points for dimension n+1
  const metaMessages = this.createMetaMessagesFromTurningPoints(mergedLocalTurningPoints, messages);
  // Recursively process in dimension n+1
  return this.multiLayerDetection(metaMessages, dimension + 1);
} else {
  // Remain in current dimension
  return this.filterSignificantTurningPoints(mergedLocalTurningPoints);
}
```

This directly implements the paper's formal definition of Ψ.

### 3. Dimensional Expansion (n → n+1)

When complexity saturates in dimension n, the system creates meta-messages that represent higher-dimensional abstractions:

```typescript
// Create meta-messages from turning points for higher-level analysis
// This implements the dimensional expansion from n to n+1
private createMetaMessagesFromTurningPoints(
  turningPoints: TurningPoint[],
  originalMessages: Message[]
): Message[] {
  // Group turning points by category
  const groupedByCategory: Record<string, TurningPoint[]> = {};
  turningPoints.forEach(tp => {
    const category = tp.category;
    if (!groupedByCategory[category]) {
      groupedByCategory[category] = [];
    }
    groupedByCategory[category].push(tp);
  });
  
  // Create meta-messages (one per category for dimension n+1)
  const metaMessages: Message[] = [];
  
  // Process each category...
  
  return metaMessages;
}
```

### 4. Contraction Mapping for Convergence

The ARC framework guarantees convergence through contraction mappings:

```typescript
// Calculate a difference measure between two states for convergence tracking
private calculateStateDifference(
  state1: TurningPoint[],
  state2: TurningPoint[]
): number {
  if (state1.length === 0 || state2.length === 0) return 1.0;
  
  // Calculate average significance difference
  const avgSignificance1 = state1.reduce((sum, tp) => sum + tp.significance, 0) / state1.length;
  const avgSignificance2 = state2.reduce((sum, tp) => sum + tp.significance, 0) / state2.length;
  
  // Normalize by max possible difference
  return Math.abs(avgSignificance1 - avgSignificance2);
}
```

## Model-Agnostic Performance

One of the key innovations of our framework is its model-agnostic nature. The same implementation works effectively across different LLMs:

| Model | Processing Time | Max Dimension | Turning Points | Max Complexity |
|-------|----------------|---------------|----------------|----------------|
| Qwen 2.5 (7B) | 2:58 | n=2 | 5 | 5.00 |
| Phi-4-mini (3.8B) | 2:07 | n=1 | 10 | 4.84 |
| GPT-4o | 0:48 | n=1 | 10 | 4.84 |

This consistent behavior demonstrates that ARC/CRA captures fundamental principles of recursive convergence and dimensional expansion regardless of model architecture. See the results that can be found in the `results` directory after running the detector, and the provided ones for the model.

To see the results as well from the readme, checkout [README.results.md](README.results.md) for the output of the detector on a sample conversation, for each model used.


## Example Output

Running the detector produces turning points with complexity scores and dimensional information:

```json
{
  "id": "tp-0-8-9",
  "label": "Memory, Not Wear Insight",
  "category": "Insight",
  "span": {
    "startId": "msg-8",
    "endId": "msg-9",
    "startIndex": 8,
    "endIndex": 9
  },
  "semanticShiftMagnitude": 0.881,
  "keywords": ["memory", "wear", "temporal stresses", "cognition"],
  "quotes": ["It's memory, not wear! CRG-007 is recording temporal stresses into its alloy, actively consuming itself through cognition."],
  "emotionalTone": "surprise",
  "detectionLevel": 0,
  "significance": 0.98,
  "complexityScore": 4.79
}
```

The convergence analysis shows how the framework transitions between dimensions:

```json
[
  {
    "dimension": 1,
    "convergenceDistance": 0.032,
    "hasConverged": true,
    "didEscalate": true,
    "turningPoints": 3
  },
  {
    "dimension": 2,
    "convergenceDistance": 0.070,
    "hasConverged": true,
    "didEscalate": true,
    "turningPoints": 2
  }
]
```

## Theoretical Foundations

The implementation is grounded in the mathematical foundations described in our paper:

1. **Banach Fixed-Point Theorem**: Ensures convergence in each dimension through contraction mappings
2. **Complexity-Based Transitions**: Uses a discrete complexity classifier to determine dimensional saturation
3. **Bounded Dimensional Escalation**: Prevents infinite recursion through careful complexity management
4. **Knowledge Graph Embeddings**: Leverages semantic relationships through vector representations
5. **Dynamic Attention Mechanisms**: Identifies significant shifts using attention-like mechanisms

## Conclusion

The Semantic Turning Point Detector demonstrates that the theoretical ARC/CRA framework can be successfully implemented in practice. By combining local refinements with dimensional escalation triggered by complexity saturation, we achieve a system that adaptively processes conversations at the appropriate level of abstraction.

This implementation validates the core claims of our paper:
- Adaptive recursion can be achieved through complexity-based dimensional transitions
- Formal convergence is guaranteed through contraction mappings
- Dimensional escalation is bounded and occurs only when necessary
- The framework is model-agnostic and works across different LLM architectures

## References

- "Adaptive Recursive Convergence (ARC) with Cascading Re-Dimensional Attention (CRA) for Multi-Step Reasoning and Dynamic AI Systems" - Ziping Liu, et al. (TBD)