# Semantic Turning Point Detector

This repository provides a **TypeScript-based implementation** of a conversation analysis system that **identifies and classifies “Turning Points”** in dialogue. Unlike traditional summarization—where you compress content into an abstract—this approach pinpoints key **semantic shifts**: moments where the **topic**, **tone**, **insight**, **decision**, or **purpose** changes in a meaningful way.

The code draws on the **recursive** and **adaptive** ideas outlined in the research paper’s **ARC/CRA** framework. Specifically:

- **ARC (Adaptive Recursive Convergence)** is reflected in how the detector **recursively** identifies turning points at multiple “levels,” combining them into higher-level patterns.
- **CRA (Causal-Relational Abstraction)** underlies the classification of these shifts—where we see not just mechanical transitions but also **causal and relational** significance, merging data-driven embeddings with classification to interpret *why* a shift matters.

Below is an overview of the code and how each part connects to these research concepts.

---

## Contents

- [Semantic Turning Point Detector](#semantic-turning-point-detector)
  - [Contents](#contents)
  - [Core Idea: Semantic Turning Points](#core-idea-semantic-turning-points)
  - [Relation to the Research Paper (ARC/CRA)](#relation-to-the-research-paper-arccra)
  - [Repository Layout](#repository-layout)
  - [Usage](#usage)
  - [Key Implementation Details](#key-implementation-details)
    - [1. The `Message` and `TurningPoint` Interfaces](#1-the-message-and-turningpoint-interfaces)
    - [2. Configurable Parameters](#2-configurable-parameters)
    - [3. Multi-Level Detection](#3-multi-level-detection)
    - [4. Embedding Generation \& Chunking](#4-embedding-generation--chunking)
    - [5. Classification via LLM](#5-classification-via-llm)
    - [6. Merging and Filtering Turning Points](#6-merging-and-filtering-turning-points)
  - [Example](#example)
  - [Notes on Token Counting and Performance](#notes-on-token-counting-and-performance)

---

## Core Idea: Semantic Turning Points

Instead of “summarizing” entire conversations, the system locates discrete **turning points**:
- Distinct changes in **topic** (“We were discussing budget, now we’re talking technology”)
- **Insight** or realization (“We just discovered the root cause of a bug”)
- Shifts in **emotional tone** or **motivation** (“We were neutral, now someone is upset or elated”)
- **Decisions** or **objections** that steer the conversation’s direction
- **Meta-reflections** where participants discuss the conversation itself
- **Question** or **Problem** that reframes the prior assumptions

By combining **semantic embeddings** with an **LLM-based classifier**, we detect these changes and attach a short label, category, emotional tone, keywords, and an approximate “significance” rating.

---

## Relation to the Research Paper (ARC/CRA)

1. **Adaptive Recursive Convergence (ARC)**  
   The code **recursively** processes conversation chunks and aggregates turning points at multiple levels. Each level outputs “meta-messages” summarizing local shifts, which are then *re-embedded* and reanalyzed. This method parallels **ARC**’s iterative sweeps and synergy across partial solutions.

2. **Causal-Relational Abstraction (CRA)**  
   CRA posits that meaningful structure emerges from **relations**—not just superficial changes. Our classification step encourages the LLM to interpret the *causal significance* of a shift (e.g., an “Objection” can derail a decision process). We focus on “why” the shift matters, not just “when.”

Hence, the system is a practical demonstration of **ARC**’s multi-level recurrences and **CRA**’s interpretive lens on relationship-based meaning.

---

## Repository Layout

```
semantic-turning-point-detector/
├── README.md               // This file
├── package.json            // NPM metadata (if applicable)
├── src/
│   ├── semanticTurningPointDetector.ts  // Main detection + recursion + classification
│   ├── tokensUtil.ts                   // Utility function for token counting
│   └── ...
└── ...
```

---

## Usage

Below is a basic example of how you might integrate the **SemanticTurningPointDetector** into a project:

```ts
import { SemanticTurningPointDetector, Message } from './src/semanticTurningPointDetector';

const messages: Message[] = [
  { id: 'msg-1', author: 'user', message: "Hi, I'm facing issues with our new API." },
  { id: 'msg-2', author: 'assistant', message: "Can you clarify what part is not working?" },
  // ...
];

// Instantiate with custom settings
const detector = new SemanticTurningPointDetector({
  apiKey: 'YOUR_OPENAI_KEY',
  classificationModel: 'gpt-4',
  embeddingModel: 'text-embedding-ada-002',
  debug: true,
  maxRecursionDepth: 2,
  // etc...
});

(async () => {
  const turningPoints = await detector.detectTurningPoints(messages);
  console.log(turningPoints);
})();
```

Running this processes your `messages` array, identifies turning points, and returns an array of labeled **TurningPoint** objects.

---

## Key Implementation Details

### 1. The `Message` and `TurningPoint` Interfaces

- **`Message`**: Each piece of conversation has an `id`, an `author`, and a `message` body (text). 
  - The optional `spanData` helps track references if a message is a “meta-message” at higher levels.
- **`TurningPoint`**: The result object for each semantic shift. Includes:
  - `label` (short human description)
  - `category` (one of `Topic`, `Insight`, `Emotion`, `Decision`, etc.)
  - `semanticShiftMagnitude` (numeric measure of how big the shift was)
  - `significance` (overall importance, 0-1)
  - `keywords`, `quotes`, `emotionalTone`, etc.
  - `detectionLevel` indicates which “layer” of recursion found this shift.

These data structures reflect **CRA** by capturing both local semantics (the actual shift) and relational context (the cause-effect or rhetorical meaning).

### 2. Configurable Parameters

You can control the system’s sensitivity and depth via constructor options:

- **`semanticShiftThreshold`**: If the cosine distance (with a sigmoid adjustment) between two adjacent messages exceeds this value, we flag a potential turning point.
- **`maxRecursionDepth`**: The number of hierarchical layers for detection.
- **`onlySignificantTurningPoints`** + **`significanceThreshold`**: Filter out less important shifts.
- **`minTokensPerChunk`**, **`maxTokensPerChunk`**, **`minMessagesPerChunk`**: Govern how big each chunk is, balancing cost/performance with contextual accuracy.
- **`endpoint`**: Optionally point to a self-hosted or alternative LLM endpoint.

### 3. Multi-Level Detection

**ARC** principles appear here:
- We chunk the conversation into segments and detect local turning points (“level 0”).
- We then create **meta-messages** describing those turning points and feed them back into the detector, ascending recursion levels until `maxRecursionDepth` is reached.
- This strategy allows “higher-level arcs” to emerge from aggregated turning points. The code merges them and prunes duplicates or overlapping ones.

### 4. Embedding Generation & Chunking

**CRA** starts with local relationships—here:
- We compute embeddings for each message (via OpenAI’s embeddings endpoint or a custom function).
- **Chunking** ensures we handle large conversations in increments without hitting token limits. 
- The chunking logic merges small segments or splits large ones so each portion is within configured size constraints. This step is also reminiscent of **ARC** chunking tasks: break the conversation into smaller sub-problems, then unify the results.

### 5. Classification via LLM

Once we spot a big semantic jump between two messages, the system calls a **classification model** to label it. This uses a “system prompt” explaining the categories, significance scoring, emotional tone, etc. That helps the model interpret the nature of each turning point. It’s essentially applying **CRA**: an LLM “reasoning” about *why* a shift matters.

### 6. Merging and Filtering Turning Points

The system merges overlapping or similar turning points. If two shifts occur close together with the same category (e.g., repeated “Emotion” or “Objection”), we unify them. Afterwards:
- We **boost** higher-level turning points and combine them with local ones.
- We filter them down to keep only the most “significant,” or as many as `maxTurningPoints`.

This final pass yields a concise set of turning points that highlight the conversation’s core changes—precisely the hallmark of **ARC**: the synergy of partial results (local turning points) consolidated into a final integrated solution.

---

## Example

A more in-depth sample can be found at the bottom of [`semanticTurningPointDetector.ts`](./src/semanticTurningPointDetector.ts). It demonstrates a fictional “fantasy-lore” conversation with multiple re-framings and changes in tone, culminating in detection of 10+ turning points. 

Running that example:

```bash
# If using ts-node
ts-node src/semanticTurningPointDetector.ts
```

You’ll see logs about chunk creation, recursion levels, and final turning points. The system writes them to `turningPoints.json`.

---

## Notes on Token Counting and Performance

1. **`countTokens`**: We use a custom or third-party tokenizer to estimate how many tokens a message requires. This ensures we neither overflow the LLM’s context window nor create unnecessarily large chunks.
2. **LRU caching** helps skip re-counting tokens for identical strings, improving speed in repeated analyses.
3. **Embedding** calls can be expensive if your conversation is large. The system chunking logic attempts to minimize overhead by grouping messages. 
4. **Rate Limits**: If using an external endpoint or OpenAI, large conversations with deep recursion can hit rate or cost limits. Adjust concurrency (`eachOfLimit`) in the code or reduce `maxRecursionDepth` and `minTokensPerChunk` to mitigate costs.

---

**In summary,** this module serves as a **practical demonstration** of key **ARC** and **CRA** ideas—recursive chunk-based synergy and interpretive classification of emergent structures. By analyzing a conversation’s embedded meaning and bridging it with an LLM’s classification capacity, the **Semantic Turning Point Detector** moves beyond flattening data, offering a method to identify conversation-defining changes that shape how dialogues evolve.