## Test Case Simple Conversation
- from `src/conversations.ts`

### Results By LLM Model
#### With Qwen 2.5:7b-instruct-q5_k_m

 ```bash
 ts-node src/semanticTurningPointDetector.ts
    [TurningPointDetector] Initialized with config: {
    apiKey: '[REDACTED]',
    classificationModel: 'qwen2.5:7b-instruct-q5_k_m',
    embeddingModel: 'text-embedding-3-large',
    semanticShiftThreshold: 0.35,
    minTokensPerChunk: 512,
    maxTokensPerChunk: 4096,
    maxRecursionDepth: 3,
    onlySignificantTurningPoints: true,
    significanceThreshold: 0.75,
    minMessagesPerChunk: 11,
    maxTurningPoints: 10,
    debug: true,
    endpoint: 'http://localhost:11434/v1',
    complexitySaturationThreshold: 4.5,
    measureConvergence: true
  }
  [TurningPointDetector] Starting turning point detection using ARC/CRA framework for conversation with 72 messages
  [TurningPointDetector] Total conversation tokens: 1551
  [TurningPointDetector] Starting dimensional analysis at n=0
  [TurningPointDetector] Created 7 chunks, avg 222 tokens, avg 10 messages per chunk
  [TurningPointDetector] Dimension 0: Split into 7 chunks
  [TurningPointDetector]  - Dimension 0: Processing chunk 1/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-1 and msg-2 (distance: 0.628, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-2 and msg-3 (distance: 0.937, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-3 and msg-4 (distance: 0.509, complexity: 4.2)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-4 and msg-5 (distance: 0.945, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-5 and msg-6 (distance: 0.890, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-6 and msg-7 (distance: 0.743, complexity: 4.5)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-7 and msg-8 (distance: 0.548, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-8 and msg-9 (distance: 0.882, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-9 and msg-10 (distance: 0.956, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-10 and msg-11 (distance: 0.975, complexity: 4.6)
  [TurningPointDetector]     - Processed in 25.8s, estimated remaining time: 154.8s (14.3% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 2/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-10 and msg-11 (distance: 0.975, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-11 and msg-12 (distance: 0.906, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-12 and msg-13 (distance: 0.982, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-13 and msg-14 (distance: 0.980, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-14 and msg-15 (distance: 0.949, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-15 and msg-16 (distance: 0.960, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-16 and msg-17 (distance: 0.975, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-17 and msg-18 (distance: 0.857, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-18 and msg-19 (distance: 0.895, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-19 and msg-20 (distance: 0.878, complexity: 4.6)
  [TurningPointDetector]     - Processed in 20.1s, estimated remaining time: 114.7s (28.6% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 3/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-19 and msg-20 (distance: 0.878, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-20 and msg-21 (distance: 0.773, complexity: 4.5)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-21 and msg-22 (distance: 0.981, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-22 and msg-23 (distance: 0.984, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-23 and msg-24 (distance: 0.898, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-24 and msg-25 (distance: 0.855, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-25 and msg-26 (distance: 0.948, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-26 and msg-27 (distance: 0.991, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-27 and msg-28 (distance: 0.440, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-28 and msg-29 (distance: 0.899, complexity: 4.6)
  [TurningPointDetector]     - Processed in 20.5s, estimated remaining time: 88.5s (42.9% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 4/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-28 and msg-29 (distance: 0.899, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-29 and msg-30 (distance: 0.980, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-30 and msg-31 (distance: 0.984, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-31 and msg-32 (distance: 0.955, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-32 and msg-33 (distance: 0.906, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-33 and msg-34 (distance: 0.859, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-34 and msg-35 (distance: 0.980, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-35 and msg-36 (distance: 0.966, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-36 and msg-37 (distance: 0.989, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-37 and msg-38 (distance: 0.996, complexity: 4.6)
  [TurningPointDetector]     - Processed in 20.0s, estimated remaining time: 64.7s (57.1% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 5/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-37 and msg-38 (distance: 0.996, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-38 and msg-39 (distance: 0.864, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-39 and msg-40 (distance: 0.936, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-40 and msg-41 (distance: 0.958, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-41 and msg-42 (distance: 0.743, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-42 and msg-43 (distance: 0.759, complexity: 4.5)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-43 and msg-44 (distance: 0.914, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-44 and msg-45 (distance: 0.982, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-45 and msg-46 (distance: 0.789, complexity: 4.5)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-46 and msg-47 (distance: 0.977, complexity: 4.6)
  [TurningPointDetector]     - Processed in 20.5s, estimated remaining time: 42.7s (71.4% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 6/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-46 and msg-47 (distance: 0.977, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-47 and msg-48 (distance: 0.972, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-48 and msg-49 (distance: 0.982, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-49 and msg-50 (distance: 0.960, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-50 and msg-51 (distance: 0.980, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-51 and msg-52 (distance: 0.919, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-52 and msg-53 (distance: 0.840, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-53 and msg-54 (distance: 0.597, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-54 and msg-55 (distance: 0.829, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-55 and msg-56 (distance: 0.761, complexity: 4.5)
  [TurningPointDetector]     - Processed in 22.1s, estimated remaining time: 21.5s (85.7% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 7/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-55 and msg-56 (distance: 0.760, complexity: 4.5)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-56 and msg-57 (distance: 0.962, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-57 and msg-58 (distance: 0.974, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-58 and msg-59 (distance: 0.949, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-59 and msg-60 (distance: 0.967, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-60 and msg-61 (distance: 0.813, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-61 and msg-62 (distance: 0.755, complexity: 4.5)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-62 and msg-63 (distance: 0.923, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-63 and msg-64 (distance: 0.867, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-64 and msg-65 (distance: 0.959, complexity: 4.6)
  [TurningPointDetector]     - Processed in 21.7s, estimated remaining time: 0.0s (100.0% complete)
  [TurningPointDetector] Dimension 0: Found 70 turning points
  [TurningPointDetector] Dimension 0: Merged to 35 turning points
  [TurningPointDetector] Dimension 0: Max complexity = 4.84, saturation threshold = 4.5
  [TurningPointDetector] Dimension 0: ESCALATING to n+1
  [TurningPointDetector] Dimension 0: Escalating to dimension 1
  [TurningPointDetector] Created 13 meta-messages for dimensional expansion: meta-cat-0, meta-cat-1, meta-cat-2, meta-cat-3, meta-cat-4, meta-cat-5, meta-cat-6, meta-cat-7, meta-cat-8, meta-section-0, meta-section-1, meta-section-2, meta-section-3
  [TurningPointDetector] Dimension 0: Created 13 meta-messages for dimension 1
  [TurningPointDetector] Starting dimensional analysis at n=1
  [TurningPointDetector] Created 2 chunks, avg 2785 tokens, avg 7 messages per chunk
  [TurningPointDetector] Dimension 1: Split into 2 chunks
  [TurningPointDetector]  - Dimension 1: Processing chunk 1/2 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages meta-cat-6 and meta-cat-7 (distance: 0.358, complexity: 4.5)
  [TurningPointDetector]     ...Potential turning point detected between messages meta-cat-7 and meta-cat-8 (distance: 0.448, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages meta-cat-8 and meta-section-0 (distance: 0.517, complexity: 4.4)
  [TurningPointDetector]     - Processed in 9.5s, estimated remaining time: 9.5s (50.0% complete)
  [TurningPointDetector]  - Dimension 1: Processing chunk 2/2 (4 messages)
  [TurningPointDetector]     - Processed in 0.5s, estimated remaining time: 0.0s (100.0% complete)
  [TurningPointDetector] Dimension 1: Found 3 turning points
  [TurningPointDetector] Dimension 1: Merged to 2 turning points
  [TurningPointDetector] Dimension 1: Max complexity = 4.53, saturation threshold = 4.5
  [TurningPointDetector] Dimension 1: ESCALATING to n+1
  [TurningPointDetector] Dimension 0 → 1: Escalation resulted in convergence distance: 0.033

  Turning point detection took as MM:SS: 00:02:40 for 1551 tokens in the conversation

  === DETECTED TURNING POINTS (ARC/CRA Framework) ===

  1. memory Discussion and Shift from Insight to Action and Shift to Investigation (Insight)
    Messages: "msg-1" → "msg-11"
    Dimension: n=0
    Complexity Score: 4.79 of 5
    Best indicator message ID: "msg-2"
    Emotion: neutral
    Significance: 0.99
    Keywords: anomalous, composition, quantum anomalies, micro-fractures, gear
    Notable quotes:
  - "Scanning CRG-007 now, Elara. Composition is anomalous—iridium-tungsten alloy showing quantum anomalies."
  - "Micro-fractures detected along unusual crystalline boundaries."

  2. Shift to Action Commitment and Shift to Security Preparation (Decision)
    Messages: "msg-16" → "msg-18"
    Dimension: n=0
    Complexity Score: 4.84 of 5
    Best indicator message ID: "msg-17"
    Emotion: determined
    Significance: 0.96
    Keywords: decided, bypass key, patrol routes, decision, security preparation
    Notable quotes:
  - "**I've decided—we must use the bypass key.**"
  - "I've decided—we must use the bypass key."

  3. issue negative entropy Discussion and Shift to Internal Stress Concerns (Problem)
    Messages: "msg-18" → "msg-21"
    Dimension: n=0
    Complexity Score: 4.80 of 5
    Best indicator message ID: "msg-19"
    Emotion: surprise
    Significance: 1.00
    Keywords: anomaly, pulsed, sensors, critical, negative entropy
    Notable quotes:
  - "Wait—your sensors pulsed oddly. What's happening?"
  - "Anomalous negative entropy surge detected in Orrery core—Omega sector. Separate critical issue emerging."

  4. gear Discussion and Shift to Echo Priority (Insight)
    Messages: "msg-21" → "msg-26"
    Dimension: n=0
    Complexity Score: 4.84 of 5
    Best indicator message ID: "msg-21"
    Emotion: neutral
    Significance: 1.00
    Keywords: echoes, resonance, CRG-007, stress, confirmed
    Notable quotes:
  - "Negative entropy? Could these 'echoes' be causing CRG-007's stress internally?"
  - "Confirmed—echo frequencies precisely match CRG-007’s memory-etching resonance."

  5. safeguards Discussion (Insight)
    Messages: "msg-29" → "msg-32"
    Dimension: n=0
    Complexity Score: 4.64 of 5
    Best indicator message ID: "msg-30"
    Emotion: neutral
    Significance: 0.98
    Keywords: ethics, implications, restoration, hypothesis, intentional
    Notable quotes:
  - "Implications immense; restoration ethics uncertain."
  - "Maybe CRG-007’s damage and star omissions are intentional safeguards post-Vorlag’s incident."

  6. emotional response Discussion (Emotion)
    Messages: "msg-35" → "msg-38"
    Dimension: n=0
    Complexity Score: 4.65 of 5
    Best indicator message ID: "msg-37"
    Emotion: mixed
    Significance: 0.98
    Keywords: fear, curiosity, interference, heart racing, action
    Notable quotes:
  - "My heart races, Silus. Fear mixed with intense curiosity."
  - "My heart races, Silus."

  7. codex discovery Discussion (Topic)
    Messages: "msg-40" → "msg-54"
    Dimension: n=1
    Complexity Score: 4.41 of 5
    Best indicator message ID: "msg-41"
    Emotion: neutral
    Significance: 1.00
    Keywords: codex, located, dais, specifications, harmonic
    Notable quotes:
  - "Codex located on central dais."
  - "Codex specifies a tri-harmonic null field tuned to 3.14, 6.28, and 9.42 petahertz."

  8. technical details Discussion (Insight)
    Messages: "msg-54" → "msg-56"
    Dimension: n=0
    Complexity Score: 4.56 of 5
    Best indicator message ID: "msg-55"
    Emotion: neutral
    Significance: 0.98
    Keywords: tri-harmonic, null field, entropic echoes, calculations, suppression
    Notable quotes:
  - "That matches the dominant frequency of the entropic echoes."
  - "Codex specifies a tri-harmonic null field tuned to 3.14, 6.28, and 9.42 petahertz."

  9. Shift to Technical Execution and Shift to Gear Stabilization (Action)
    Messages: "msg-59" → "msg-61"
    Dimension: n=0
    Complexity Score: 4.83 of 5
    Best indicator message ID: "msg-60"
    Emotion: optimism
    Significance: 0.99
    Keywords: energized, null field, insertion, engaging, ambient vibration
    Notable quotes:
  - "Array energized. Null field will stabilize in 12 seconds."
  - "There’s a shift in ambient vibration—it’s working!"

  10. Shift to Gear Stabilization and Shift to Synchronization and Shift to Mnemonic Alignment (Insight)
    Messages: "msg-61" → "msg-64"
    Dimension: n=0
    Complexity Score: 4.61 of 5
    Best indicator message ID: "msg-62"
    Emotion: optimism
    Significance: 0.98
    Keywords: stabilizing, temporal, gear, harmonizing, alignment
    Notable quotes:
  - "There’s a shift in ambient vibration—it’s working!"
  - "Temporal turbulence decreasing. Gear surface stabilizing."


  === ARC/CRA FRAMEWORK CONVERGENCE ANALYSIS ===

  Iteration 1:
    Dimension: n=1
    Convergence Distance: 0.000
    Dimensional Escalation: No
    Turning Points: 2

  Iteration 2:
    Dimension: n=1
    Convergence Distance: 0.033
    Dimensional Escalation: Yes
    Turning Points: 1

  Results saved to files.
 ```


#### With phi-4-mini-Q5_K_M

  ```bash

  [TurningPointDetector] Initialized with config: {
    apiKey: '[REDACTED]',
    classificationModel: 'phi-4-mini-Q5_K_M:3.8B',
    embeddingModel: 'text-embedding-3-large',
    semanticShiftThreshold: 0.35,
    minTokensPerChunk: 512,
    maxTokensPerChunk: 4096,
    maxRecursionDepth: 3,
    onlySignificantTurningPoints: true,
    significanceThreshold: 0.75,
    minMessagesPerChunk: 11,
    maxTurningPoints: 10,
    debug: true,
    endpoint: 'http://10.3.28.24:7223/v1',
    complexitySaturationThreshold: 4.5,
    measureConvergence: true
  }
  [TurningPointDetector] Starting turning point detection using ARC/CRA framework for conversation with 72 messages
  [TurningPointDetector] Total conversation tokens: 1551
  [TurningPointDetector] Starting dimensional analysis at n=0
  [TurningPointDetector] Created 7 chunks, avg 222 tokens, avg 10 messages per chunk
  [TurningPointDetector] Dimension 0: Split into 7 chunks
  [TurningPointDetector]  - Dimension 0: Processing chunk 1/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-1 and msg-2 (distance: 0.628, complexity: 3.9)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-2 and msg-3 (distance: 0.937, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-3 and msg-4 (distance: 0.509, complexity: 3.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-4 and msg-5 (distance: 0.945, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-5 and msg-6 (distance: 0.890, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-6 and msg-7 (distance: 0.743, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-7 and msg-8 (distance: 0.548, complexity: 4.2)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-8 and msg-9 (distance: 0.881, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-9 and msg-10 (distance: 0.956, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-10 and msg-11 (distance: 0.975, complexity: 4.4)
  [TurningPointDetector]     - Processed in 19.4s, estimated remaining time: 116.3s (14.3% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 2/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-10 and msg-11 (distance: 0.975, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-11 and msg-12 (distance: 0.906, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-12 and msg-13 (distance: 0.982, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-13 and msg-14 (distance: 0.980, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-14 and msg-15 (distance: 0.949, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-15 and msg-16 (distance: 0.960, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-16 and msg-17 (distance: 0.975, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-17 and msg-18 (distance: 0.857, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-18 and msg-19 (distance: 0.895, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-19 and msg-20 (distance: 0.878, complexity: 4.8)
  [TurningPointDetector]     - Processed in 16.9s, estimated remaining time: 90.7s (28.6% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 3/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-19 and msg-20 (distance: 0.878, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-20 and msg-21 (distance: 0.773, complexity: 3.9)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-21 and msg-22 (distance: 0.981, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-22 and msg-23 (distance: 0.984, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-23 and msg-24 (distance: 0.898, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-24 and msg-25 (distance: 0.855, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-25 and msg-26 (distance: 0.948, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-26 and msg-27 (distance: 0.991, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-27 and msg-28 (distance: 0.440, complexity: 3.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-28 and msg-29 (distance: 0.899, complexity: 4.4)
  [TurningPointDetector]     - Processed in 15.9s, estimated remaining time: 69.6s (42.9% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 4/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-28 and msg-29 (distance: 0.899, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-29 and msg-30 (distance: 0.980, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-30 and msg-31 (distance: 0.984, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-31 and msg-32 (distance: 0.955, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-32 and msg-33 (distance: 0.906, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-33 and msg-34 (distance: 0.859, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-34 and msg-35 (distance: 0.980, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-35 and msg-36 (distance: 0.966, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-36 and msg-37 (distance: 0.989, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-37 and msg-38 (distance: 0.996, complexity: 4.4)
  [TurningPointDetector]     - Processed in 15.9s, estimated remaining time: 51.1s (57.1% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 5/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-37 and msg-38 (distance: 0.996, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-38 and msg-39 (distance: 0.864, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-39 and msg-40 (distance: 0.936, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-40 and msg-41 (distance: 0.958, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-41 and msg-42 (distance: 0.743, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-42 and msg-43 (distance: 0.759, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-43 and msg-44 (distance: 0.914, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-44 and msg-45 (distance: 0.982, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-45 and msg-46 (distance: 0.789, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-46 and msg-47 (distance: 0.977, complexity: 4.4)
  [TurningPointDetector]     - Processed in 15.5s, estimated remaining time: 33.4s (71.4% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 6/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-46 and msg-47 (distance: 0.977, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-47 and msg-48 (distance: 0.973, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-48 and msg-49 (distance: 0.981, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-49 and msg-50 (distance: 0.960, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-50 and msg-51 (distance: 0.980, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-51 and msg-52 (distance: 0.919, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-52 and msg-53 (distance: 0.840, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-53 and msg-54 (distance: 0.597, complexity: 4.2)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-54 and msg-55 (distance: 0.829, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-55 and msg-56 (distance: 0.760, complexity: 4.3)
  [TurningPointDetector]     - Processed in 17.2s, estimated remaining time: 16.8s (85.7% complete)
  [TurningPointDetector]  - Dimension 0: Processing chunk 7/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-55 and msg-56 (distance: 0.761, complexity: 3.9)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-56 and msg-57 (distance: 0.962, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-57 and msg-58 (distance: 0.974, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-58 and msg-59 (distance: 0.949, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-59 and msg-60 (distance: 0.967, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-60 and msg-61 (distance: 0.813, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-61 and msg-62 (distance: 0.755, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-62 and msg-63 (distance: 0.922, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-63 and msg-64 (distance: 0.867, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-64 and msg-65 (distance: 0.959, complexity: 4.4)
  [TurningPointDetector]     - Processed in 17.5s, estimated remaining time: 0.0s (100.0% complete)
  [TurningPointDetector] Dimension 0: Found 70 turning points
  [TurningPointDetector] Dimension 0: Merged to 48 turning points
  [TurningPointDetector] Dimension 0: Max complexity = 4.84, saturation threshold = 4.5
  [TurningPointDetector] Dimension 0: ESCALATING to n+1
  [TurningPointDetector] Dimension 0: Escalating to dimension 1
  [TurningPointDetector] Created 22 meta-messages for dimensional expansion: meta-cat-0, meta-cat-1, meta-cat-2, meta-cat-3, meta-cat-4, meta-cat-5, meta-cat-6, meta-cat-7, meta-cat-8, meta-cat-9, meta-cat-10, meta-cat-11, meta-cat-12, meta-cat-13, meta-cat-14, meta-cat-15, meta-cat-16, meta-cat-17, meta-section-0, meta-section-1, meta-section-2, meta-section-3
  [TurningPointDetector] Dimension 0: Created 22 meta-messages for dimension 1
  [TurningPointDetector] Starting dimensional analysis at n=1
  [TurningPointDetector] Created 3 chunks, avg 3237 tokens, avg 7 messages per chunk
  [TurningPointDetector] Dimension 1: Split into 3 chunks
  [TurningPointDetector]  - Dimension 1: Processing chunk 1/3 (11 messages)
  [TurningPointDetector]     - Processed in 1.5s, estimated remaining time: 3.1s (33.3% complete)
  [TurningPointDetector]  - Dimension 1: Processing chunk 2/3 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages meta-cat-12 and meta-cat-13 (distance: 0.375, complexity: 4.1)
  [TurningPointDetector]     ...Potential turning point detected between messages meta-cat-14 and meta-cat-15 (distance: 0.410, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages meta-cat-17 and meta-section-0 (distance: 0.423, complexity: 4.4)
  [TurningPointDetector]     - Processed in 7.1s, estimated remaining time: 4.3s (66.7% complete)
  [TurningPointDetector]  - Dimension 1: Processing chunk 3/3 (4 messages)
  [TurningPointDetector]     - Processed in 0.5s, estimated remaining time: 0.0s (100.0% complete)
  [TurningPointDetector] Dimension 1: Found 3 turning points
  [TurningPointDetector] Dimension 1: Merged to 3 turning points
  [TurningPointDetector] Dimension 1: Max complexity = 4.36, saturation threshold = 4.5
  [TurningPointDetector] Dimension 1: Remaining in current dimension
  [TurningPointDetector] Dimension 0 → 1: Escalation resulted in convergence distance: 0.007

  Turning point detection took as MM:SS: 00:02:07 for 1551 tokens in the conversation

  === DETECTED TURNING POINTS (ARC/CRA Framework) ===

  1. Insight (Other)
    Messages: "msg-8" → "msg-9"
    Dimension: n=0
    Complexity Score: 4.79 of 5
    Best indicator message ID: "msg-8"
    Emotion: neutral
    Significance: 0.90
    Keywords: memory, wear, temporal stresses, cognition
    Notable quotes:
  - ""It's memory, not wear! CRG-007 is recording temporal stresses into its alloy, actively consuming itself through cognition.""

  2. Shift to Urgency and Consequences (Problem, Emotion, Decision)
    Messages: "msg-12" → "msg-13"
    Dimension: n=0
    Complexity Score: 4.84 of 5
    Best indicator message ID: "(problem_decision_emotion)"
    Emotion: fear
    Significance: 0.90
    Keywords: urgent, catastrophic, time-sensitive
    Notable quotes:
  - "**Initiating formal request protocols for Codex access in Noctua Vault, Level 12-Delta.** Warden authorization required; delays likely."
  - "*We don't have that kind of time.* The slippage could cause catastrophic destabilization before the Convergence."

  3. Meta-Reflection and Decision and Action and Shift to Technical Issue (Other)
    Messages: "msg-15" → "msg-19"
    Dimension: n=0
    Complexity Score: 4.84 of 5
    Best indicator message ID: "reflection_on_decision"
    Emotion: neutral
    Significance: 0.96
    Keywords: communication process, perspective, decided, use by-pass key, decision
    Notable quotes:
  - ""Risks understood, but Orrery collapse is worse." - User deciding to use Artificer's bypass key despite known penalties."
  - ""Possessing that key has severe penalties." - Assistant objecting and reflecting on the decision-making process."

  4. insight Discussion (Other)
    Messages: "msg-20" → "msg-25"
    Dimension: n=0
    Complexity Score: 4.84 of 5
    Best indicator message ID: "msg-20"
    Emotion: neutral
    Significance: 0.97
    Keywords: negative entropy, CRG-007, correlation, insightful deduction, memory-etching resonance
    Notable quotes:
  - ""Anomalous negative entropy surge detected in Orrery core—Omega sector." **Separate critical issue emerging.**"
  - ""Negative entropy? Could these 'echoes' be causing CRG-007's stress internally?
  " and "Analyzing correlation...
  Confirmed—echo frequencies precisely match CRG-007’s memory-etching resonance.""

  5. Shift to Technical Issue (Action/Problem)
    Messages: "msg-26" → "msg-36"
    Dimension: n=1
    Complexity Score: 4.14 of 5
    Best indicator message ID: "Shift to Technical Issue"
    Emotion: fear mixed with intense curiosity
    Significance: 1.00
    Keywords: Noctua Vault, bypass key, security, harmonics
    Notable quotes:
  - ""My heart races, Silus." Fear mixed with intense curiosity."
  - "Security mapped. Ward harmonics shifting—key causing interference already."

  6. action Discussion (Other)
    Messages: "msg-36" → "msg-38"
    Dimension: n=0
    Complexity Score: 4.45 of 5
    Best indicator message ID: "msg-36"
    Emotion: anticipation
    Significance: 0.92
    Keywords: fear, intense curiosity, optimal, insertion, timing
    Notable quotes:
  - "**My heart races, Silus.** Fear mixed with intense curiosity."
  - "Optimal insertion timing now."

  7. Meta-Reflection (# meta-reflection Turning Points)
    Messages: "msg-38" → "msg-58"
    Dimension: n=1
    Complexity Score: 4.36 of 5
    Best indicator message ID: "meta-cat-15"
    Emotion: neutral, concern
    Significance: 1.00
    Keywords: interface, warning, chronomantic flux
    Notable quotes:
  - "**Warning: interface will expose you to residual chronomantic flux.** Recommend protective incantation layering."

  8. action Discussion (Action)
    Messages: "msg-59" → "msg-61"
    Dimension: n=0
    Complexity Score: 4.43 of 5
    Best indicator message ID: "Shift to Technical Action"
    Emotion: neutral
    Significance: 0.88
    Keywords: null field, array, energized, stabilize, array energized
    Notable quotes:
  - ""Already weaving defensive runes." - User expressing concern and effort."
  - ""Empathetic framework acknowledged. Array energized... Null field will stabilize in 12 seconds. Prepare for insertion." - Assistant providing assurance and moving forward with the task."

  9. Shift to Technical Status Update (Action)
    Messages: "msg-61" → "msg-13"
    Dimension: n=1
    Complexity Score: 4.36 of 5
    Best indicator message ID: "[1-2]"
    Emotion: neutral
    Significance: 1.00
    Keywords: CRG-007, null field engaging
    Notable quotes:
  - ""CRG-007 seated... aligning spindle housing... null field engaging now." - User, "There’s a shift in ambient vibration—it’s working!""

  10. Insight (insight)
    Messages: "msg-62" → "msg-64"
    Dimension: n=0
    Complexity Score: 4.78 of 5
    Best indicator message ID: "'insight'"
    Emotion: neutral
    Significance: 0.94
    Keywords: harmonizing, recognizes, alignment, emergent, synchronization
    Notable quotes:
  - ""Incredible. It's no longer resisting—it’s... harmonizing.""
  - ""Incredible." It's no longer resisting—it’s... harmonizing."


  === ARC/CRA FRAMEWORK CONVERGENCE ANALYSIS ===

  Iteration 1:
    Dimension: n=1
    Convergence Distance: 0.000
    Dimensional Escalation: No
    Turning Points: 3

  Iteration 2:
    Dimension: n=1
    Convergence Distance: 0.007
    Dimensional Escalation: Yes
    Turning Points: 3

  Results saved to files.
  ```

#### With Gpt-4o

 ```sh
  [TurningPointDetector] Initialized with config: {
    apiKey: '[REDACTED]',
    classificationModel: 'gpt-4o',
    embeddingModel: 'text-embedding-3-large',
    semanticShiftThreshold: 0.35,
    minTokensPerChunk: 512,
    maxTokensPerChunk: 4096,
    maxRecursionDepth: 3,
    onlySignificantTurningPoints: true,
    significanceThreshold: 0.75,
    minMessagesPerChunk: 11,
    maxTurningPoints: 10,
    debug: true,
    endpoint: undefined,
    complexitySaturationThreshold: 4.5,
    measureConvergence: true
  }
  [TurningPointDetector] Starting turning point detection using ARC/CRA framework for conversation with 72 messages
  [TurningPointDetector] Total conversation tokens: 1551
  [TurningPointDetector] Starting dimensional analysis at n=0
  [TurningPointDetector] Created 7 chunks, avg 222 tokens, avg 10 messages per chunk
  [TurningPointDetector] Dimension 0: Split into 7 chunks
  [TurningPointDetector]  - Dimension 0: Processing chunk 1/7 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-1 and msg-2 (distance: 0.628, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-10 and msg-11 (distance: 0.975, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-19 and msg-20 (distance: 0.878, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-2 and msg-3 (distance: 0.937, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-11 and msg-12 (distance: 0.906, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-20 and msg-21 (distance: 0.773, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-21 and msg-22 (distance: 0.981, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-12 and msg-13 (distance: 0.982, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-3 and msg-4 (distance: 0.509, complexity: 4.2)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-13 and msg-14 (distance: 0.980, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-22 and msg-23 (distance: 0.984, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-4 and msg-5 (distance: 0.945, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-23 and msg-24 (distance: 0.898, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-14 and msg-15 (distance: 0.949, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-5 and msg-6 (distance: 0.890, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-24 and msg-25 (distance: 0.855, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-15 and msg-16 (distance: 0.960, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-6 and msg-7 (distance: 0.743, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-25 and msg-26 (distance: 0.949, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-16 and msg-17 (distance: 0.975, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-7 and msg-8 (distance: 0.548, complexity: 4.6)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-26 and msg-27 (distance: 0.991, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-17 and msg-18 (distance: 0.857, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-8 and msg-9 (distance: 0.881, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-27 and msg-28 (distance: 0.440, complexity: 4.2)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-18 and msg-19 (distance: 0.895, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-9 and msg-10 (distance: 0.956, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-19 and msg-20 (distance: 0.878, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-28 and msg-29 (distance: 0.899, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-10 and msg-11 (distance: 0.975, complexity: 4.4)
  [TurningPointDetector]     - Processed in 15.3s, estimated remaining time: 59.8s (42.9% complete)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-28 and msg-29 (distance: 0.899, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-37 and msg-38 (distance: 0.996, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-46 and msg-47 (distance: 0.977, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-38 and msg-39 (distance: 0.864, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-29 and msg-30 (distance: 0.980, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-47 and msg-48 (distance: 0.973, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-30 and msg-31 (distance: 0.984, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-39 and msg-40 (distance: 0.936, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-48 and msg-49 (distance: 0.982, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-40 and msg-41 (distance: 0.958, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-49 and msg-50 (distance: 0.960, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-41 and msg-42 (distance: 0.743, complexity: 3.9)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-31 and msg-32 (distance: 0.955, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-50 and msg-51 (distance: 0.980, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-42 and msg-43 (distance: 0.759, complexity: 3.9)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-51 and msg-52 (distance: 0.919, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-32 and msg-33 (distance: 0.906, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-43 and msg-44 (distance: 0.914, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-33 and msg-34 (distance: 0.859, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-44 and msg-45 (distance: 0.982, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-52 and msg-53 (distance: 0.840, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-45 and msg-46 (distance: 0.789, complexity: 3.5)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-34 and msg-35 (distance: 0.980, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-53 and msg-54 (distance: 0.597, complexity: 4.2)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-54 and msg-55 (distance: 0.829, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-35 and msg-36 (distance: 0.966, complexity: 4.0)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-55 and msg-56 (distance: 0.761, complexity: 3.9)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-46 and msg-47 (distance: 0.977, complexity: 4.8)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-36 and msg-37 (distance: 0.989, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-37 and msg-38 (distance: 0.996, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-55 and msg-56 (distance: 0.761, complexity: 3.9)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-56 and msg-57 (distance: 0.962, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-57 and msg-58 (distance: 0.974, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-58 and msg-59 (distance: 0.949, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-59 and msg-60 (distance: 0.967, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-60 and msg-61 (distance: 0.813, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-61 and msg-62 (distance: 0.755, complexity: 4.3)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-62 and msg-63 (distance: 0.922, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-63 and msg-64 (distance: 0.867, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages msg-64 and msg-65 (distance: 0.959, complexity: 4.4)
  [TurningPointDetector] Dimension 0: Found 70 turning points
  [TurningPointDetector] Dimension 0: Merged to 38 turning points
  [TurningPointDetector] Dimension 0: Max complexity = 4.84, saturation threshold = 4.5
  [TurningPointDetector] Dimension 0: ESCALATING to n+1
  [TurningPointDetector] Dimension 0: Escalating to dimension 1
  [TurningPointDetector] Created 13 meta-messages for dimensional expansion: meta-cat-0, meta-cat-1, meta-cat-2, meta-cat-3, meta-cat-4, meta-cat-5, meta-cat-6, meta-cat-7, meta-cat-8, meta-section-0, meta-section-1, meta-section-2, meta-section-3
  [TurningPointDetector] Dimension 0: Created 13 meta-messages for dimension 1
  [TurningPointDetector] Starting dimensional analysis at n=1
  [TurningPointDetector] Created 2 chunks, avg 2860 tokens, avg 7 messages per chunk
  [TurningPointDetector] Dimension 1: Split into 2 chunks
  [TurningPointDetector]  - Dimension 1: Processing chunk 1/2 (11 messages)
  [TurningPointDetector]     ...Potential turning point detected between messages meta-cat-3 and meta-cat-4 (distance: 0.377, complexity: 4.4)
  [TurningPointDetector]     ...Potential turning point detected between messages meta-cat-8 and meta-section-0 (distance: 0.503, complexity: 3.8)
  [TurningPointDetector]     - Processed in 4.4s, estimated remaining time: 0.0s (100.0% complete)
  [TurningPointDetector] Dimension 1: Found 2 turning points
  [TurningPointDetector] Dimension 1: Merged to 2 turning points
  [TurningPointDetector] Dimension 1: Max complexity = 4.42, saturation threshold = 4.5
  [TurningPointDetector] Dimension 1: Remaining in current dimension
  [TurningPointDetector] Dimension 0 → 1: Escalation resulted in convergence distance: 0.032

  Turning point detection took as MM:SS: 00:00:48 for 1551 tokens in the conversation

  === DETECTED TURNING POINTS (ARC/CRA Framework) ===

  1. adaptive behavior hypothesis Discussion (Insight)
    Messages: "msg-2" → "msg-6"
    Dimension: n=0
    Complexity Score: 4.42 of 5
    Best indicator message ID: "msg-3"
    Emotion: surprise
    Significance: 0.92
    Keywords: phase shifting, temporal stabilization, chronomantic interventions, temporal synchronization, internal management
    Notable quotes:
  - "Exactly! It appears more like phase shifting than mechanical wear. My attempts at temporal stabilization caused a violent backlash."
  - "Backlash registered as chroniton shear. Perhaps the gear’s 'temporal synchronization' is internally managed rather than externally imposed?"

  2. recognition Discussion (Insight)
    Messages: "msg-7" → "msg-10"
    Dimension: n=0
    Complexity Score: 4.79 of 5
    Best indicator message ID: "msg-9"
    Emotion: surprise
    Significance: 0.98
    Keywords: memory, mnemonic alloy, cognition, recognition, temporal stresses
    Notable quotes:
  - "Analyzing Artificer log 77.4—fracture patterns match mnemonic alloy memory-etching effects with 92% certainty."
  - "It's memory, not wear! CRG-007 is recording temporal stresses into its alloy, actively consuming itself through cognition."

  3. Shift to Codex Consultation and Shift to Action and Urgency (Action)
    Messages: "msg-10" → "msg-12"
    Dimension: n=0
    Complexity Score: 4.44 of 5
    Best indicator message ID: "msg-11"
    Emotion: anticipation
    Significance: 0.92
    Keywords: Codex, consultation, stress, absorption, Codex access
    Notable quotes:
  - "Consulting the Codex of Whispering Metals is essential."
  - "Initiating formal request protocols for Codex access in Noctua Vault, Level 12-Delta."

  4. Security Protocol Objection (Objection)
    Messages: "msg-12" → "msg-16"
    Dimension: n=1
    Complexity Score: 4.42 of 5
    Best indicator message ID: "msg-14"
    Emotion: fear
    Significance: 1.00
    Keywords: security, protocol, breach, objection, risk
    Notable quotes:
  - "Unauthorized Vault access breaches critical security protocols. I must formally object."
  - "Risks understood, but Orrery collapse is worse."

  5. identification critical issue Discussion (Problem)
    Messages: "msg-18" → "msg-20"
    Dimension: n=0
    Complexity Score: 4.79 of 5
    Best indicator message ID: "msg-20"
    Emotion: surprise
    Significance: 0.98
    Keywords: anomaly, sensors, issue, surge, anomalous
    Notable quotes:
  - "Wait—your sensors pulsed oddly. What's happening?"
  - "Anomalous negative entropy surge detected in Orrery core—Omega sector. Separate critical issue emerging."

  6. gear's Discussion (Insight)
    Messages: "msg-22" → "msg-25"
    Dimension: n=0
    Complexity Score: 4.44 of 5
    Best indicator message ID: "msg-23"
    Emotion: trust
    Significance: 0.92
    Keywords: confirmation, correlation, echo frequencies, memory-etching, gear
    Notable quotes:
  - "Confirmed—echo frequencies precisely match CRG-007’s memory-etching resonance."
  - "The gear actively absorbs harmful temporal echoes. It's protecting the Orrery internally!"

  7. Ethical Implications of Orrery and Hypothesis on Safeguards and Shift to Restoration Risks (Insight)
    Messages: "msg-29" → "msg-32"
    Dimension: n=0
    Complexity Score: 4.84 of 5
    Best indicator message ID: "msg-30"
    Emotion: surprise
    Significance: 0.95
    Keywords: Orrery, curating timelines, ethics, implications, CRG-007
    Notable quotes:
  - "Could the Orrery itself be curating timelines?"
  - "Implications immense; restoration ethics uncertain."

  8. mission revision Discussion (Decision)
    Messages: "msg-32" → "msg-34"
    Dimension: n=0
    Complexity Score: 4.78 of 5
    Best indicator message ID: "msg-33"
    Emotion: neutral
    Significance: 0.94
    Keywords: revision, objective, restoration, understanding, mission
    Notable quotes:
  - "Our objective needs revision—no blind restoration without understanding."
  - "Revised mission: prioritize comprehensive understanding through the Codex first."

  9. action execution Discussion and Vault Access Achieved (Action)
    Messages: "msg-37" → "msg-39"
    Dimension: n=0
    Complexity Score: 4.78 of 5
    Best indicator message ID: "msg-38"
    Emotion: anticipation
    Significance: 0.98
    Keywords: insertion, key, execution, action, timing
    Notable quotes:
  - "Optimal insertion timing now. Insert key precisely three centimeters, rotate 90 degrees counter-clockwise."
  - "Insert key precisely three centimeters, rotate 90 degrees counter-clockwise."

  10. ethical Discussion (Question)
    Messages: "msg-46" → "msg-48"
    Dimension: n=0
    Complexity Score: 4.84 of 5
    Best indicator message ID: "msg-47"
    Emotion: pessimism
    Significance: 0.95
    Keywords: sentient system, ethical dilemma, self-restriction, right to alter, sentience
    Notable quotes:
  - "Do we have the right to alter or repair a potentially sentient system that intentionally self-restricts?"


  === ARC/CRA FRAMEWORK CONVERGENCE ANALYSIS ===

  Iteration 1:
    Dimension: n=1
    Convergence Distance: 0.000
    Dimensional Escalation: No
    Turning Points: 2

  Iteration 2:
    Dimension: n=1
    Convergence Distance: 0.032
    Dimensional Escalation: Yes
    Turning Points: 1

  Results saved to files.
```


Note: Be sure to review the outputs in the `results` directory, which contains the output from the well-known `Pariah` script.

- A full results markdown view is not provided because the `src/conversationPariah.json` file contains over 200 messages, resulting in multiple pages of logs.    

- For complete logs, see the zipped file `results/output_gpt_pariah_logs.zip`, which includes updated logging score outputs featuring the newly added confidence score for the final set of points returned.
    

