// file: semanticTurningPointDetector.ts
import fs from 'fs-extra';
/*****************************************************************************************
 * SEMANTIC TURNING POINT DETECTOR
 *
 * A TypeScript implementation of the Adaptive Recursive Convergence (ARC) with
 * Cascading Re-Dimensional Attention (CRA) framework for conversation analysis.
 * 
 * This detector identifies semantic "Turning Points" in conversations as a concrete
 * application of the ARC/CRA theoretical framework for multi-step reasoning 
 * and dynamic dimensional expansion.
 * 
 * Framework implementation:
 * 1. Analyze semantic relationships between messages using embeddings (dimension n)
 * 2. Calculate semantic distances that correspond to the contraction mapping
 * 3. Apply the complexity function χ to determine dimensional saturation
 * 4. Use the transition operator Ψ to determine whether to stay in dimension n or escalate
 * 5. Employ meta-messages and recursive analysis for dimensional expansion (n → n+1)
 * 6. Merge and prune results to demonstrate formal convergence
 *****************************************************************************************/

import async from 'async';
import { OpenAI } from 'openai';
import { LRUCache } from 'lru-cache';
import crypto from 'crypto';
import { countTokens } from './tokensUtil';
import { conversation } from './conversation';
import { ResponseFormatJSONSchema } from 'openai/resources/shared';

// Cache for token counts to avoid recalculating - implements atomic memory concept
const tokenCountCache = new LRUCache<string, number>({
  max: 10000,
  ttl: 1000 * 60 * 60 * 24
});

// -----------------------------------------------------------------------------
// Embedding Generation
// -----------------------------------------------------------------------------

/**
 * Generates an embedding for a given text using the OpenAI API
 * This provides the vector representation for semantic distance calculation
 */
async function generateEmbedding(
  text: string,
  model?: string
): Promise<Float32Array> {
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const response = await openai.embeddings.create({
    model: model,
    input: text,
    encoding_format: "float",
  });
  return new Float32Array(response.data?.[0].embedding);
}

// -----------------------------------------------------------------------------
// Core Interfaces
// -----------------------------------------------------------------------------

/**
 * Represents a single message in a conversation
 */
export interface Message {
  /** Unique identifier for this message or meta-abstraction */
  id: string;
  /** The sender of the message (e.g., "user", "assistant") */
  author: string;
  /** The message content */
  message: string;
  /** Span data for meta-messages (used in recursive analysis) */
  spanData?: MessageSpan;
}

/**
 * Message span identifies a range of messages
 * Used for tracking dimensional representations across recursion levels
 */
interface MessageSpan {
  /** Start message ID */
  startId: string;
  /** End message ID */
  endId: string;
  /** Start index in the original message array */
  startIndex: number;
  /** End index in the original message array */
  endIndex: number;
  /** Original message span if this is a meta-message span */
  originalSpan?: MessageSpan;
}

/**
 * Represents a semantic turning point in a conversation
 * This corresponds to a significant semantic shift detected by the system
 */
export interface TurningPoint {
  /** Unique identifier for this turning point */
  id: string;
  /** Human-readable short description of what this turning point represents */
  label: string;
  /** The type of semantic shift this turning point represents */
  category: TurningPointCategory;
  /** The span of messages covered by this turning point */
  span: MessageSpan;
  /** Legacy span format for backward compatibility */
  deprecatedSpan: {
    startIndex: number;
    endIndex: number;
    startMessageId: string;
    endMessageId: string;
  };
  /** The semantic distance/shift that triggered this turning point */
  semanticShiftMagnitude: number;
  /** Key terms that characterize this turning point */
  keywords?: string[];
  /** Notable quotes from the messages in this turning point's span */
  quotes?: string[];
  /** The emotional tone of this turning point if applicable */
  emotionalTone?: string;
  /** The level at which this turning point was detected (dimension n) */
  detectionLevel: number;
  /** Significance score (higher = more significant) */
  significance: number;
  /** A suggested different point of the message in which the shift should start */
  best_start_id: string;
  /** A suggested different point of the message in which the shift should end */
  best_end_id: string;
  /** An assessed best point representing the turning point */
  best_id: string;
  /** The complexity score (1-5) representing saturation in current dimension */
  complexityScore: number;
}

/**
 * Categories of turning points
 */
export type TurningPointCategory =
  | 'Topic'           // A shift to a new subject
  | 'Insight'         // A realization or discovery
  | 'Emotion'         // An emotional shift or response
  | 'Meta-Reflection' // Thinking about the conversation itself
  | 'Decision'        // A choice or commitment being made
  | 'Question'        // A significant question being posed
  | 'Problem'         // Identification of an issue or challenge
  | 'Action'          // A commitment to do something
  | 'Clarification'   // Clearing up a misunderstanding
  | 'Objection'       // Disagreement or pushback
  | 'Other';          // Any other type of shift

/**
 * Configuration options for the turning point detector
 */
export interface TurningPointDetectorConfig {
  /** OpenAI API key */
  apiKey: string;
  /** Model for turning point classification */
  classificationModel: string;
  /** Model for generating embeddings */
  embeddingModel: string;
  /** Semantic shift threshold for detecting potential turning points */
  semanticShiftThreshold: number;
  /** Minimum tokens per chunk when processing conversation */
  minTokensPerChunk: number;
  /** Maximum tokens per chunk */
  maxTokensPerChunk: number;
  /** Maximum recursive depth (dimensional expansion limit) */
  maxRecursionDepth: number;
  /** Whether to filter by significance */
  onlySignificantTurningPoints: boolean;
  /** Significance threshold for filtering */
  significanceThreshold: number;
  /** Minimum messages per chunk */
  minMessagesPerChunk: number;
  /** Maximum turning points in final results */
  maxTurningPoints: number;
  /** Enable verbose logging */
  debug: boolean;
  /** Custom OpenAI API endpoint (optional) */
  endpoint?: string;
  /** Complexity saturation threshold (dimension escalation trigger) */
  complexitySaturationThreshold: number;
  /** Enable convergence measurement across iterations */
  measureConvergence: boolean;
}

/**
 * Chunking result with message segments and metrics
 */
interface ChunkingResult {
  /** Array of message chunks */
  chunks: Message[][];
  /** Total number of chunks created */
  numChunks: number;
  /** Average tokens per chunk */
  avgTokensPerChunk: number;
}

/**
 * Embedding with associated message data
 */
interface MessageEmbedding {
  /** The message ID */
  id: string;
  /** The message index in original array */
  index: number;
  /** The embedding vector */
  embedding: Float32Array;
}

/**
 * Tracks state changes across iteration for convergence measurement
 */
interface ConvergenceState {
  /** Previous state turning points */
  previousTurningPoints: TurningPoint[];
  /** Current state turning points */
  currentTurningPoints: TurningPoint[];
  /** Current dimension */
  dimension: number;
  /** Convergence measure between states (lower = more converged) */
  distanceMeasure: number;
  /** Whether the state has converged */
  hasConverged: boolean;
  /** Whether dimension escalation occurred */
  didEscalate: boolean;
}

// -----------------------------------------------------------------------------
// Main Detector Class
// -----------------------------------------------------------------------------

export class SemanticTurningPointDetector {
  private config: TurningPointDetectorConfig;
  private openai: OpenAI;
  private originalMessages: Message[] = [];
  private convergenceHistory: ConvergenceState[] = [];

  /**
   * Creates a new instance of the semantic turning point detector
   */
  constructor(config: Partial<TurningPointDetectorConfig> = {}) {
    // Default configuration
    this.config = {
      apiKey: config.apiKey || process.env.OPENAI_API_KEY || '',
      classificationModel: config.classificationModel || 'gpt-4o-mini',
      embeddingModel: config.embeddingModel || 'text-embedding-3-small',
      semanticShiftThreshold: config.semanticShiftThreshold || 0.22,
      minTokensPerChunk: config.minTokensPerChunk || 250,
      maxTokensPerChunk: config.maxTokensPerChunk || 2000,
      maxRecursionDepth: config.maxRecursionDepth || 3,
      onlySignificantTurningPoints: config.onlySignificantTurningPoints ?? true,
      significanceThreshold: config.significanceThreshold || 0.5,
      minMessagesPerChunk: config.minMessagesPerChunk || 3,
      maxTurningPoints: config.maxTurningPoints || 5,
      debug: config.debug || false,
      endpoint: config.endpoint,
      // New ARC/CRA specific parameters
      complexitySaturationThreshold: config.complexitySaturationThreshold || 4.5,
      measureConvergence: config.measureConvergence ?? true
    };

    // Initialize OpenAI client
    this.openai = new OpenAI({
      apiKey: this.config.apiKey,
      baseURL: this.config.endpoint
    });

    if (this.config.debug) {
      console.log('[TurningPointDetector] Initialized with config:', {
        ...this.config,
        apiKey: '[REDACTED]'
      });
    }
  }

  /**
   * Main entry point: Detect turning points in a conversation
   * Implements the full ARC/CRA framework
   */
  public async detectTurningPoints(messages: Message[]): Promise<TurningPoint[]> {
    this.log('Starting turning point detection using ARC/CRA framework for conversation with', messages.length, 'messages');
    this.convergenceHistory = [];

    // Store original messages for reference
    const totalTokens = await this.getMessageArrayTokenCount(messages);
    this.log(`Total conversation tokens: ${totalTokens}`);
    this.originalMessages = messages;

    // Begin dimensional analysis at level 0
    return this.multiLayerDetection(messages, 0);
  }

  /**
   * Multi-layer detection implementing the ARC/CRA dimensional processing
   * This is the primary implementation of the transition operator Ψ
   */
  private async multiLayerDetection(
    messages: Message[],
    dimension: number
  ): Promise<TurningPoint[]> {
    this.log(`Starting dimensional analysis at n=${dimension}`);
    
    // Check recursion depth - hard limit on dimensional expansion
    if (dimension >= this.config.maxRecursionDepth) {
      this.log(`Maximum dimension (n=${dimension}) reached, processing directly without further expansion`);
      return this.detectTurningPointsInChunk(messages, dimension);
    }

    // For very small conversations (or at deeper levels), use sliding window
    let localTurningPoints: TurningPoint[] = [];

    if (messages.length <= 10 && dimension === 0) {
      this.log(`Dimension ${dimension}: Small conversation, using sliding window analysis`);
      // Temporarily increase the threshold for small conversations
      const originalThreshold = this.config.semanticShiftThreshold;
      this.config.semanticShiftThreshold = Math.max(0.4, originalThreshold);

      // Process directly
      localTurningPoints = await this.detectTurningPointsInChunk(messages, dimension);
      
      // Restore config
      this.config.semanticShiftThreshold = originalThreshold;
    } else {
      // Chunk the conversation
      const { chunks } = await this.chunkConversation(messages, dimension);
      this.log(`Dimension ${dimension}: Split into ${chunks.length} chunks`);

      // Process each chunk in parallel to find local turning points
      const chunkTurningPoints: TurningPoint[][] = new Array(chunks.length);
      const durationsSeconds: number[] = new Array(chunks.length).fill(-1);
      const limit = this.config.endpoint ? 1 : 5; // Limit API calls to avoid rate limits, set to 1 for local ollama endpoint
      
      await async.eachOfLimit(
        chunks,
        limit,
        async (chunk, indexStr) => {
          const index = Number(indexStr);
          const startTime = Date.now();
          
          if (index % 10 === 0 || limit === 1) {
            this.log(` - Dimension ${dimension}: Processing chunk ${index + 1}/${chunks.length} (${chunk.length} messages)`);
          }
          
          chunkTurningPoints[index] = await this.detectTurningPointsInChunk(chunk, dimension);
          const durationSecs = (Date.now() - startTime) / 1000;
          durationsSeconds[index] = durationSecs;

          if (index % 10 === 0 || limit === 1) {
            const averageDuration = durationsSeconds.filter(d => d > 0).reduce((a, b) => a + b, 0) / 
                                  durationsSeconds.filter(d => d > 0).length;
            const remainingChunks = durationsSeconds.length - 
                                   durationsSeconds.filter(d => d > 0).length;
            const remainingTime = (averageDuration * remainingChunks).toFixed(1);
            const percentageComplete = 100 - (remainingChunks / durationsSeconds.length * 100);
            
            this.log(`    - Processed in ${durationSecs.toFixed(1)}s, estimated remaining time: ${remainingTime}s (${percentageComplete.toFixed(1)}% complete)`);
          }
        }
      );

      // Flatten all turning points from all chunks
      localTurningPoints = chunkTurningPoints.flat();
    }

    this.log(`Dimension ${dimension}: Found ${localTurningPoints.length} turning points`);

    // If we found zero or one turning point at this level, return it directly
    if (localTurningPoints.length <= 1) {
      return localTurningPoints;
    }

    // First merge any similar turning points at this level
    const mergedLocalTurningPoints = this.mergeSimilarTurningPoints(localTurningPoints);
    this.log(`Dimension ${dimension}: Merged to ${mergedLocalTurningPoints.length} turning points`);

    // ------------------- CRITICAL ARC/CRA IMPLEMENTATION -------------------
    // This is where we determine whether to expand dimension based on complexity saturation
    
    // Calculate the maximum complexity in this dimension to determine if we need to escalate
    const maxComplexity = Math.max(...mergedLocalTurningPoints.map(tp => tp.complexityScore));
    
    // Implement Transition Operator Ψ from the ARC/CRA framework
    // If complexity reaches saturation threshold, we need to escalate to dimension n+1
    const needsDimensionalEscalation = maxComplexity >= this.config.complexitySaturationThreshold;
    
    // Log the decision point - this is the crucial Ψ operation from the paper
    this.log(`Dimension ${dimension}: Max complexity = ${maxComplexity.toFixed(2)}, saturation threshold = ${this.config.complexitySaturationThreshold}`);
    this.log(`Dimension ${dimension}: ${needsDimensionalEscalation ? 'ESCALATING to n+1' : 'Remaining in current dimension'}`);

    // If we're at the last recursion level, or have few turning points, or don't need escalation
    if (dimension === this.config.maxRecursionDepth - 1 || 
        mergedLocalTurningPoints.length <= 3 || 
        !needsDimensionalEscalation) {
      // Track convergence for this dimension
      if (this.config.measureConvergence) {
        this.convergenceHistory.push({
          previousTurningPoints: [],
          currentTurningPoints: mergedLocalTurningPoints,
          dimension,
          distanceMeasure: 0,
          hasConverged: true,
          didEscalate: false
        });
      }
      
      return this.filterSignificantTurningPoints(mergedLocalTurningPoints);
    }

    // ----- DIMENSIONAL ESCALATION (n → n+1) -----
    this.log(`Dimension ${dimension}: Escalating to dimension ${dimension + 1}`);
    
    // Create meta-messages from turning points for next level analysis
    // This represents the dimensional expansion from n to n+1
    const metaMessages = this.createMetaMessagesFromTurningPoints(mergedLocalTurningPoints, messages);
    this.log(`Dimension ${dimension}: Created ${metaMessages.length} meta-messages for dimension ${dimension + 1}`);

    // Recursively process the meta-messages to find higher-dimensional turning points
    const higherDimensionTurningPoints = await this.multiLayerDetection(metaMessages, dimension + 1);

    // Track convergence and dimension escalation
    if (this.config.measureConvergence) {
      const convergenceState: ConvergenceState = {
        previousTurningPoints: mergedLocalTurningPoints,
        currentTurningPoints: higherDimensionTurningPoints,
        dimension: dimension + 1,
        distanceMeasure: this.calculateStateDifference(mergedLocalTurningPoints, higherDimensionTurningPoints),
        hasConverged: higherDimensionTurningPoints.length > 0,
        didEscalate: true
      };
      this.convergenceHistory.push(convergenceState);
      
      this.log(`Dimension ${dimension} → ${dimension + 1}: Escalation resulted in convergence distance: ${convergenceState.distanceMeasure.toFixed(3)}`);
    }

    // If we didn't find any higher-dimension turning points, use the merged local ones
    if (higherDimensionTurningPoints.length === 0) {
      return this.filterSignificantTurningPoints(mergedLocalTurningPoints);
    }

    // Combine and return turning points with proper filtering
    return this.combineTurningPoints(mergedLocalTurningPoints, higherDimensionTurningPoints);
  }

  /**
   * Calculate a difference measure between two states for convergence tracking
   */
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

  /**
   * Apply complexity function χ from the ARC/CRA framework
   * Maps continuous significance measure to discrete {1,2,3,4,5} complexity levels
   */
  private calculateComplexityScore(significance: number, semanticShiftMagnitude: number): number {
    // Start with base complexity from significance
    let complexity = 1 + significance * 4; // Maps [0,1] to [1,5]
    
    // Adjust based on semantic shift magnitude (stronger shifts may be more complex)
    complexity += (semanticShiftMagnitude - 0.5) * 0.5;
    
    // Ensure complexity is in [1,5] range
    return Math.max(1, Math.min(5, complexity));
  }

  /**
   * Detect turning points within a single chunk of the conversation
   * This represents the local refinement process in the current dimension
   */
  private async detectTurningPointsInChunk(
    messages: Message[],
    dimension: number
  ): Promise<TurningPoint[]> {
    if (messages.length < 2) return [];

    // Generate embeddings for all messages in the chunk
    const embeddings = await this.generateMessageEmbeddings(messages);

    // Find significant semantic shifts between adjacent messages
    const turningPoints: TurningPoint[] = [];
    for (let i = 0; i < embeddings.length - 1; i++) {
      const current = embeddings[i];
      const next = embeddings[i + 1];

      // Calculate semantic distance between current and next message
      const distance = this.calculateSemanticDistance(current.embedding, next.embedding);

      // If the distance exceeds our threshold, we've found a turning point
      if (distance > this.config.semanticShiftThreshold) {
        // Use direct array indices to get the messages
        const beforeMessage = messages[i];
        const afterMessage = messages[i + 1];

        if (beforeMessage && afterMessage) {
          // Classify the turning point using LLM
          const turningPoint = await this.classifyTurningPoint(
            beforeMessage,
            afterMessage,
            distance,
            dimension,
            this.originalMessages
          );
          
          this.log(`    ...Potential turning point detected between messages ${current.id} and ${next.id} (distance: ${distance.toFixed(3)}, complexity: ${turningPoint.complexityScore.toFixed(1)})`);

          // Adjust span if best_start_id and best_end_id were identified
          if (turningPoint.best_start_id && turningPoint.best_end_id && 
              turningPoint.span.startId !== turningPoint.best_start_id && 
              turningPoint.span.endId !== turningPoint.best_end_id) {
            
            this.log(`  ... found best span and adjusted to ${turningPoint.best_start_id} and ${turningPoint.best_end_id}`);
            
            const newSpan = {
              startId: turningPoint.best_start_id,
              endId: turningPoint.best_end_id,
              startIndex: this.originalMessages.findIndex(m => m.id === turningPoint.best_start_id),
              endIndex: this.originalMessages.findIndex(m => m.id === turningPoint.best_end_id)
            };

            if (newSpan.startIndex !== -1 && newSpan.endIndex !== -1) {
              turningPoint.span = newSpan;
            }
          }

          turningPoints.push(turningPoint);
        } else {
          this.log(`Warning: Could not find messages at indices ${i} and ${i + 1}`);
        }
      }
    }

    return turningPoints;
  }

  /**
   * Use LLM to classify a turning point and generate metadata
   * This is the semantic classification component
   */
  private async classifyTurningPoint(
    beforeMessage: Message,
    afterMessage: Message,
    distance: number,
    dimension: number,
    originalMessages?: Message[]
  ): Promise<TurningPoint> {
    // Extract span information
    let originalSpan = {
      startIndex: 0,
      endIndex: 0,
      startMessageId: '',
      endMessageId: ''
    };
    
    let span: MessageSpan = {
      startId: beforeMessage.id,
      endId: afterMessage.id,
      startIndex: parseInt(beforeMessage.id.replace(/\D/g, ''), 10) || 0,
      endIndex: parseInt(afterMessage.id.replace(/\D/g, ''), 10) || 0
    };
    
    // Ensure chronological ordering of span
    if (span.startIndex > span.endIndex) {
      const tempIndex = span.startIndex;
      span.startIndex = span.endIndex;
      span.endIndex = tempIndex;

      const tempId = span.startId;
      span.startId = span.endId;
      span.endId = tempId;
    }

    // First check for spanData in meta-messages (higher priority)
    if ((beforeMessage as Message).spanData) {
      const metaData = (beforeMessage as Message).spanData as MessageSpan;
      span.startId = metaData.startId;
      span.startIndex = metaData.startIndex;

      if (!(afterMessage as Message).spanData) {
        span.endId = metaData.endId;
        span.endIndex = metaData.endIndex;
      }
    }

    if ((afterMessage as Message).spanData) {
      const metaData = (afterMessage as Message).spanData as MessageSpan;
      span.endId = metaData.endId;
      span.endIndex = metaData.endIndex;

      if (!(beforeMessage as Message).spanData) {
        span.startId = metaData.startId;
        span.startIndex = metaData.startIndex;
      }
    }

    // Check if this is a meta-message
    if (beforeMessage.author === 'meta' || afterMessage.author === 'meta') {
      // Extract span information from meta-messages
      const beforeMatches = beforeMessage.message.match(/SpanIndices: (\d+)-(\d+)/);
      const beforeMsgMatches = beforeMessage.message.match(/SpanMessageIds: ([^-\s→]+(?:-[^-\s→]+)*)-([^-\s→]+(?:-[^-\s→]+)*)/);

      const afterMatches = afterMessage.message.match(/SpanIndices: (\d+)-(\d+)/);
      const afterMsgMatches = afterMessage.message.match(/SpanMessageIds: ([^-\s→]+(?:-[^-\s→]+)*)-([^-\s→]+(?:-[^-\s→]+)*)/);

      // Process the extracted span information
      if (beforeMatches && beforeMatches.length >= 3) {
        originalSpan.startIndex = parseInt(beforeMatches[1], 10);
        if (!afterMatches) {
          originalSpan.endIndex = parseInt(beforeMatches[2], 10);
        }
      }

      if (beforeMsgMatches && beforeMsgMatches.length >= 3) {
        originalSpan.startMessageId = beforeMsgMatches[1];
        if (!afterMsgMatches) {
          originalSpan.endMessageId = beforeMsgMatches[2];
        }
      }

      if (afterMatches && afterMatches.length >= 3) {
        originalSpan.endIndex = parseInt(afterMatches[2], 10);
        if (!beforeMatches) {
          originalSpan.startIndex = parseInt(afterMatches[1], 10);
        }
      }

      if (afterMsgMatches && afterMsgMatches.length >= 3) {
        originalSpan.endMessageId = afterMsgMatches[2];
        if (!beforeMsgMatches) {
          originalSpan.startMessageId = afterMsgMatches[1];
        }
      }
    }

    // Prompt for LLM classification
    const systemPrompt = `
You are an expert conversation analyst tasked with identifying and classifying semantic turning points. A potential turning point has been detected between the two messages provided below based on semantic distance analysis.

**Your Goal:** Accurately classify the *type* of shift occurring between Message 1 and Message 2, considering the surrounding context provided in the 'Contextual Aid' section (if available).

**Semantic Distance:** The calculated cosine distance is ${distance.toFixed(2)}. While this distance indicates a shift, focus primarily on the *content and interaction* to determine the category.

**Analysis Steps:**
1.  Read the 'Contextual Aid' messages (if provided) to understand the conversational flow.
2.  Carefully compare Message 1 and Message 2.
3.  Identify the *primary nature* of the change between them.
4.  Select the *single best category* from the list below that describes this change, as the category for 'category'.
    - Topic, Insight, Emotion, Meta-Reflection, Decision, Question, Problem, Action, Clarification, Objection, Other


**Turning Point Categories & Definitions:**
  * **Topic:** A clear shift to a *new subject* or a distinct sub-aspect of the current subject.
  * **Insight:** A moment of realization, discovery, or deeper understanding expressed by a participant (e.g., "Ah, I see!", "So that's why...", recognizing a connection).
  * **Emotion:** A significant change in the expressed *feeling* or emotional tone (e.g., sudden frustration, expressed relief, shared joy, emerging fear). Look for feeling words, exclamations, or shifts in sentiment.
  * **Meta-Reflection:** Participants discuss the conversation *itself*, their communication process, or their understanding/perspective (e.g., "We seem to be stuck," "Let's re-evaluate our approach," "I wasn't clear before").
  * **Decision:** A clear choice is made, or a commitment to a specific course of action is declared (e.g., "Okay, let's do X," "I've decided we need to...", "We will proceed with Y").
  * **Question:** A *pivotal* question is asked that redirects the conversation, challenges assumptions, or opens a new line of inquiry (not just any question).
  * **Problem:** A *new* issue, challenge, obstacle, or difficulty is explicitly identified or surfaces.
  * **Action:** A commitment or agreement to perform a specific task or action is stated (often follows a Decision, e.g., "I will start the analysis," "Can you send the report?").
  * **Clarification:** Resolving a misunderstanding, correcting information, or providing necessary explanation to ensure mutual understanding.
  * **Objection:** Explicit disagreement, pushback, expression of concern, or refusal regarding a proposal, statement, or course of action.
  * **Other:** Use only if the shift is significant but clearly fits none of the above categories. DO NOT use this label unless absolutely necessary.

** In addition: Analyze what type of shift occurs between these messages and provide:**
1. A short, specific label for this turning point (e.g., "Shift to Budget Concerns")
2. 2-4 keywords that characterize this turning point as a list of keyword.
3. An emotional tone of the turningpoint, or an assessemnt of emotional shift 
    - Of only one of the following: \`anticipation\`, \`sadness\`, \`optimism\`, \`trust\`, \`joy\`, \`neutral\`, \`anger\`, \`disgust\`, \`fear\`, \`surprise\`, \`pessimism\`, \`love\`
4. A sentiment value, either "positive" or "negative"
5. A significance score (0.0-1.0) representing how important this turning point is to the overall conversation
6. A memorable quote(s) from either message that captures the essence of this shift, or even from the surrounding context.
7. The best ID representing this turning point.

Respond with a JSON object containing these fields. Do not include any text outside the JSON object.
`;

    const userMessage = `
[Message 1]
Author: ${beforeMessage.author}
ID: ${beforeMessage.id}
Content: ${beforeMessage.message}

[Message 2]
Author: ${afterMessage.author}
ID: ${afterMessage.id}
Content: ${afterMessage.message}
`;

    // Add contextual information
    const neighborsToAdd = Math.max(Math.round(this.config.minMessagesPerChunk / 2), 1);
    
    const originalMessagesNeighborsBefore = originalMessages?.slice(
      Math.max(0, span.startIndex - neighborsToAdd), 
      span.startIndex
    ).filter(Boolean);
    
    const originalMessagesNeighborsAfter = originalMessages?.slice(
      span.endIndex + 1, 
      span.endIndex + 1 + neighborsToAdd
    ).filter(Boolean);
    
    let originalMessagesText = `
    ## Contextual Aid
    - The following text provides broader context to showcase prior and subsequent messages related to this point. Use this context to help formulate a more accurate assessment of the provided turning point in the conversation.
    - This should not be the basis of your response, nor should it be included in your response; it is meant solely as contextual aid.
    `;
    
    const originalNeighborsWithinSpan = originalMessages?.slice(
      span.startIndex, 
      span.endIndex + 1
    ).filter(Boolean);

    // Add context before the turning point
    if (originalMessagesNeighborsBefore?.length) {
      originalMessagesText +=
        `\n### Messages Before As Context\n` +
        originalMessagesNeighborsBefore.map(m => 
          `Author: ${m.author}\nID: "${m.spanData?.startId ?? m.id}"\nContent: ${m.message}`
        ).join('\n\n');
    }

    // Add context after the turning point
    if (originalMessagesNeighborsAfter?.length) {
      originalMessagesText +=
        `\n### Messages After As Context\n` +
        originalMessagesNeighborsAfter.map(m => 
          `Author: ${m.author}\nID: "${m.spanData?.startId ?? m.id}"\nContent: ${m.message}`
        ).join('\n\n');
    }

    // Add context within the turning point span
    if (originalNeighborsWithinSpan?.length) {
      const startNeighborsWithin = originalNeighborsWithinSpan.slice(0, neighborsToAdd);
      const endNeighborsWithin = originalNeighborsWithinSpan.slice(-neighborsToAdd);

      if (startNeighborsWithin.length) {
        originalMessagesText +=
          `\n### Messages within Turning Point Start\n` +
          startNeighborsWithin.map(m => 
            `Author: ${m.author}\nID: "${m.spanData?.startId ?? m.id}"\nContent: ${m.message}`
          ).join('\n\n');
      }

      if (startNeighborsWithin.length && originalNeighborsWithinSpan.length > startNeighborsWithin.length) {
        originalMessagesText += `\n### Messages ${endNeighborsWithin.length > 0 ? 'between the start and end of context within the turning point' : 'that follow and are within the turning point'} turning point following start have been omitted for brevity\n`;
      }

      if (endNeighborsWithin.length && endNeighborsWithin !== startNeighborsWithin) {
        originalMessagesText +=
          `\n### Messages within Turning Point End\n` +
          endNeighborsWithin.map(m => 
            `Author: ${m.author}\nID: "${m.spanData?.startId ?? m.id}"\nContent: ${m.message}`
          ).join('\n\n');
      }
    }

    try {
      // Call LLM for classification
      const response = await this.openai.chat.completions.create({
        model: this.config.classificationModel,
        messages: [
          { 
            role: 'system', 
            content: systemPrompt + (
              (originalMessagesNeighborsBefore?.length || originalMessagesNeighborsAfter?.length) 
                ? `\n\n${originalMessagesText}` 
                : ''
            )
          },
          { role: 'user', content: userMessage }
        ],
        temperature: 0.6,
        //@ts-ignore 
        repeat_penalty: this.config.endpoint ? 1.005 : undefined,
        top_k: this.config.endpoint ? 20 : undefined,
        num_ctx: this.config.endpoint ? 32638 : undefined,
        response_format: {
          "type": "json_schema",
          "json_schema": {
            "name": "turning_point",
            "strict": true,
            "schema": {
              "type": "object",
              "properties": {
                "label": {
                  "type": "string",
                  "description": "A short, specific label for this turning point. This should clearly describe the shift that occurred."
                },
                "category": {
                  "type": "string",
                  "description": "This turning point signifies a specific type of semantic shift. Please classify it using one of the following categories: Topic, Insight, Emotion, Meta-Reflection, Decision, Question, Problem, Action, Clarification, Objection, or Other."
                },
                "keywords": {
                  "type": "array",
                  "description": "A list of keywords associated with this turning point.",
                  "items": {
                    "type": "string"
                  }
                },
                "emotionalTone": {
                  "type": "string",
                  "description": "An emotional tone or sentiment associated with this turning point, only label with one of the following: `anticipation`, `sadness`, `optimism`, `trust`, `joy`, `neutral`, `anger`, `disgust`, `fear`, `surprise`, `pessimism`, `love`."
                },
                "sentiment": {
                  "type": "string",
                  "description": "The sentiment of the turning point, one of: positive, negative. Do not use any other values and provide only one sentiment."
                },
                "significance": {
                  "type": "number",
                  "description": "A significance score from (0.0-1.0) representing how important this turning point is to the overall conversation, or the aspect of the turning point's level of shift that it represents. Higher values indicate a more significant turning point, assess this carefully, and no lower than 0 or higher than 1."
                },
                "quotes": {
                  "type": "array",
                  "description": "A list of notable quotes from the messages in this turning point's span.",
                  "items": {
                    "type": "string"
                  }
                },
                "best_id": {
                  "type": "string",
                  "description": "A turning point is the moment when a change occurs. Based on your assessment of a message ID, identify a single message ID that best represents this turning point, considering the estimated range between the start ID and end ID. Ensure that the selected message ID exists within this range."
                }
              },
              "required": [
                "label",
                "category",
                "keywords",
                "emotionalTone",
                "sentiment",
                "significance",
                "quotes",
                "best_id"
              ],
              "additionalProperties": false
            }
          }
        
        },
        top_p: 0.8,
      });

      const content = response.choices[0]?.message?.content || '';
      let classification: any = {};

      try {
        classification = JSON.parse(content);
      } catch (err) {
        this.log('Error parsing LLM response as JSON:', err);
        // Extract JSON from potential text wrapping
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          classification = JSON.parse(jsonMatch[0]);
        } else {
          this.log('Could not extract JSON from response:', content);
          classification = {
            label: 'Unclassified Turning Point',
            category: 'Other',
            keywords: [],
            emotionalTone: 'neutral',
            sentiment: 'neutral',
            significance: 0.5,
            quotes: [],
            best_id: span.startId
          };
        }
      }

      // Calculate complexity score from significance and distance
      const complexityScore = this.calculateComplexityScore(
        classification.significance || 0.5, 
        distance
      );

      // Create turning point object
      return {
        id: `tp-${dimension}-${span.startIndex}-${span.endIndex}`,
        label: classification.label || 'Unclassified Turning Point',
        category: (classification.category as TurningPointCategory) || 'Other',
        span,
        best_start_id: span.startId,
        best_end_id: span.endId,
        best_id: classification.best_id || span.startId,
        deprecatedSpan: originalSpan,
        semanticShiftMagnitude: distance,
        keywords: classification.keywords || [],
        quotes: Array.isArray(classification.quotes)
          ? classification.quotes
          : (classification.quotes ? [classification.quotes] : []),
        emotionalTone: classification.emotionalTone || 'neutral',
        detectionLevel: dimension,
        significance: classification.significance || 0.5,
        complexityScore // ARC/CRA framework: explicit complexity score
      };
    } catch (err) {
      this.log('Error classifying turning point:', err);

      // Fallback classification
      return {
        id: `tp-${beforeMessage.id}-to-${afterMessage.id}`,
        label: 'Unclassified Turning Point',
        category: 'Other',
        span,
        best_start_id: span.startId,
        best_end_id: span.endId,
        best_id: span.startId,
        semanticShiftMagnitude: distance,
        keywords: [],
        detectionLevel: dimension,
        deprecatedSpan: originalSpan,
        significance: 0.3,
        complexityScore: 2.0 // Default low complexity for fallback
      };
    }
  }

  /**
   * Create meta-messages from turning points for higher-level analysis
   * This implements the dimensional expansion from n to n+1 in the ARC/CRA framework
   */
  private createMetaMessagesFromTurningPoints(
    turningPoints: TurningPoint[],
    originalMessages: Message[]
  ): Message[] {
    if (turningPoints.length === 0) return [];

    // Group turning points by category (first-level abstraction)
    const groupedByCategory: Record<string, TurningPoint[]> = {};

    turningPoints.forEach(tp => {
      const category = tp.category;
      if (!groupedByCategory[category]) {
        groupedByCategory[category] = [];
      }
      groupedByCategory[category].push(tp);
    });

    // Create meta-messages (one per category to find higher-level patterns)
    const metaMessages: Message[] = [];

    // First create category messages - represents dimension n to n+1 transformation
    Object.entries(groupedByCategory).forEach(([category, points], index) => {
      const quotes = points.flatMap(tp => tp.quotes || []).filter(Boolean);
      const keywords = points.flatMap(tp => tp.keywords || []).filter(Boolean);

      // Find the overall span of all turning points in this category
      const minStartIndex = Math.min(...points.map(p => p.span.startIndex));
      const maxEndIndex = Math.max(...points.map(p => p.span.endIndex));

      // Find corresponding message IDs
      const startMsgId = points.find(p => p.span.startIndex === minStartIndex)?.span.startId || '';
      const endMsgId = points.find(p => p.span.endIndex === maxEndIndex)?.span.endId || '';
      
      // Create meta-message content representing higher-dimensional abstraction
      const categoryContent = `
# ${category} Turning Points
Significance: ${Math.max(...points.map(p => p.significance)).toFixed(2)}
Complexity: ${Math.max(...points.map(p => p.complexityScore)).toFixed(2)}
Keywords: ${Array.from(new Set(keywords)).slice(0, 10).join(', ')}
SpanIndices: ${minStartIndex}-${maxEndIndex}
SpanMessageIds: ${startMsgId}-${endMsgId}

## Message Spans:
${points.map(tp => `- ${tp.label}: ${tp.span.startId} → ${tp.span.endId} (${tp.span.startIndex}-${tp.span.endIndex})`).join('\n')}
## Notable Quotes:
${quotes.length > 0 ? quotes.map(q => `- "${q}"`).join('\n') : 'None identified'}
`;

      // Add contextual information
      let builtContext = ``;
      const startMessagesContext = this.originalMessages.slice(Math.max(0, minStartIndex - 3), minStartIndex).filter(Boolean);
      const endMessagesContext = this.originalMessages.slice(maxEndIndex, maxEndIndex + 3).filter(Boolean);

      if (startMessagesContext.length > 0 || endMessagesContext.length > 0) {
        builtContext = `\n\n## Contextual Aid\n- The following text provides broader context to showcase a truncated view of the messages within this span in the turning point.`;

        if (startMessagesContext.length > 0) {
          builtContext += `\n### Messages start of turning point that are within span as Context\n` +
            startMessagesContext.map(m => `Author: ${m.author}\nID: "${m.spanData?.startId ?? m.id}"\nContent: ${m.message}`).join('\n\n');
        }
        
        builtContext += `\n### The messages in between the turning points have been omitted for brevity\n`;
        
        if (endMessagesContext.length > 0) {
          builtContext += `\n### Messages end of turning point span as Context\n` +
            endMessagesContext.map(m => `Author: ${m.author}\nID: "${m.spanData?.startId ?? m.id}"\nContent: ${m.message}`).join('\n\n');
        }
      }

      const span = {
        startId: startMsgId,
        endId: endMsgId,
        startIndex: minStartIndex,
        endIndex: maxEndIndex
      };

      // Add the meta-message representing higher dimensional abstraction
      metaMessages.push({
        id: `meta-cat-${index}`,
        spanData: this.ensureChronologicalSpan(span),
        author: 'meta',
        message: categoryContent + builtContext
      });
    });

    // Then create timeline messages (sequential groups)
    const sortedPoints = [...turningPoints].sort((a, b) => a.span.startIndex - b.span.startIndex);
    const sectionCount = Math.min(4, Math.ceil(sortedPoints.length / 2));
    const pointsPerSection = Math.ceil(sortedPoints.length / sectionCount);

    // Create chronological section meta-messages
    for (let i = 0; i < sectionCount; i++) {
      const sectionPoints = sortedPoints.slice(
        i * pointsPerSection,
        Math.min((i + 1) * pointsPerSection, sortedPoints.length)
      );

      if (sectionPoints.length === 0) continue;

      // Find the overall span of all turning points in this section
      const minStartIndex = Math.min(...sectionPoints.map(p => p.span.startIndex));
      const maxEndIndex = Math.max(...sectionPoints.map(p => p.span.endIndex));

      // Find corresponding message IDs
      const startMsgId = sectionPoints.find(p => p.span.startIndex === minStartIndex)?.span.startId || '';
      const endMsgId = sectionPoints.find(p => p.span.endIndex === maxEndIndex)?.span.endId || '';
      
      // Create section meta-message content
      const sectionContent = `
# Conversation Section ${i + 1}
Span: ${sectionPoints[0].span.startId} → ${sectionPoints[sectionPoints.length - 1].span.endId}
SpanIndices: ${minStartIndex}-${maxEndIndex}
SpanMessageIds: ${startMsgId}-${endMsgId}
Contains ${sectionPoints.length} turning points
Max Complexity: ${Math.max(...sectionPoints.map(p => p.complexityScore)).toFixed(2)}

## Turning Points in this Section:
${sectionPoints.map(tp => `- ${tp.label} (${tp.category}) [${tp.span.startIndex}-${tp.span.endIndex}]`).join('\n')}
## Keywords:
${Array.from(new Set(sectionPoints.flatMap(tp => tp.keywords || []))).slice(0, 10).join(', ')}
`;

      // Add the section meta-message
      metaMessages.push({
        id: `meta-section-${i}`,
        author: 'meta',
        message: sectionContent,
        spanData: {
          startId: startMsgId,
          endId: endMsgId,
          startIndex: minStartIndex,
          endIndex: maxEndIndex
        }
      });
    }

    this.log(`Created ${metaMessages.length} meta-messages for dimensional expansion: ${metaMessages.map(m => m.id).join(', ')}`);
    return metaMessages;
  }

  /**
   * Filter turning points to keep only significant ones
   */
  private filterSignificantTurningPoints(turningPoints: TurningPoint[]): TurningPoint[] {
    if (!this.config.onlySignificantTurningPoints || turningPoints.length === 0) {
      return turningPoints;
    }

    // Sort by significance and complexity
    const sorted = [...turningPoints].sort((a, b) => {
      // First by significance
      if (b.significance !== a.significance) {
        return b.significance - a.significance;
      }
      // Then by complexity (higher complexity points might be more important)
      if (b.complexityScore !== a.complexityScore) {
        return b.complexityScore - a.complexityScore;
      }
      // Finally by semantic shift magnitude
      return b.semanticShiftMagnitude - a.semanticShiftMagnitude;
    });

    // Keep separate turning points from different parts of the conversation
    const result: TurningPoint[] = [];
    const coveredRanges: Set<string> = new Set();

    for (const tp of sorted) {
      // Check if this turning point significantly overlaps with any already included
      const range = `${tp.span.startIndex}-${tp.span.endIndex}`;
      let hasSignificantOverlap = false;

      for (const coverRange of coveredRanges) {
        const [startStr, endStr] = coverRange.split('-');
        const start = parseInt(startStr, 10);
        const end = parseInt(endStr, 10);

        // Calculate overlap
        const overlapStart = Math.max(start, tp.span.startIndex);
        const overlapEnd = Math.min(end, tp.span.endIndex);

        if (overlapStart <= overlapEnd) {
          const overlapSize = overlapEnd - overlapStart + 1;
          const tpSize = tp.span.endIndex - tp.span.startIndex + 1;

          if (overlapSize / tpSize > 0.4) { // 40% overlap threshold
            hasSignificantOverlap = true;
            break;
          }
        }
      }

      if (!hasSignificantOverlap && tp.significance >= this.config.significanceThreshold) {
        result.push(tp);
        coveredRanges.add(range);

        // Stop if we've reached the maximum number of turning points
        if (result.length >= this.config.maxTurningPoints) {
          break;
        }
      }
    }

    // If we didn't find any above threshold but have turning points
    if (result.length === 0 && sorted.length > 0) {
      // Include at least the most significant one
      result.push(sorted[0]);

      // And a second one from a different part of the conversation if possible
      for (let i = 1; i < sorted.length; i++) {
        const tp = sorted[i];
        if (Math.abs(tp.span.startIndex - sorted[0].span.startIndex) > 3) {
          result.push(tp);
          break;
        }
      }
    }

    // Sort by position in conversation
    return result.sort((a, b) => a.span.startIndex - b.span.startIndex);
  }

  /**
   * Combine turning points from different dimensions
   * with priority given to higher-dimensional turning points
   */
  private combineTurningPoints(
    localTurningPoints: TurningPoint[],
    higherDimensionTurningPoints: TurningPoint[]
  ): TurningPoint[] {
    // Prioritize higher-dimensional turning points by boosting their significance
    const boostedHigher = higherDimensionTurningPoints.map(tp => ({
      ...tp,
      significance: Math.min(1.0, tp.significance * 1.5), // Boost significance but cap at 1.0
      detectionLevel: tp.detectionLevel // Keep track of original detection level
    }));

    // Combine all turning points
    const allTurningPoints = [...localTurningPoints, ...boostedHigher];

    // Merge overlapping turning points with priority to higher dimensions
    const mergedTurningPoints = this.mergeAcrossLevels(allTurningPoints);

    // Filter to keep only the most significant turning points
    const filteredTurningPoints = this.filterSignificantTurningPoints(mergedTurningPoints);

    // Sort by position in conversation
    return filteredTurningPoints.sort((a, b) => a.span.startIndex - b.span.startIndex);
  }

  /**
   * Merge similar or overlapping turning points within the same dimension
   * Implements contraction mapping for convergence
   */
  private mergeSimilarTurningPoints(turningPoints: TurningPoint[]): TurningPoint[] {
    if (turningPoints.length <= 1) return turningPoints;

    // Sort turning points by start index
    const sorted = [...turningPoints].sort((a, b) => a.span.startIndex - b.span.startIndex);
    const merged: TurningPoint[] = [];
    let current = sorted[0];

    for (let i = 1; i < sorted.length; i++) {
      const next = sorted[i];

      // Check if turning points overlap or are adjacent
      const isOverlapping = (next.span.startIndex <= current.span.endIndex + 2); // Allow 1 message gap
      // Check if turning points are semantically similar
      const isSimilarCategory = (next.category === current.category);
      const hasCloseIndices = (next.span.startIndex - current.span.endIndex) <= 3; // Within 3 messages

      if ((isOverlapping && isSimilarCategory) || (hasCloseIndices && isSimilarCategory)) {
        // Merge the turning points with a more concise label
        const newLabel = this.createMergedLabel(current.label, next.label);

        // Ensure chronological ordering by taking min for start and max for end
        const startIndex = Math.min(current.span.startIndex, next.span.startIndex);
        const endIndex = Math.max(current.span.endIndex, next.span.endIndex);
        // Get IDs that correspond to the correct indices
        const startId = startIndex === current.span.startIndex ? current.span.startId : next.span.startId;
        const endId = endIndex === current.span.endIndex ? current.span.endId : next.span.endId;
        
        // Create a properly updated span
        const mergedSpan: MessageSpan = {
          startId,
          endId,
          startIndex,
          endIndex
        };
        
        // Update the deprecated span too for consistency
        const mergedDeprecatedSpan = {
          startIndex: Math.min(current.deprecatedSpan.startIndex, next.deprecatedSpan.startIndex),
          endIndex: Math.max(current.deprecatedSpan.endIndex, next.deprecatedSpan.endIndex),
          startMessageId: startIndex === current.span.startIndex ?
            current.deprecatedSpan.startMessageId : next.deprecatedSpan.startMessageId,
          endMessageId: endIndex === current.span.endIndex ?
            current.deprecatedSpan.endMessageId : next.deprecatedSpan.endMessageId
        };
        
        // Create merged turning point
        current = {
          ...current,
          id: `${current.id}+${next.id}`,
          label: newLabel,
          span: mergedSpan,
          deprecatedSpan: mergedDeprecatedSpan,
          semanticShiftMagnitude: (current.semanticShiftMagnitude + next.semanticShiftMagnitude) / 2,
          keywords: [...(current.keywords || []), ...(next.keywords || [])].filter((v, i, a) => a.indexOf(v) === i).slice(0, 5),
          quotes: [...(current.quotes || []), ...(next.quotes || [])].filter((v, i, a) => a.indexOf(v) === i).slice(0, 2),
          // Boost significance of merged turning points
          significance: Math.min(1.0, ((current.significance + next.significance) / 2) * 1.1),
          // Take the max complexity score as merging complex items stays complex
          complexityScore: Math.max(current.complexityScore, next.complexityScore)
        };
      } else {
        merged.push(current);
        current = next;
      }
    }

    // Add the last current item
    merged.push(current);

    return merged;
  }

  /**
   * Merge turning points across different dimensions with priority to higher dimensions
   * Implements the integration of results across dimensions (n, n+1, etc.)
   */
  private mergeAcrossLevels(turningPoints: TurningPoint[]): TurningPoint[] {
    if (turningPoints.length <= 1) return turningPoints;

    // Sort by dimension (higher dimensions first) then by position
    const sorted = [...turningPoints].sort((a, b) => {
      if (b.detectionLevel !== a.detectionLevel) {
        return b.detectionLevel - a.detectionLevel; // Higher dimensions first
      }
      return a.span.startIndex - b.span.startIndex;
    });

    const merged: TurningPoint[] = [];
    const coveredSpans: Set<string> = new Set();

    // Process turning points from higher dimensions first
    for (const tp of sorted) {
      // Check if this span is already covered by a higher-dimension turning point
      const spanKey = this.getSpanKey(tp);
      if (!this.isSpanOverlapping(tp, coveredSpans)) {
        merged.push(tp);
        coveredSpans.add(spanKey);

        // Add all sub-spans within this turning point
        for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
          for (let j = i; j <= tp.span.endIndex; j++) {
            coveredSpans.add(`${i}-${j}`);
          }
        }
      }
    }

    // Sort the result by position in conversation
    return merged.sort((a, b) => a.span.startIndex - b.span.startIndex);
  }

  /**
   * Create a better merged label from two turning point labels
   */
  private createMergedLabel(label1: string, label2: string): string {
    // If labels are identical, return one of them
    if (label1 === label2) return label1;

    // If one is unclassified, use the other
    if (label1.includes('Unclassified')) return label2;
    if (label2.includes('Unclassified')) return label1;

    // Try to create a more concise combined label
    const commonWords = this.findCommonWords(label1, label2);
    if (commonWords.length > 0) {
      // Use common words as a base, then add distinctive terms
      return commonWords.join(' ') + ' Discussion';
    }

    // If no good merging strategy, use a compound label
    return `${label1} and ${label2}`;
  }

  /**
   * Find common significant words between two labels
   */
  private findCommonWords(label1: string, label2: string): string[] {
    const words1 = label1.toLowerCase().split(/\s+/);
    const words2 = label2.toLowerCase().split(/\s+/);

    // Define stopwords to ignore
    const stopwords = new Set(['to', 'the', 'in', 'of', 'and', 'a', 'an', 'on', 'for', 'with', 'shift']);

    // Find common words that aren't stopwords
    return words1.filter(word =>
      words2.includes(word) && !stopwords.has(word) && word.length > 3
    );
  }

  /**
   * Get a unique key for a message span
   */
  private getSpanKey(tp: TurningPoint): string {
    return `${tp.span.startIndex}-${tp.span.endIndex}`;
  }

  /**
   * Check if a span overlaps with any spans in the covered set
   */
  private isSpanOverlapping(tp: TurningPoint, coveredSpans: Set<string>): boolean {
    // Check if the exact span is covered
    if (coveredSpans.has(this.getSpanKey(tp))) return true;

    // Check if any part of the span is covered
    for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
      for (let j = i; j <= tp.span.endIndex; j++) {
        if (coveredSpans.has(`${i}-${j}`)) {
          // If the overlap is significant (>50%), consider it covered
          const overlapSize = j - i + 1;
          const tpSize = tp.span.endIndex - tp.span.startIndex + 1;
          if (overlapSize >= tpSize * 0.5) {
            return true;
          }
        }
      }
    }

    return false;
  }

  /**
   * Generate embeddings for an array of messages
   */
  private async generateMessageEmbeddings(messages: Message[]): Promise<MessageEmbedding[]> {
    const embeddings: MessageEmbedding[] = new Array(messages.length);

    await async.eachOfLimit(messages, 4, async (message, indexStr) => {
      const index = Number(indexStr);
      const embedding = await this.getEmbedding(message.message);

      embeddings[index] = {
        id: message.id,
        index,
        embedding
      };
    });

    return embeddings;
  }

  /**
   * Get embedding for a text string with caching (atomic memory)
   */
  private async getEmbedding(text: string): Promise<Float32Array> {
    try {
      // Use the external embedding function if available, otherwise call OpenAI directly
      if (typeof generateEmbedding === 'function') {
        return await generateEmbedding(text, this.config.embeddingModel);
      }

      // Direct call to OpenAI
      const response = await this.openai.embeddings.create({
        model: this.config.embeddingModel,
        input: text
      });

      return new Float32Array(response.data[0].embedding);
    } catch (err) {
      this.log('Error generating embedding:', err);
      // Return a random embedding in case of error (not ideal but prevents crashing)
      const randomEmbedding = new Float32Array(1536);
      for (let i = 0; i < randomEmbedding.length; i++) {
        randomEmbedding[i] = Math.random();
      }
      return randomEmbedding;
    }
  }

  /**
   * Calculate semantic distance between two embeddings using cosine distance
   * with additional context-aware adjustment
   * Implements the contraction mapping from the ARC theory
   */
  private calculateSemanticDistance(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let magA = 0;
    let magB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }

    if (magA === 0 || magB === 0) return 1; // Maximum distance if either is a zero vector

    const similarity = dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));
    const distance = 1 - similarity; // Convert similarity to distance

    // Apply a sigmoid-like normalization to emphasize significant shifts
    // and de-emphasize minor variations
    // This helps prevent detecting turning points between every message
    const adjustedDistance = 1 / (1 + Math.exp(-10 * (distance - 0.35)));

    return adjustedDistance;
  }

  /**
   * Chunk a conversation based on token count with strict enforcement of minimum message requirements
   */
  private async chunkConversation(messages: Message[], depth = 0): Promise<ChunkingResult> {
    const chunks: Message[][] = [];

    // If conversation is small, still create chunks to enforce minimum message requirements
    // Unless it's smaller than the minimum chunk size
    if (messages.length <= this.config.minMessagesPerChunk) {
      chunks.push([...messages]);
      return {
        chunks,
        numChunks: 1,
        avgTokensPerChunk: await this.getMessageArrayTokenCount(messages)
      };
    }

    let currentChunk: Message[] = [];
    let currentTokens = 0;
    let totalTokens = 0;

    // Number of messages to overlap between chunks
    const overlapSize = Math.min(2, this.config.minMessagesPerChunk - 1);

    // Force chunking even for small conversations by calculating the ideal chunk size
    const idealChunkSize = Math.max(
      this.config.minMessagesPerChunk,
      Math.min(7, Math.ceil(messages.length / 2))
    );

    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      const tokens = await this.getMessageTokenCount(message.message);
      totalTokens += tokens;

      // Add message to current chunk
      currentChunk.push(message);
      currentTokens += tokens;

      // Determine if we should close this chunk
      const hasMinMessages = currentChunk.length >= this.config.minMessagesPerChunk;
      const hasIdealSize = currentChunk.length >= idealChunkSize;
      const approachingMaxTokens = currentTokens >= this.config.maxTokensPerChunk * 0.8;
      const isLastMessage = i === messages.length - 1;

      // Close chunk when:
      // 1. It has minimum messages AND (approaching max tokens OR reached ideal size), OR
      // 2. It's the last message
      if ((hasMinMessages && (approachingMaxTokens || hasIdealSize)) || isLastMessage) {
        // Only add the chunk if it has at least the minimum messages
        if (currentChunk.length >= this.config.minMessagesPerChunk || chunks.length === 0 || depth > 0) {
          chunks.push([...currentChunk]);

          // If not the last message, create overlap for the next chunk
          if (!isLastMessage) {
            currentChunk = currentChunk.slice(Math.max(0, currentChunk.length - overlapSize));
            currentTokens = await this.getMessageArrayTokenCount(currentChunk);
          }
        }
      }
    }

    // If we only created one chunk and the conversation is large enough,
    // force split it into at least two chunks
    if (chunks.length === 1 && messages.length >= this.config.minMessagesPerChunk * 2) {
      const singleChunk = chunks[0];
      const midPoint = Math.floor(singleChunk.length / 2);

      // Ensure both chunks have at least minMessagesPerChunk
      if (midPoint >= this.config.minMessagesPerChunk &&
        singleChunk.length - midPoint >= this.config.minMessagesPerChunk) {

        const firstChunk = singleChunk.slice(0, midPoint);
        const secondChunk = singleChunk.slice(midPoint - overlapSize);

        chunks.splice(0, 1, firstChunk, secondChunk);
      }
    }

    // Calculate average tokens
    const avgTokens = totalTokens / Math.max(1, chunks.length);

    this.log(`Created ${chunks.length} chunks, avg ${Math.round(avgTokens)} tokens, avg ${Math.round(messages.length / chunks.length)} messages per chunk`);

    return {
      chunks,
      numChunks: chunks.length,
      avgTokensPerChunk: avgTokens
    };
  }

  /**
   * Ensures a message span is in chronological order
   */
  private ensureChronologicalSpan(span: MessageSpan): MessageSpan {
    if (span.startIndex > span.endIndex) {
      // Create a new span with swapped values to maintain immutability
      return {
        startId: span.endId,
        endId: span.startId,
        startIndex: span.endIndex,
        endIndex: span.startIndex,
        originalSpan: span.originalSpan
      };
    }
    return span;
  }

  /**
   * Get token count for a message with caching (atomic memory)
   */
  private async getMessageTokenCount(text: string): Promise<number> {
    // Create a hash of the text for cache lookup
    const hash = crypto.createHash('sha256').update(text).digest('hex');

    // Check if we have a cached value
    if (tokenCountCache.has(hash)) {
      return tokenCountCache.get(hash) || 0;
    }

    // Get the token count
    let count: number;
    try {
      count = countTokens(text);
    } catch (err) {
      this.log('Error counting tokens:', err);
      count = Math.ceil(text.length / 4);
    }

    // Cache the result (atomic memory)
    tokenCountCache.set(hash, count);
    return count;
  }

  /**
   * Get token count for multiple messages
   */
  async getMessageArrayTokenCount(messages: Message[]): Promise<number> {
    let total = 0;
    for (const message of messages) {
      total += await this.getMessageTokenCount(message.message);
    }
    return total;
  }

  /**
   * Log debug messages if debug mode is enabled
   */
  private log(...args: any[]): void {
    if (this.config.debug) {
      console.log('[TurningPointDetector]', ...args);
    }
  }

  /**
   * Get the convergence history for analysis
   * Useful for visualizing the ARC/CRA process
   */
  public getConvergenceHistory(): ConvergenceState[] {
    return this.convergenceHistory;
  }
}

// -----------------------------------------------------------------------------
// Example Usage demonstrating the ARC/CRA Framework
// -----------------------------------------------------------------------------

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

// Run the example to demonstrate the ARC/CRA framework
runTurningPointDetectorExample().catch(console.error);