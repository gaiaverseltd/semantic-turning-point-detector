// file: semanticTurningPointDetector.ts
import fs from 'fs-extra';
import winston from 'winston';

// setup winston 

fs.ensureDirSync('results'); // Ensure the results directory exists
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({

      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.timestamp(
          { format: 'YYYY-MM-DD HH:mm:ss' }
        ),
        winston.format.printf(({ timestamp, level, message }) => {
          return `${timestamp} ${level}: ${message}`;
        })
      )
    }),
    new winston.transports.File({
      filename: 'results/semanticTurningPointDetector.log',
      format: winston.format.json()
    })
  ]
});


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
import { MetaMessage, type Message } from './Message';
import { returnFormattedMessageContent } from './stripContent';
import { formResponseFormatSchema, formSystemMessage, formSystemPromptEnding, formUserMessage } from './prompt';

// Cache for token counts to avoid recalculating - implements atomic memory concept
const tokenCountCache = new LRUCache<string, number>({
  max: 10000,
  ttl: 1000 * 60 * 60 * 24
});

// -----------------------------------------------------------------------------
// Embedding Generation
// -----------------------------------------------------------------------------



// -----------------------------------------------------------------------------
// Core Interfaces
// -----------------------------------------------------------------------------



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
  /** Legacy span format no longer utilized due to new class instantiations for MetaMessages */
  deprecatedSpan?: {
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
  /** The emotionality of this turning point if applicable */
  emotionalTone?: string;
  /**
   * The dimension at which this turning point was detected. 
   * If detectionLevel > 0, it indicates that this turning point was analyzed based on a span of turning points, rather than a span of messages.
   */
  detectionLevel: number;
  /** Significance score (higher = more significant) */
  significance: number;

  /** An assessed best point representing the turning point */
  /** The complexity score (1-5) representing saturation in current dimension */
  complexityScore: number;

  /** 
   * A potential label assigned by the LLM, which can be either 'positive' or 'negative'. 
   * However, this label is not definitive and may be improved using a zero-shot model, 
   * based on the classification provided by the LLM.
   */
  sentiment?: string;


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
  /** Model for generating embeddings, e.g 'text-embedding-3-small', or some model custom, from a configurable openai api compatible endpoint v1/embeddings endpoint */
  embeddingModel: string;

  /** Settable openai compatible embedding endpoint */
  embeddingEndpoint?: string;

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

  customResponseFormatJsonSchema?: ResponseFormatJSONSchema;

  /** Inject a custom system instruction into the LLM prompt, not recommended unless you know what you are doing 
   * - Will repeat your system instruction after the contextual aid text, and before the system prompt ending.
   * - This enforces and reminds the llm of the task, which may become blured from the contextual aid text.
  */
  customSystemInstruction?: string;

  /**
   * Inject a custom user message into the LLM prompt, in which the analysis content is added as context.
   * - Not recommended unless you know what you are doing.
   * - Repeats your custom user message after the system prompt ending.
   */
  customUserInstruction?: string;

  /** The maximum number of characters to use when adding a message content as context to analyze */
  max_character_length?: number;


  /**
   * This option determines whether to fail and halt the process if an analysis 
   * encounters an error during a potential turning point.
   * 
   * - Instead of returning a placeholder for an empty analysis that will be ignored,
   *   the process will stop on failure.
   * - This option is provided because, although an analysis may fail, 
   *   it is important to consider that the analysis spans multiple message intervals. 
   *   A single failure in one interval is treated the same as an analysis 
   *   that does not indicate a significant turning point.
   * - This is useful, as one may debug and discover the appropriate settings for the llm request, and once discovered would set this to true, or incorporate a retry mechanism. 
   */
  throwOnError?: boolean;


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
    // Default configuration (from your provided code)
    this.config = {
      apiKey: config.apiKey || process.env.OPENAI_API_KEY || '',
      classificationModel: config.classificationModel || 'gpt-4o-mini',
      embeddingModel: config.embeddingModel || 'text-embedding-3-small',
      embeddingEndpoint: config.embeddingEndpoint,
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
      complexitySaturationThreshold: config.complexitySaturationThreshold || 4.5,
      measureConvergence: config.measureConvergence ?? true
    };

    // Initialize OpenAI client
    this.openai = new OpenAI({
      apiKey: this.config.apiKey,
      baseURL: this.config.endpoint
    });

    if (this.config.debug) {
      logger.info('[TurningPointDetector] Initialized with config:', {
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
    logger.info('Starting turning point detection using ARC/CRA framework for conversation with', messages.length, 'messages');
    this.convergenceHistory = [];

    // Store original messages for reference
    const totalTokens = await this.getMessageArrayTokenCount(messages);
    logger.info(`Total conversation tokens: ${totalTokens}`);
    // Ensure originalMessages is a fresh copy if messages might be mutated elsewhere
    this.originalMessages = messages.map(m => ({ ...m }));

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
    logger.info(`Starting dimensional analysis at n=${dimension}`);

    // Check recursion depth - hard limit on dimensional expansion
    if (dimension >= this.config.maxRecursionDepth) {
      logger.info(`Maximum dimension (n=${dimension}) reached, processing directly without further expansion`);
      // Pass originalMessages context only at dimension 0 if needed by detectTurningPointsInChunk->classifyTurningPoint
      return await this.detectTurningPointsInChunk(messages, dimension, 0, this.originalMessages);
    }

    // For very small conversations (or at deeper levels), use sliding window
    let localTurningPoints: TurningPoint[] = [];

    // Adjusted condition to handle small message counts more directly
    if (messages.length < this.config.minMessagesPerChunk * 2 && dimension === 0) {
      logger.info(`Dimension ${dimension}: Small conversation (${messages.length} msgs), processing directly`);
      // Optionally adjust threshold for small conversations
      const originalThreshold = this.config.semanticShiftThreshold;
      this.config.semanticShiftThreshold = Math.max(0.3, originalThreshold * 1.1); // Slightly higher threshold

      localTurningPoints = await this.detectTurningPointsInChunk(messages, dimension, 0, this.originalMessages);

      // Restore config
      this.config.semanticShiftThreshold = originalThreshold;
    } else {
      // Chunk the conversation
      const { chunks } = await this.chunkConversation(messages, dimension);
      logger.info(`Dimension ${dimension}: Split into ${chunks.length} chunks`);

      if (chunks.length === 0) {
        logger.info(`Dimension ${dimension}: No valid chunks created, returning empty.`);
        return [];
      }

      // Process each chunk in parallel to find local turning points
      const chunkTurningPoints: TurningPoint[][] = new Array(chunks.length);
      const durationsSeconds: number[] = new Array(chunks.length).fill(-1);
      const limit = this.config.endpoint ? 1 : 5; // Limit API calls

      await async.eachOfLimit(
        chunks,
        limit,
        async (chunk, indexStr) => {
          const index = Number(indexStr);
          const startTime = Date.now();

          if (index % 10 === 0 || limit === 1 || this.config.debug) {
            logger.info(` - Dimension ${dimension}: Processing chunk ${index + 1}/${chunks.length} (${chunk.length} messages)`);
          }

          // Pass originalMessages context only at dimension 0
          chunkTurningPoints[index] = await this.detectTurningPointsInChunk(chunk, dimension, index, this.originalMessages);
          const durationSecs = (Date.now() - startTime) / 1000;
          durationsSeconds[index] = durationSecs;

          if (index % 10 === 0 || limit === 1 || this.config.debug) {
            const processedCount = durationsSeconds.filter(d => d > 0).length;
            if (processedCount > 0) {
              const averageDuration = durationsSeconds.filter(d => d > 0).reduce((a, b) => a + b, 0) / processedCount;
              const remainingChunks = durationsSeconds.length - processedCount;
              const remainingTime = (averageDuration * remainingChunks).toFixed(1);
              const percentageComplete = (processedCount / durationsSeconds.length * 100);
              logger.info(`    - Chunk ${index + 1} processed in ${durationSecs.toFixed(1)}s. Est. remaining: ${remainingTime}s (${percentageComplete.toFixed(1)}% complete)`);
            } else {
              logger.info(`    - Chunk ${index + 1} processed in ${durationSecs.toFixed(1)}s.`);
            }
          }
        }
      );

      // Flatten all turning points from all chunks
      localTurningPoints = chunkTurningPoints.flat();
    }

    logger.info(`Dimension ${dimension}: Found ${localTurningPoints.length} raw turning points`);

    // If we found zero or one turning point at this level, return it directly (after potential filtering if needed)
    if (localTurningPoints.length <= 1) {
      // Apply filtering even for single points if configured
      return this.config.onlySignificantTurningPoints
        ? this.filterSignificantTurningPoints(localTurningPoints)
        : localTurningPoints;
    }

    // First merge any similar turning points at this level
    const mergedLocalTurningPoints = this.mergeSimilarTurningPoints(localTurningPoints);
    logger.info(`Dimension ${dimension}: Merged similar TPs to ${mergedLocalTurningPoints.length}`);

    // If merging resulted in 0 or 1 TP, return it (after filtering)
    if (mergedLocalTurningPoints.length <= 1) {
      return this.config.onlySignificantTurningPoints
        ? this.filterSignificantTurningPoints(mergedLocalTurningPoints)
        : mergedLocalTurningPoints;
    }

    // ------------------- CRITICAL ARC/CRA IMPLEMENTATION -------------------
    // Determine whether to expand dimension based on complexity saturation

    // Calculate the maximum complexity in this dimension
    const maxComplexity = Math.max(0, ...mergedLocalTurningPoints.map(tp => tp.complexityScore)); // Ensure non-negative

    // Implement Transition Operator Ψ
    const needsDimensionalEscalation = maxComplexity >= this.config.complexitySaturationThreshold;

    logger.info(`Dimension ${dimension}: Max complexity = ${maxComplexity.toFixed(2)}, Saturation threshold = ${this.config.complexitySaturationThreshold}`);
    logger.info(`Dimension ${dimension}: Needs Escalation (Ψ)? ${needsDimensionalEscalation}`);

    // Conditions to STOP escalation and finalize at this dimension:
    // 1. Max recursion depth reached
    // 2. Too few turning points to warrant higher-level analysis
    // 3. Complexity hasn't saturated (no need to escalate)
    if (dimension >= this.config.maxRecursionDepth - 1 ||
      mergedLocalTurningPoints.length <= 2 || // Adjusted slightly, maybe 2 TPs isn't enough to find meta-patterns
      !needsDimensionalEscalation) {
      logger.info(`Dimension ${dimension}: Finalizing at this level. Applying final filtering.`);
      // Track convergence for this dimension
      if (this.config.measureConvergence) {
        this.convergenceHistory.push({
          previousTurningPoints: [], // No previous state at the final level of processing
          currentTurningPoints: mergedLocalTurningPoints, // TPs before final filtering
          dimension,
          distanceMeasure: 0, // No comparison needed at final step
          hasConverged: true, // Considered converged as processing stops here
          didEscalate: false
        });
      }
      // Filter the merged points before returning
      return this.filterSignificantTurningPoints(mergedLocalTurningPoints);
    }

    // ----- DIMENSIONAL ESCALATION (n → n+1) -----
    logger.info(`Dimension ${dimension}: Escalating to dimension ${dimension + 1}`);

    // Create meta-messages from the merged turning points at this level
    // Pass originalMessages for context if needed by createMetaMessagesFromTurningPoints
    const metaMessages = this.createMetaMessagesFromTurningPoints(mergedLocalTurningPoints, this.originalMessages);
    logger.info(`Dimension ${dimension}: Created ${metaMessages.length} meta-messages for dimension ${dimension + 1}`);

    if (metaMessages.length < 2) {
      logger.info(`Dimension ${dimension}: Not enough meta-messages (${metaMessages.length}) to perform higher-level analysis. Finalizing with current TPs.`);
      if (this.config.measureConvergence) {
        this.convergenceHistory.push({
          previousTurningPoints: mergedLocalTurningPoints, // State before attempted escalation
          currentTurningPoints: mergedLocalTurningPoints, // State after failed escalation
          dimension: dimension + 1, // Represents the attempted next dimension
          distanceMeasure: 0, // No change
          hasConverged: true, // Converged because escalation failed
          didEscalate: false // Escalation attempted but yielded no processable result
        });
      }
      return this.filterSignificantTurningPoints(mergedLocalTurningPoints);
    }

    // Recursively process the meta-messages to find higher-dimensional turning points
    const higherDimensionTurningPoints = await this.multiLayerDetection(metaMessages, dimension + 1);
    logger.info(`Dimension ${dimension + 1}: Found ${higherDimensionTurningPoints.length} higher-dimension TPs.`);


    // Track convergence and dimension escalation
    if (this.config.measureConvergence) {
      const convergenceState: ConvergenceState = {
        previousTurningPoints: mergedLocalTurningPoints, // TPs from dim n
        currentTurningPoints: higherDimensionTurningPoints, // TPs found in dim n+1
        dimension: dimension + 1,
        distanceMeasure: this.calculateStateDifference(mergedLocalTurningPoints, higherDimensionTurningPoints),
        hasConverged: higherDimensionTurningPoints.length > 0, // Converged if TPs were found at higher level
        didEscalate: true
      };
      this.convergenceHistory.push(convergenceState);
      logger.info(`Dimension ${dimension} → ${dimension + 1}: Convergence distance: ${convergenceState.distanceMeasure.toFixed(3)}. Converged: ${convergenceState.hasConverged}`);
    }

    // Combine turning points from local (n) and higher (n+1) dimensions
    // The combine function will handle merging, prioritizing higher-dim, and filtering
    return this.combineTurningPoints(mergedLocalTurningPoints, higherDimensionTurningPoints);
  }

  /**
   * Calculate a difference measure between two states (sets of turning points)
   * Used for convergence tracking. Considers significance and location.
   */
  private calculateStateDifference(
    state1: TurningPoint[],
    state2: TurningPoint[]
  ): number {
    // Handle empty states
    if (state1.length === 0 && state2.length === 0) return 0.0; // No difference
    if (state1.length === 0 || state2.length === 0) return 1.0; // Maximum difference

    // 1. Average Significance Difference
    const avgSig1 = state1.reduce((sum, tp) => sum + tp.significance, 0) / state1.length;
    const avgSig2 = state2.reduce((sum, tp) => sum + tp.significance, 0) / state2.length;
    const sigDiff = Math.abs(avgSig1 - avgSig2); // Range [0, 1]

    // 2. Structural Difference (using Jaccard index on span ranges)
    const spans1 = new Set(state1.map(tp => `${tp.span.startIndex}-${tp.span.endIndex}`));
    const spans2 = new Set(state2.map(tp => `${tp.span.startIndex}-${tp.span.endIndex}`));
    const intersection = new Set([...spans1].filter(span => spans2.has(span)));
    const union = new Set([...spans1, ...spans2]);
    const jaccardDistance = union.size > 0 ? 1.0 - (intersection.size / union.size) : 0.0; // Range [0, 1]

    // Combine the measures (e.g., weighted average)
    const combinedDistance = (sigDiff * 0.5) + (jaccardDistance * 0.5);

    return Math.min(1.0, Math.max(0.0, combinedDistance)); // Ensure bounds [0, 1]
  }

  /**
   * Apply complexity function χ from the ARC/CRA framework
   */
  private calculateComplexityScore(significance: number, semanticShiftMagnitude: number): number {
    // Base complexity from significance (maps [0,1] to [1, 5])
    let complexity = 1 + significance * 4;

    // Adjust based on semantic shift magnitude (distance, scaled 0-1)
    // Larger shifts slightly increase complexity, centered around a baseline distance
    const baselineDistance = 0.3; // Assumes threshold is around here
    complexity += (semanticShiftMagnitude - baselineDistance) * 1.0; // Adjust sensitivity as needed

    // Ensure complexity is within the [1, 5] range
    return Math.max(1, Math.min(5, complexity));
  }

  /**
   * Detect turning points within a single chunk of the conversation
   */
  /**
   * Detect turning points within a single chunk of the conversation
   * This represents the local refinement process in the current dimension
   */
  private async detectTurningPointsInChunk(
    messages: MetaMessage[] | Message[],
    dimension: number,

    chunkIndex: number, // Optional index for logging purposes
    originalMessages: Message[],
  ): Promise<TurningPoint[]> {
    if (messages.length < 2) return [];

    // Generate embeddings for all messages in the chunk
    const embeddings = await this.generateMessageEmbeddings(messages, dimension);



    // Find significant semantic shifts between adjacent messages
    const turningPoints: TurningPoint[] = [];
    const distances: {
      current: number;
      next: number;
      distance: number;
    }[] = []; // Store distances for logging
    const allDistances: {
      current: number;
      next: number;
      distance: number;
    }[] = []; // Store all distances for logging
    for (let i = 0; i < embeddings.length - 1; i++) {
      const current = embeddings[i];
      const next = embeddings[i + 1];

      // Calculate semantic distance between current and next message

      const distance = this.calculateSemanticDistance(
        current.embedding,
        next.embedding,
      );
      const beforeMessage = messages.find((m) => m.id === current.id);
      const afterMessage = messages.find((m) => m.id === next.id);
   
        let thresholdScaleFactor;
        const baseThreshold = this.config.semanticShiftThreshold;
        
        if (baseThreshold > 0.7) {
          // For high initial thresholds (like 0.75), scale down more aggressively
          thresholdScaleFactor = Math.pow(0.25, dimension); // More aggressive (0.25 instead of 0.4)
        } else if (baseThreshold > 0.5) {
          // For medium thresholds
          thresholdScaleFactor = Math.pow(0.35, dimension);
        } else {
          // For already low thresholds
          thresholdScaleFactor = Math.pow(0.5, dimension);
        }
        
        const dimensionAdjustedThreshold = baseThreshold * thresholdScaleFactor;
      if (
        dimensionAdjustedThreshold <= distance
        
      ) {
        distances.push({
          current: current.index,
          next: next.index,
          distance: distance,
        }); // Store distance for logging
      }
      allDistances.push({
        current: current.index,
        next: next.index,
        distance: distance,
      });


    }

    logger.info(
      `For a total number of points: ${embeddings.length}, there were ${distances.length} distances found as being greater than the threshold of ${this.config.semanticShiftThreshold}. 
        - The top 3 greatest distances are: ${allDistances.slice(0, 3).sort((a, b) => b.distance - a.distance).map(d => d.distance.toFixed(3)).join(', ')}
      
      
      This means there were ${distances.length} potential turning points detected ${dimension === 0 ? "with valid user-assistant turn pairs" : "with valid meta-messages"}`,
    );
    if (distances.length === 0) {
      logger.info(
        `No significant semantic shifts detected in chunk ${chunkIndex}`,
      );
      return [];
    }
    for (let d = 0; d < distances.length - 1; d++) {
      const distanceObj = distances[d];
      const i = distanceObj.current; // Current message index
      const current = embeddings[i]; // Current message embedding
      const next = embeddings[distanceObj.next]; // Next message embedding
      // If the distance exceeds our threshold, we've found a turning point
      // Use direct array indices to get the messages
      const distance = distanceObj.distance; // Semantic distance between current and next message
      const beforeMessage = messages[i];
      const afterMessage = messages[i + 1];
      if (beforeMessage == undefined || afterMessage == undefined) {
        logger.info(
          `detectTurningPointsInChunk: warning beforeMessage or afterMessage is undefined, beforeMessage: ${beforeMessage}, afterMessage: ${afterMessage}`,
        );
        continue;
      }

      // Classify the turning point using LLM
      const turningPoint = await this.classifyTurningPoint(
        beforeMessage,
        afterMessage,
        distance,
        dimension,
        originalMessages,
        d,
      );

      logger.info(
        `    ...${chunkIndex ? `[Chunk ${chunkIndex}] ` : ""
        }Potential turning point detected between messages ${current.id
        } and ${next.id} (distance: ${distance.toFixed(
          3,
        )}, complexity: ${turningPoint.complexityScore.toFixed(
          1,
        )}), signif: ${turningPoint.significance.toFixed(2)} category: ${turningPoint.category
        }`,
      );
      if (turningPoint.significance > 1) {
        if (turningPoint.significance > 10) {
          turningPoint.significance = turningPoint.significance / 100;
        } else {
          turningPoint.significance = turningPoint.significance / 10; // Adjusting for scale
        }
      }



      turningPoints.push(turningPoint);
    }

    return turningPoints;
  }

  /**
   * Use LLM to classify a turning point and generate metadata.
   * *** MODIFIED to prioritize message.spanData over regex ***
   */
  private async classifyTurningPoint(
    beforeMessage: Message,
    afterMessage: Message,
    distance: number,
    dimension: number,
    originalMessages: Message[],
    index: number = 0
  ): Promise<TurningPoint> {


    let span: MessageSpan;

    if (dimension > 0) {
      if (beforeMessage instanceof MetaMessage === false || afterMessage instanceof MetaMessage === false) {
        throw new Error("Before or after message is not a MetaMessage");
      }
      const beforeMessageMeta = beforeMessage as MetaMessage;
      const afterMessageMeta = afterMessage as MetaMessage;
      // For higher dimensions, use meta-message and inner methods to get the the span ids for the start and end 
      span = {
        startId: beforeMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0].id,
        endId: afterMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0].id,
        startIndex: this.originalMessages.findIndex((candidateM) => {
          return beforeMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0].id === candidateM.id;
        }),
        endIndex: this.originalMessages.findIndex((candidateM) => {
          return afterMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0].id === candidateM.id;
        }),
        originalSpan: {
          startId: beforeMessage.id,
          endId: afterMessage.id,
          startIndex: index,
          endIndex: index + 1,
        }
      };
    } else {
      // For dimension 0, use original message IDs and find indices
      span = {
        startId: beforeMessage.id,
        endId: afterMessage.id,
        startIndex: MetaMessage.findIndexOfMessageFromId({
          id: beforeMessage.id,
          beforeMessage,
          afterMessage,
          messages: originalMessages,
        }),
        endIndex: MetaMessage.findIndexOfMessageFromId({
          id: afterMessage.id,
          beforeMessage,
          afterMessage,
          messages: originalMessages,
        }),
      };
    }

    // --- REMOVED Regex block for extracting originalSpan from meta-message content ---
    // const originalSpan = { startIndex: 0, endIndex: 0, startMessageId: '', endMessageId: '' };
    // if (beforeMessage.author === 'meta' || afterMessage.author === 'meta') {
    //   ... regex matching logic ...
    // }
    // --- End Removal ---

    // --- LLM Prompt Setup (using original prompt structure) ---

    const systemPrompt = formSystemMessage({
      dimension,
      distance
    })
    const userMessage = formUserMessage({
      config: this.config,
      afterMessage,
      beforeMessage,
      dimension,
      addUserInstructions: this.config.customUserInstruction && this.config.customUserInstruction.length > 0 ? true : false,
    })

    const contextualAidText = this.prepareContextualInfoMeta(
      beforeMessage,
      afterMessage,
      span,
      originalMessages,
      dimension,
      2,
      dimension > 0);

    try {
      // --- Call LLM (using original parameters and schema) ---
      const response = await this.openai.chat.completions.create({
        model: this.config.classificationModel,
        messages: [
          {
            role: 'system', content:
              `${this.config.customSystemInstruction ? this.config.customSystemInstruction : systemPrompt
              }\n\n${contextualAidText}\n------- end of contextual background info see below as reminder of instructions -------\n\n${this.config.customSystemInstruction ? this.config.customSystemInstruction : formSystemPromptEnding(dimension)
              }`,

          },
          { role: 'user', content: this.config.customUserInstruction ? `${this.config.customUserInstruction}\n\n${userMessage}\n\n${this.config.customUserInstruction}` : userMessage },
        ],


        temperature: 0.6,
        //@ts-ignore - Allow vendor-specific params if needed
        repeat_penalty: this.config.endpoint ? 1.005 : undefined,
        top_k: this.config.endpoint ? 20 : undefined,


        stop: ['<|im_end|>'],
        response_format: formResponseFormatSchema(dimension),

        top_p: 0.9,
      });

      const content = response.choices[0]?.message?.content || '{}';
      let classification: any = {};

      try {
        classification = JSON.parse(content);
        console.info(` got classification: ${JSON.stringify(classification, null, 2)}`);
      } catch (err: any) {
        logger.info('Error parsing LLM response as JSON:', err.message);
        // Attempt to extract JSON from markdown code block if necessary
        const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch && jsonMatch[1]) {
          try {
            classification = JSON.parse(jsonMatch[1]);
            logger.info('Successfully extracted JSON from markdown block.');
          } catch (parseErr: any) {
            logger.info('Failed to parse extracted JSON:', parseErr.message);
            classification = {}; // Reset on secondary failure
          }
        } else {
          const plainJsonMatch = content.match(/\{[\s\S]*\}/); // Fallback to find any JSON structure
          if (plainJsonMatch) {
            try {
              classification = JSON.parse(plainJsonMatch[0]);
              logger.info('Successfully extracted JSON using simple match.');
            } catch (parseErr: any) {
              logger.info('Failed to parse simple JSON match:', parseErr.message);
              classification = {};
            }
          } else {
            logger.info('Could not extract JSON from response:', content);
            classification = {};
          }
        }
        // Provide default values if parsing failed completely
        if (Object.keys(classification).length === 0) {
          classification = {
            label: 'Parsing Error - Unclassified', category: 'Other', keywords: [],
            emotionalTone: 'neutral', sentiment: 'neutral', significance: 0.1,
            quotes: [], best_id: span.startId
          };
        }
      }

      // --- Validate and Sanitize LLM Output ---
      const validatedClassification = {
        label: typeof classification.label === 'string' ? classification.label.substring(0, 50) : 'Unknown Turning Point',
        category: typeof classification.category === 'string' ? classification.category as TurningPointCategory : 'Other',
        keywords: Array.isArray(classification.keywords) ? classification.keywords.map(String).slice(0, 4) : [], // Limit count
        emotionalTone: typeof classification.emotionalTone === 'string' ? classification.emotionalTone : 'neutral',
        sentiment: ['positive', 'negative', 'neutral'].includes(classification.sentiment) ? classification.sentiment : 'neutral',
        significance: typeof classification.significance === 'number' ? Math.max(0, Math.min(1, classification.significance)) : 0.5,
        quotes: Array.isArray(classification.quotes) ? classification.quotes.map(String).slice(0, 3) : [], // Limit count
        best_id: typeof classification.best_id === 'string' ? classification.best_id : span.startId, // Default to start of span
      };


      // Calculate complexity score
      const complexityScore = this.calculateComplexityScore(
        validatedClassification.significance,
        distance // Use the raw distance (0-1)
      );

      // --- Construct TurningPoint Object ---
      return {
        id: `tp-${dimension}-${span.startIndex}-${span.endIndex}`,
        label: validatedClassification.label,
        category: validatedClassification.category,
        span: span, // Use the span derived at the beginning

        // deprecatedSpan is no longer populated from regex results
        semanticShiftMagnitude: distance,
        keywords: validatedClassification.keywords,
        quotes: validatedClassification.quotes,
        emotionalTone: validatedClassification.emotionalTone,
        sentiment: validatedClassification.sentiment,
        detectionLevel: dimension,
        significance: validatedClassification.significance,
        complexityScore: complexityScore
      };

    } catch (err: any) {
      logger.info(`Error during LLM call for turning point classification: ${err.message}`);
      // Fallback classification on API error
      if (this.config.throwOnError) {

      } else {
        return {
          id: `tp-err-${dimension}-${span.startId}`,
          label: 'LLM Error - Unclassified',
          category: 'Other',
          span: span,

          semanticShiftMagnitude: distance,
          keywords: [],
          quotes: [],
          emotionalTone: 'neutral',
          sentiment: 'neutral',
          detectionLevel: dimension,
          significance: 0.1,
          complexityScore: 1.0 // Minimum complexity
        };
      }
    }
  }

  /**
   * Updated to utilize new classes of Message and MetaMessage for better structure and clarity
   * @param turningPoints
   * @param originalMessages
   * @returns
   */
  private createMetaMessagesFromTurningPoints(
    turningPoints: TurningPoint[],
    originalMessages: Message[],
  ): Message[] {
    if (turningPoints.length === 0) return [];

    // Group turning points by category (first-level abstraction)
    const groupedByCategory: Record<string, TurningPoint[]> = {};

    turningPoints.forEach((tp) => {
      const category = tp.category;
      if (!groupedByCategory[category]) {
        groupedByCategory[category] = [];
      }
      groupedByCategory[category].push(tp);
    });

    logger.info(
      `Grouped categories: `,
      JSON.stringify(groupedByCategory, null, 2),
    );

    // Create meta-messages (one per category to find higher-level patterns)
    const metaMessages: Message[] = [];

    // First create category messages - represents dimension n to n+1 transformation
    Object.entries(groupedByCategory).forEach(([category, points], index) => {
      // Use the factory method from MetaMessage class to create a properly typed meta-message
      const metaMessage = MetaMessage.createCategoryMetaMessage(
        category,
        points,
        index,
        originalMessages,
      );

      metaMessages.push(metaMessage);
    });

    // Create timeline/section meta-messages
    const sortedPoints = [...turningPoints].sort(
      (a, b) => a.span.startIndex - b.span.startIndex,
    );
    const sectionCount = Math.min(4, Math.ceil(sortedPoints.length / 2));
    const pointsPerSection = Math.ceil(sortedPoints.length / sectionCount);

    // Create chronological section meta-messages
    for (let i = 0; i < sectionCount; i++) {
      const sectionPoints = sortedPoints.slice(
        i * pointsPerSection,
        Math.min((i + 1) * pointsPerSection, sortedPoints.length),
      );

      if (sectionPoints.length === 0) continue;

      // Create a section meta-message using the factory method
      const sectionMetaMessage = MetaMessage.createSectionMetaMessage(
        sectionPoints,
        i,
        this.originalMessages,
      );
      console.info('created sectionMetageMessage')
      metaMessages.push(sectionMetaMessage);
    }

    logger.info(
      `Created ${
        metaMessages.length
      } meta-messages for dimensional expansion: ${metaMessages
        .map((m) => m.id)
        .join(", ")}`,
    );
    return metaMessages;
  }

  // --- Remaining methods are kept identical to your second provided version ---

  /**
   * Filter turning points to keep only significant ones
   * (Using original logic from the second code block)
   */
  private filterSignificantTurningPoints(turningPoints: TurningPoint[]): TurningPoint[] {
    if (!this.config.onlySignificantTurningPoints || turningPoints.length === 0) {
      // Ensure sorted return even if not filtering
      return turningPoints.sort((a, b) => a.span.startIndex - b.span.startIndex);
    }

    logger.info(`Filtering ${turningPoints.length} TPs based on significance >= ${this.config.significanceThreshold} and maxPoints = ${this.config.maxTurningPoints}`);

    // Sort by significance, complexity, magnitude
    const sorted = [...turningPoints].sort((a, b) => {
      if (b.significance !== a.significance) return b.significance - a.significance;
      if (b.complexityScore !== a.complexityScore) return b.complexityScore - a.complexityScore;
      return b.semanticShiftMagnitude - a.semanticShiftMagnitude;
    });

    const result: TurningPoint[] = [];
    const coveredIndices: Set<number> = new Set(); // Use indices for overlap check
    const maxPoints = this.config.maxTurningPoints;

    for (const tp of sorted) {
      // Check significance threshold first
      if (tp.significance < this.config.significanceThreshold) {
        // Only consider points below threshold if we haven't found enough significant ones yet
        if (result.length >= Math.ceil(maxPoints / 2)) { // Heuristic: if we have half the max points, stop adding insignificant ones
          continue;
        }
      }

      // Check for significant overlap with already selected points
      let overlapRatio = 0;
      let isOverlapping = false;
      const tpSpanSize = tp.span.endIndex - tp.span.startIndex + 1;
      if (tpSpanSize > 0) {
        let overlapCount = 0;
        for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
          if (coveredIndices.has(i)) {
            overlapCount++;
          }
        }
        overlapRatio = overlapCount / tpSpanSize;
      }

      // Define significant overlap threshold (e.g., 40% from original code)
      const overlapThreshold = 0.4;
      isOverlapping = overlapRatio > overlapThreshold;


      if (!isOverlapping && result.length < maxPoints) {
        result.push(tp);
        // Mark indices covered by this TP
        for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
          coveredIndices.add(i);
        }
      } else if (isOverlapping) {
        logger.info(`    TP ${tp.id} (Sig: ${tp.significance.toFixed(2)}) overlaps significantly (${(overlapRatio * 100).toFixed(0)}%) with existing TPs. Skipping.`);
      } else if (result.length >= maxPoints) {
        logger.info(`    Reached max turning points (${maxPoints}). Skipping TP ${tp.id}.`);
      }
    }

    // Ensure at least one TP is returned if any were found initially
    if (result.length === 0 && sorted.length > 0) {
      logger.info("No TPs met significance/overlap criteria, returning the single most significant one.");
      result.push(sorted[0]);
    }
    // Add a second diverse TP if only one was kept and more exist (original logic)
    else if (result.length === 1 && sorted.length > 1) {
      for (let i = 1; i < sorted.length; i++) {
        const nextTp = sorted[i];
        // Check if it's sufficiently far from the first one (e.g., > 3 messages gap)
        if (Math.abs(nextTp.span.startIndex - result[0].span.startIndex) > 3) {
          // Check minimal overlap with the first one
          let overlapsFirst = false;
          for (let j = nextTp.span.startIndex; j <= nextTp.span.endIndex; j++) {
            if (j >= result[0].span.startIndex && j <= result[0].span.endIndex) {
              overlapsFirst = true;
              break;
            }
          }
          if (!overlapsFirst) {
            logger.info("Adding a second, non-overlapping TP for diversity.");
            result.push(nextTp);
            break;
          }
        }
      }
    }


    logger.info(`Filtered down to ${result.length} significant turning points.`);
    // Final sort by position in conversation
    return result.sort((a, b) => a.span.startIndex - b.span.startIndex);
  }

  /**
   * Combine turning points from different dimensions
   * (Using original logic from the second code block)
   */
  private combineTurningPoints(
    localTurningPoints: TurningPoint[],
    higherDimensionTurningPoints: TurningPoint[]
  ): TurningPoint[] {
    logger.info(`Combining ${localTurningPoints.length} local (dim ${localTurningPoints[0]?.detectionLevel ?? 'N/A'}) and ${higherDimensionTurningPoints.length} higher (dim ${higherDimensionTurningPoints[0]?.detectionLevel ?? 'N/A'}) TPs.`);

    // Prioritize higher-dimensional turning points by boosting their significance (original logic)
    const boostedHigher = higherDimensionTurningPoints.map(tp => ({
      ...tp,
      // Apply a boost, ensuring it doesn't exceed 1.0
      significance: Math.min(1.0, tp.significance * 1.2), // Adjusted boost factor slightly
      // Keep original detectionLevel for merging logic
    }));

    // Combine all turning points
    const allTurningPoints = [...localTurningPoints, ...boostedHigher];
    logger.info(`Total TPs before cross-level merge: ${allTurningPoints.length}`);


    // Merge overlapping turning points across dimensions, prioritizing higher dimensions/significance
    const mergedTurningPoints = this.mergeAcrossLevels(allTurningPoints);
    logger.info(`Merged across levels to ${mergedTurningPoints.length} TPs.`);

    // Filter the combined & merged list to keep the most significant ones overall
    const filteredTurningPoints = this.filterSignificantTurningPoints(mergedTurningPoints);

    logger.info(`Final combined and filtered TPs: ${filteredTurningPoints.length}`);
    // Sort by position in conversation before returning
    return filteredTurningPoints.sort((a, b) => a.span.startIndex - b.span.startIndex);
  }

  /**
   * Merge similar or overlapping turning points *within* the same dimension
   * (Using original logic from the second code block)
   */
  private mergeSimilarTurningPoints(turningPoints: TurningPoint[]): TurningPoint[] {
    if (turningPoints.length <= 1) return turningPoints;

    // Sort turning points by start index
    const sorted = [...turningPoints].sort((a, b) => a.span.startIndex - b.span.startIndex);
    const merged: TurningPoint[] = [];
    let currentTp = sorted[0]; // Use a more descriptive name

    for (let i = 1; i < sorted.length; i++) {
      const nextTp = sorted[i];

      // Check conditions for merging (original logic)
      const isOverlapping = (nextTp.span.startIndex <= currentTp.span.endIndex + 2); // Allow small gap
      const isSimilarCategory = (nextTp.category === currentTp.category);
      // Added closeness check from original code
      const hasCloseIndices = (nextTp.span.startIndex - currentTp.span.endIndex) <= 3;

      // Merge if overlapping OR close, AND same category
      if ((isOverlapping || hasCloseIndices) && isSimilarCategory) {
        logger.info(`    Merging similar TPs (Dim ${currentTp.detectionLevel}): ${currentTp.id} and ${nextTp.id}`);
        // Merge the turning points
        const newLabel = this.createMergedLabel(currentTp.label, nextTp.label);

        // Create merged span (min start, max end)
        const mergedSpan = this.ensureChronologicalSpan({
          startId: currentTp.span.startIndex <= nextTp.span.startIndex ? currentTp.span.startId : nextTp.span.startId,
          endId: currentTp.span.endIndex >= nextTp.span.endIndex ? currentTp.span.endId : nextTp.span.endId,
          startIndex: Math.min(currentTp.span.startIndex, nextTp.span.startIndex),
          endIndex: Math.max(currentTp.span.endIndex, nextTp.span.endIndex)
        });

        // Update the deprecated span too (original logic, though less relevant now)
        // Note: deprecatedSpan might not exist if TPs came from meta-messages
        const mergedDeprecatedSpan = (currentTp.deprecatedSpan && nextTp.deprecatedSpan) ? {
          startIndex: Math.min(currentTp.deprecatedSpan.startIndex, nextTp.deprecatedSpan.startIndex),
          endIndex: Math.max(currentTp.deprecatedSpan.endIndex, nextTp.deprecatedSpan.endIndex),
          startMessageId: mergedSpan.startIndex === currentTp.deprecatedSpan.startIndex ?
            currentTp.deprecatedSpan.startMessageId : nextTp.deprecatedSpan.startMessageId,
          endMessageId: mergedSpan.endIndex === currentTp.deprecatedSpan.endIndex ?
            currentTp.deprecatedSpan.endMessageId : nextTp.deprecatedSpan.endMessageId
        } : undefined; // Handle cases where deprecatedSpan might be missing

        // Combine keywords and quotes (unique, limited)
        const mergedKeywords = Array.from(new Set([...(currentTp.keywords || []), ...(nextTp.keywords || [])])).slice(0, 5);
        const mergedQuotes = Array.from(new Set([...(currentTp.quotes || []), ...(nextTp.quotes || [])])).slice(0, 3); // Limit quotes too

        // Update the current TP to be the merged version
        currentTp = {
          ...currentTp, // Keep most properties of the first TP
          id: `${currentTp.id}-merged-${nextTp.span.startIndex}`, // Indicate merge in ID
          label: newLabel,
          span: mergedSpan,
          // Only include deprecatedSpan if it was successfully merged
          ...(mergedDeprecatedSpan && { deprecatedSpan: mergedDeprecatedSpan }),
          semanticShiftMagnitude: (currentTp.semanticShiftMagnitude + nextTp.semanticShiftMagnitude) / 2,
          keywords: mergedKeywords,
          quotes: mergedQuotes,
          // Boost significance slightly, cap at 1.0 (original logic)
          significance: Math.min(1.0, ((currentTp.significance + nextTp.significance) / 2) * 1.1),
          // Take max complexity (original logic)
          complexityScore: Math.max(currentTp.complexityScore, nextTp.complexityScore),
          // Combine emotional tone/sentiment logically (e.g., take the one from the more significant TP)
          emotionalTone: currentTp.significance >= nextTp.significance ? currentTp.emotionalTone : nextTp.emotionalTone,
          sentiment: currentTp.significance >= nextTp.significance ? currentTp.sentiment : nextTp.sentiment,
        };
      } else {
        // If not merging, push the completed current TP and move to the next
        merged.push(currentTp);
        currentTp = nextTp;
      }
    }

    // Add the last processed TP
    merged.push(currentTp);

    return merged;
  }

  /**
   * Merge turning points across different dimensions with priority to higher dimensions
   * (Using original logic from the second code block)
   */
  private mergeAcrossLevels(turningPoints: TurningPoint[]): TurningPoint[] {
    if (turningPoints.length <= 1) return turningPoints;

    // Sort by dimension DESC (higher first), then by significance DESC, then by start index ASC
    const sorted = [...turningPoints].sort((a, b) => {
      if (b.detectionLevel !== a.detectionLevel) return b.detectionLevel - a.detectionLevel;
      // Add secondary sort by significance within the same level
      if (b.significance !== a.significance) return b.significance - a.significance;
      return a.span.startIndex - b.span.startIndex; // Tertiary sort by position
    });

    const merged: TurningPoint[] = [];
    // Use a Set of covered *indices* for more granular overlap checking
    const coveredIndices: Set<number> = new Set();

    logger.info(`    Merging across levels. Input count: ${sorted.length}. Prioritizing higher dimension/significance.`);

    for (const tp of sorted) {
      // Check how much of this TP's span is already covered
      let overlapCount = 0;
      const spanSize = tp.span.endIndex - tp.span.startIndex + 1;
      if (spanSize <= 0) continue; // Skip invalid spans

      for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
        if (coveredIndices.has(i)) {
          overlapCount++;
        }
      }
      const overlapRatio = overlapCount / spanSize;

      // Define significant overlap threshold (e.g., 50% - adjust as needed)
      const significantOverlapThreshold = 0.5;

      // Keep the TP if it's not significantly overlapped by higher-priority ones
      if (overlapRatio < significantOverlapThreshold) {
        merged.push(tp);
        // Mark its indices as covered *only if it wasn't already significantly covered*
        for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
          coveredIndices.add(i);
        }
        logger.info(`      Keeping TP ${tp.id} (Dim ${tp.detectionLevel}, Sig ${tp.significance.toFixed(2)}). Overlap: ${(overlapRatio * 100).toFixed(0)}%`);
      } else {
        logger.info(`      Skipping TP ${tp.id} (Dim ${tp.detectionLevel}, Sig ${tp.significance.toFixed(2)}) due to significant overlap (${(overlapRatio * 100).toFixed(0)}%).`);
      }
    }


    logger.info(`    Finished merging across levels. Output count: ${merged.length}.`);
    // Sort the final result by position in conversation
    return merged.sort((a, b) => a.span.startIndex - b.span.startIndex);
  }


  /**
   * Create a merged label (Using original logic)
   */
  private createMergedLabel(label1: string, label2: string): string {
    if (label1 === label2) return label1;
    if (label1.includes('Unclassified')) return label2;
    if (label2.includes('Unclassified')) return label1;

    const commonWords = this.findCommonWords(label1, label2);
    if (commonWords.length > 0) {
      // Simple common word approach (original)
      return commonWords.join(' ') + ' Discussion';
    }
    // Fallback concatenation (original)
    return `${label1} / ${label2}`.substring(0, 70); // Add length limit
  }

  /**
   * Find common significant words (Using original logic)
   */
  private findCommonWords(label1: string, label2: string): string[] {
    const words1 = label1.toLowerCase().split(/\s+/);
    const words2 = label2.toLowerCase().split(/\s+/);
    const stopwords = new Set(['to', 'the', 'in', 'of', 'and', 'a', 'an', 'on', 'for', 'with', 'shift', 'discussion', 'about', 'summary']); // Added more stopwords
    // Filter common words, exclude stopwords, ensure decent length
    return words1.filter(word =>
      word.length > 3 && words2.includes(word) && !stopwords.has(word)
    );
  }

  /**
   * Get a unique key for a message span (Using original logic)
   */
  private getSpanKey(tp: TurningPoint): string {
    return `${tp.span.startIndex}-${tp.span.endIndex}`;
  }

  /**
   * Check if a span overlaps with any spans in the covered set (Using original logic)
   * Note: This might be less accurate than the index-based check in mergeAcrossLevels now.
   * Kept for potential use by filterSignificantTurningPoints if it uses range strings.
   */
  private isSpanOverlapping(tp: TurningPoint, coveredSpans: Set<string>): boolean {
    // Check exact span match
    if (coveredSpans.has(this.getSpanKey(tp))) return true;

    // Check partial overlap (original logic)
    for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
      for (let j = i; j <= tp.span.endIndex; j++) {
        if (coveredSpans.has(`${i}-${j}`)) {
          const overlapSize = j - i + 1;
          const tpSize = tp.span.endIndex - tp.span.startIndex + 1;
          // Original 50% threshold
          if (tpSize > 0 && overlapSize / tpSize >= 0.5) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * Generate embeddings for an array of messages (Using original logic)
   */
  private async generateMessageEmbeddings(messages: Message[], dimension = 0): Promise<MessageEmbedding[]> {
    const embeddings: MessageEmbedding[] = new Array(messages.length);
    // Using original concurrency limit of 4
    console.info(`Generating embeddings for ${messages.length} messages with dimension ${dimension}.`);
    await async.eachOfLimit(messages, 4, async (message, indexStr) => {

      let candidateText = message.message;
      if (dimension > 0 && message instanceof MetaMessage) {
        // For meta-messages, use the original message text
        const metaMessage = message as MetaMessage;
        const messagesWithinMeta = metaMessage.getMessagesInTurningPointSpanToMessagesArray();
        // naiviely concatenate the last and first message
        candidateText = `${messagesWithinMeta[0].message} ${messagesWithinMeta[messagesWithinMeta.length - 1].message}`;
        console.info(`Meta message ${message.id} contains ${messagesWithinMeta.length} messages.`);
      }
      const index = Number(indexStr);
      try {
        const embedding = await this.getEmbedding(candidateText);
        // Store the original index from the input 'messages' array
        embeddings[index] = {
          id: message.id,
          index: index, // Store the index within the current chunk/message list being processed
          embedding
        };
      } catch (error: any) {
        logger.info(`Error generating embedding for msg ${message.id} at index ${index}: ${error.message}. Creating zero vector.`);
        const embeddingSize = 1536; // Assuming text-embedding-3-small
        embeddings[index] = { id: message.id, index: index, embedding: new Float32Array(embeddingSize).fill(0) };
      }
    });
    // Filter out any potential null/undefined entries if errors occurred
    return embeddings.filter(e => e);
  }


  /**
   * Retrieves the embedding for a given text string.
   * 
   * - Utilizes the configured embedding model.
   * - If the model is not an OpenAI model, it uses the OpenAI client with an optional endpoint.
   * - If the model is an OpenAI model, it creates a new OpenAI client to ensure the correct endpoint is used.
   */


  private async getEmbedding(text: string, naiveTokenLimit = 8192): Promise<Float32Array> {

    // ensure that the input text length is less that 8192 tokens in a naive way

    let tokensCount = countTokens(text);
    // do a while loop iterative slow removal of tokens
    let textCharToTokenRatio = text.length / tokensCount;
    while (tokensCount > naiveTokenLimit) {
      // remove the exact number of caracters to get the token count under 8192
      const charsToRemove = Math.ceil((tokensCount - naiveTokenLimit) * textCharToTokenRatio);
      text = text.substring(0, text.length - charsToRemove);
      tokensCount = countTokens(text);
    }


    try {


      // Direct call to OpenAI (original logic)
      let response;

      const openaiEmbeddingModels = [
        'text-embedding-ada-002',
        'text-embedding-ada-001',
        'text-embedding-3-small',
        'text-embedding-3-large',

      ]


      const openai = new OpenAI();
      openai.baseURL = this.config.embeddingEndpoint ?? openai.baseURL;
      response = await openai.embeddings.create({
        model: this.config.embeddingModel,
        input: text,
        encoding_format: 'float',
      });



      if (response.data && response.data.length > 0 && response.data[0].embedding) {
        return new Float32Array(response.data[0].embedding);
      } else {
        throw new Error('Invalid embedding response structure from OpenAI');
      }
    } catch (err: any) { // Catch specific error types if possible
      logger.info(`Error generating embedding: ${err.message}. Returning zero vector. ${(err as Error).stack} ${this.config.embeddingEndpoint}`);
      // Return a zero embedding on error (more predictable than random)
      const embeddingSize = 1536; // Match expected dimension
      return new Float32Array(embeddingSize).fill(0);
    }
  }

  /**
   * Calculate semantic distance (Using original logic with sigmoid adjustment)
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
   * Chunk a conversation (Using original logic)
   */
  private async chunkConversation(messages: Message[], dimension = 0): Promise<ChunkingResult> {
    const chunks: Message[][] = [];
    // scale down the min messages based on each depth increase 
    const baseMinMessages = this.config.minMessagesPerChunk; // Preserve original value
    const dimensionScaleFactor = Math.max(0.1, Math.pow(0.35, dimension));
    const minMessages = Math.max(2, Math.round(baseMinMessages * dimensionScaleFactor));
    // But you need to scale maxTokensPerChunk too
    const tokenScaleFactor = Math.max(0.2, Math.pow(0.5, dimension));
    const maxTokens = Math.max(this.config.minTokensPerChunk, Math.round(this.config.maxTokensPerChunk * tokenScaleFactor));



    // Handle case where input has fewer than minimum messages (original logic)
    if (messages.length < minMessages) {
      logger.info(`Input messages (${messages.length}) less than minMessagesPerChunk (${minMessages}). Returning as single chunk.`);
      // Return single chunk only if it's not empty
      return {
        chunks: messages.length > 0 ? [[...messages]] : [],
        numChunks: messages.length > 0 ? 1 : 0,
        avgTokensPerChunk: messages.length > 0 ? await this.getMessageArrayTokenCount(messages) : 0
      };
    }


    let currentChunk: Message[] = [];
    let currentTokens = 0;
    let totalTokens = 0;
    const overlapSize = Math.max(0, Math.min(2, Math.floor(minMessages / 2))); // Keep overlap small, max 2

    // Ideal chunk size logic (original) - helps guide chunking beyond just token limits
    const idealMessageCount = Math.max(
      minMessages,
      Math.min(10, Math.ceil(messages.length / Math.max(1, Math.floor(messages.length / 10)))) // Aim for ~10 chunks max?
    );
    logger.info(`    Chunking ${messages.length} messages. MinMsgs: ${minMessages}, MaxTokens: ${maxTokens}, IdealMsgCount: ${idealMessageCount}, Overlap: ${overlapSize}`);


    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      // Handle potential undefined messages just in case
      if (!message) continue;

      const tokens = await this.getMessageTokenCount(message.message);
      totalTokens += tokens;

      // Add message to current chunk
      currentChunk.push(message);
      currentTokens += tokens;

      // Determine if we should close this chunk (original logic)
      const hasMinMessages = currentChunk.length >= minMessages;
      const hasIdealSize = currentChunk.length >= idealMessageCount; // Use ideal count
      const approachingMaxTokens = currentTokens >= (maxTokens) * 0.9; // Increase threshold slightly
      const isLastMessage = i === messages.length - 1;
      const significantlyOverMaxTokens = currentTokens > (maxTokens) * 1.1; // Check if significantly over

      // Close chunk conditions (refined slightly)
      // 1. Last message: always close.
      // 2. Min messages met AND (approaching/over max tokens OR reached ideal size)
      if (isLastMessage || (hasMinMessages && (approachingMaxTokens || hasIdealSize || significantlyOverMaxTokens))) {
        // Add the chunk
        chunks.push([...currentChunk]);
        logger.info(`      Created chunk ${chunks.length} with ${currentChunk.length} messages, ${currentTokens} tokens. Ends at index ${i}.`);

        // If not the last message, start next chunk with overlap
        if (!isLastMessage) {
          const startIndexForNextChunk = Math.max(0, currentChunk.length - overlapSize);
          currentChunk = currentChunk.slice(startIndexForNextChunk);
          currentTokens = await this.getMessageArrayTokenCount(currentChunk);
          logger.info(`      Starting next chunk with ${currentChunk.length} overlapping messages, ${currentTokens} tokens.`);
        } else {
          currentChunk = []; // Clear chunk if it was the last message
          currentTokens = 0;
        }
      }
    }

    // --- Post-processing Chunks (similar to original logic but simplified) ---
    // If only one chunk was created for a large conversation, try splitting it
    if (chunks.length === 1 && messages.length >= minMessages * 2) {
      const singleChunk = chunks[0];
      const midPointIndex = Math.floor(singleChunk.length / 2);

      // Ensure split results in chunks meeting min size requirement
      if (midPointIndex >= minMessages && singleChunk.length - midPointIndex >= minMessages) {
        logger.info("    Single chunk detected for large conversation, attempting to split.");
        const firstChunk = singleChunk.slice(0, midPointIndex);
        // Apply overlap when splitting
        const secondChunkStartIndex = Math.max(0, midPointIndex - overlapSize);
        const secondChunk = singleChunk.slice(secondChunkStartIndex);

        chunks.splice(0, 1, firstChunk, secondChunk); // Replace single chunk with two
        logger.info(`    Successfully split into two chunks: ${firstChunk.length} msgs and ${secondChunk.length} msgs.`);
      }
    }

    // Ensure all chunks meet minimum message requirement, merging small trailing chunks if necessary
    if (chunks.length > 1) {
      const lastChunk = chunks[chunks.length - 1];
      if (lastChunk.length < minMessages) {
        logger.info(`    Last chunk (${lastChunk.length} msgs) is smaller than min size (${minMessages}). Merging with previous.`);
        const secondLastChunk = chunks[chunks.length - 2];
        // Combine, avoiding duplicates from overlap if possible
        const combinedChunk = [...secondLastChunk];
        const lastIdInSecondLast = secondLastChunk[secondLastChunk.length - 1]?.id;
        let appendStartIndex = 0;
        if (overlapSize > 0 && lastChunk[0]?.id === lastIdInSecondLast) {
          appendStartIndex = 1; // Skip first element of last chunk if it's the same as end of previous
        }
        combinedChunk.push(...lastChunk.slice(appendStartIndex));

        // Replace the last two chunks with the merged one
        chunks.splice(chunks.length - 2, 2, combinedChunk);
        logger.info(`    Merged last two chunks. New chunk count: ${chunks.length}.`);
      }
    }
    // --- End Post-processing ---


    const numChunks = chunks.length;
    const avgTokens = numChunks > 0 ? totalTokens / numChunks : 0; // Avoid division by zero
    const avgMessagesPerChunk = numChunks > 0 ? messages.length / numChunks : 0;

    logger.info(`    Finished chunking. Created ${numChunks} chunks. Avg Tokens: ${avgTokens.toFixed(0)}, Avg Msgs: ${avgMessagesPerChunk.toFixed(1)}`);

    return {
      chunks,
      numChunks,
      avgTokensPerChunk: avgTokens
    };
  }


  /**
   * Ensures a message span is in chronological order by index
   * (Using original logic)
   */
  private ensureChronologicalSpan(span: MessageSpan): MessageSpan {
    if (span.startIndex > span.endIndex) {
      logger.info(`Warning: Correcting reversed span indices (${span.startIndex} > ${span.endIndex}) for IDs ${span.startId}/${span.endId}.`);
      // Create a new span with swapped values to maintain immutability
      return {
        startId: span.endId, // Swap IDs
        endId: span.startId,
        startIndex: span.endIndex, // Swap indices
        endIndex: span.startIndex,
        originalSpan: span.originalSpan ?? span // Store original if needed
      };
    }
    return span; // Return original if already chronological
  }

  /**
   * Get token count for a message with caching (Using original logic)
   */
  private async getMessageTokenCount(text: string): Promise<number> {
    const hash = crypto.createHash('sha256').update(text).digest('hex');
    if (tokenCountCache.has(hash)) {
      return tokenCountCache.get(hash)!;
    }

    let count: number;
    try {
      // Use external countTokens function (original logic)
      count = countTokens(text);
    } catch (err: any) {
      logger.info(`Error counting tokens: ${err.message}. Falling back to length/4.`);
      count = Math.ceil(text.length / 4);// Fallback (original logic) based on a naive approach that uses a ratio of four characters per token. This method is inaccurate because the token count is also influenced by certain special strings or characters. The ratio can vary significantly depending on the type of text content; for example, JSON data may yield a slightly different ratio. However, this four-character ratio is generally reasonable for semantic text.

    }

    // Cache the result (original logic)
    tokenCountCache.set(hash, count);

    return count;
  }

  /**
   * Get token count for multiple messages (Using original logic)
   */
  async getMessageArrayTokenCount(messages: Message[]): Promise<number> {
    let total = 0;
    for (const message of messages) {
      // Handle potentially undefined messages in array
      if (message?.message) {
        total += await this.getMessageTokenCount(message.message);
      }
    }
    return total;
  }


  /**
   * Get the convergence history for analysis (Using original logic)
   */
  public getConvergenceHistory(): ConvergenceState[] {
    return this.convergenceHistory;
  }


  private prepareContextualInfoMeta(
    _beforeMessage: Message,
    _afterMessage: Message,
    span: MessageSpan,
    originalMessages?: Message[],
    dimension = 0,
    messagesToAddPerContextualUnit = 2,
    addMessagesWithinSpan = false,
  ) {
    if (
      _beforeMessage instanceof MetaMessage &&
      _afterMessage instanceof MetaMessage
    ) {


      const getContextualInfoForThisPotentialMeta =
        MetaMessage.getMessagesContentContextualAidFromJustProvidedBeforeAndAfterMessages(
          _beforeMessage,
          _afterMessage,
          dimension,
          messagesToAddPerContextualUnit,
          this.config.max_character_length,
          originalMessages,
          "before-and-after",
        );

      if (addMessagesWithinSpan) {
        const getContextualWithinSpan =
          MetaMessage.getMessagesContentContextualAidFromJustProvidedBeforeAndAfterMessages(
            _beforeMessage,
            _afterMessage,
            dimension,
            messagesToAddPerContextualUnit,
            this.config.max_character_length,
            originalMessages,
            "within",
          );

        return `${getContextualInfoForThisPotentialMeta}\n\n${getContextualWithinSpan}`;
      } else {
        return getContextualInfoForThisPotentialMeta;
      }
    } else {
      return this.prepareContextualInfoMessage(
        _beforeMessage,
        _afterMessage,
        span,
        originalMessages,
        dimension,
        addMessagesWithinSpan,
      );
    }
  }

  /**
 * Prepares contextual information to be appended to the LLM prompt.
 * Gathers nearby messages (before, after, and within the span) for additional context.
 * - This preperation is soley meant for dimension at 0, wherein the messages are still base level messages, rather than MetaMessages, which encompass a group of turning points.
 */
  private prepareContextualInfoMessage(
    _beforeMessage: Message,
    _afterMessage: Message,
    span: MessageSpan,
    originalMessages?: Message[],
    dimension: number = 0,
    addMessagesWithinSpan = false,
  ): string {

    if (dimension > 0 || _beforeMessage instanceof MetaMessage || _afterMessage instanceof MetaMessage) {
      throw new Error("Contextual information preparation is not supported for dimensions greater than 0.");
    }

    const neighborsToAdd = Math.max(
      Math.round(this.config.minMessagesPerChunk / 2),
      1,
    );

    const originalMessagesNeighborsBefore = originalMessages
      ?.slice(
        Math.max(
          0,
          MetaMessage.findIndexOfMessageFromId({
            beforeMessage: _beforeMessage,
            afterMessage: _afterMessage,
            id: span.startId,
            messages: originalMessages,
          }),
        ),
        span.startIndex,
      )
      .filter(Boolean);
    const messageIndexAfterStart = MetaMessage.findIndexOfMessageFromId({
      beforeMessage: _beforeMessage,
      afterMessage: _afterMessage,
      id: span.endId,
      messages: originalMessages,
    });
    const originalMessagesNeighborsAfter = originalMessages
      ?.slice(
        // span.endIndex + 1,
        // span.endIndex + 1 + neighborsToAdd
        messageIndexAfterStart + 1, // Start from the message right after the end of the span
        span.endIndex + 1 + neighborsToAdd, // Add a few more messages after the end of the span for context
      )
      .filter(Boolean);

    let contextualSystemInstruction = `### Messages Content For Contextual Aid
  - The following provides the message content within the span you are analyzing for the turning point.
  - Use this information to help you analyze the turning point and provide a more informed response (e.g. to identify quotes, and/or keywords, etc).
  - Possibly also included are messages before the turning point, and messages after the turning point, this is meant as broader contextual info.
    `;



    // Add context before the turning point
    if (originalMessagesNeighborsBefore?.length) {
      contextualSystemInstruction +=
        originalMessagesNeighborsBefore.length > 0
          ? `\n### Messages Before As Context\n` +
          originalMessagesNeighborsBefore
            .map((m) =>
              returnFormattedMessageContent(this.config, m, dimension),
            )
            .join("\n\n")
          : `\n### There does not exist any messages before this span of messages that encompass a potential ${dimension > 0 ? "meta turning point to formulate based on a grouping of turning points that encapuslate a single conversation" : "a potential turning point of two messages (that are part of a bigger single converation) being analyzed as provided in the user message content."}.\n`;
    }

    // Add context after the turning point
    if (originalMessagesNeighborsAfter?.length) {
      contextualSystemInstruction +=
        `\n### Messages After As Context\n` +
        originalMessagesNeighborsAfter
          .map((m) => {

            return returnFormattedMessageContent(this.config, m, dimension);
          })

          .join("\n\n");
    }

    // Add context within the turning point span or with the two messages if dimension is 0
    if (addMessagesWithinSpan && originalMessagesNeighborsAfter?.length) {
      contextualSystemInstruction +=
        `\n### Messages Within this potential turning point of the two messages below\n` +
        [_beforeMessage, _afterMessage]
          .map((m) => {

            return returnFormattedMessageContent(this.config, m, dimension);
          })
          .join("\n\n");

    }

    return contextualSystemInstruction;
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
  const thresholdForMinDialogueShift = 22;
  const conversationPariah = fs.readJsonSync('src/conversationPariah.json', { 'encoding': 'utf-8' }) as Message[];

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
    semanticShiftThreshold: 0.75,
    minTokensPerChunk: 1024,
    maxTokensPerChunk: 8192,
    // uses for now embeddings only from openai
     embeddingModel: "text-embedding-snowflake-arctic-embed-l-v2.0",

    // embeddingModel: 'text-embedding-3-large',
    // ARC framework: dynamic recursion depth based on conversation complexity
    maxRecursionDepth: Math.min(determineRecursiveDepth(conversationPariah), 5),

    onlySignificantTurningPoints: true,
    significanceThreshold: 0.75,

    // ARC framework: chunk size scales with complexity
    minMessagesPerChunk: Math.ceil(determineRecursiveDepth(conversationPariah) * 3.5),

    // ARC framework: number of turning points scales with conversation length
    maxTurningPoints: Math.max(6, Math.round(conversationPariah.length / 20)),

    // CRA framework: explicit complexity saturation threshold for dimensional escalation
    complexitySaturationThreshold: 4.1,

    max_character_length: 4000,

    // Enable convergence measurement for ARC analysis
    measureConvergence: true,

      classificationModel: "gpt-4o-mini",
    // classificationModel: 'phi-4-mini-Q5_K_M:3.8B',
    // classificationModel: 'gpt-4o-mini',
    // e.g. llmstudio or ollama
    embeddingEndpoint: 'http://127.0.0.1:7756/v1',

    debug: true,
    // ollama
    // endpoint: 'http:/localhost:11434/v1',
    // or lmstudio
    // endpoint: 'http://localhost:7756/v1'
  });

  try {
    // Detect turning points using the ARC/CRA framework
    const tokensInConvoFile = await detector.getMessageArrayTokenCount(conversation);
    const turningPoints = await detector.detectTurningPoints(conversation);

    const endTime = new Date().getTime();
    const difference = endTime - startTime;
    const formattedTimeDateDiff = new Date(difference).toISOString().slice(11, 19);

    logger.info(`\nTurning point detection took as MM:SS: ${formattedTimeDateDiff} for ${tokensInConvoFile} tokens in the conversation`);

    // Display results with complexity scores from the ARC framework
    logger.info('\n=== DETECTED TURNING POINTS (ARC/CRA Framework) ===\n');

    turningPoints.forEach((tp, i) => {
      logger.info(`${i + 1}. ${tp.label} (${tp.category})`);
      logger.info(`   Messages: "${tp.span.startId}" → "${tp.span.endId}"`);
      logger.info(`   Dimension: n=${tp.detectionLevel}`);
      logger.info(`   Complexity Score: ${tp.complexityScore.toFixed(2)} of 5`);
      logger.info(`   Emotional Tone: ${tp.emotionalTone || 'unknown'}`);
      logger.info(`   Semantic Shift Magnitude: ${tp.semanticShiftMagnitude.toFixed(2)}`);
      logger.info(`   Sentiment: ${tp.sentiment || 'unknown'}`);
      logger.info(`   Significance: ${tp.significance.toFixed(2)}`);
      logger.info(`   Keywords: ${tp.keywords?.join(', ') || 'none'}`);
      logger.info(`   Quotes: ${tp.quotes?.join(', ') || 'none'}`);



    });

    // Get and display convergence history to demonstrate the ARC framework
    const convergenceHistory = detector.getConvergenceHistory();

    logger.info('\n=== ARC/CRA FRAMEWORK CONVERGENCE ANALYSIS ===\n');
    convergenceHistory.forEach((state, i) => {
      logger.info(`Iteration ${i + 1}:`);
      logger.info(`  Dimension: n=${state.dimension}`);
      logger.info(`  Convergence Distance: ${state.distanceMeasure.toFixed(3)}`);
      logger.info(`  Dimensional Escalation: ${state.didEscalate ? 'Yes' : 'No'}`);
      logger.info(`  Turning Points: ${state.currentTurningPoints.length}`);

    });

    // Save turning points to file
    fs.writeJSONSync('results/turningPoints.json', turningPoints, { spaces: 2, encoding: 'utf-8' });

    // Also save convergence analysis
    fs.writeJSONSync('results/convergence_analysis.json', convergenceHistory, { spaces: 2, encoding: 'utf-8' });

    logger.info('Results saved to files.');
  } catch (err) {
    console.error('Error detecting turning points:', err);
  }



}


runTurningPointDetectorExample().catch(err => {
  console.error('Error in example run:', err);
}).finally(() => {
  process.exit(0);
});