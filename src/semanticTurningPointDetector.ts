// file: semanticTurningPointDetector.ts
import fs from "fs-extra";
import winston from "winston";
import ollama, { Ollama } from "ollama";

import dotenv from "dotenv";
dotenv.config();

// setup winston

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

import async from "async";
import { OpenAI } from "openai";
import { LRUCache } from "lru-cache";
import crypto from "crypto";
import { countTokens, createEmbeddingCache } from "./tokensUtil";
import { conversation } from "./conversation";
import { ResponseFormatJSONSchema } from "openai/resources/shared";
import { MetaMessage, type Message } from "./Message";
import { returnFormattedMessageContent } from "./stripContent";
import {
  formResponseFormatSchema,
  formSystemMessage,
  formSystemPromptEnding,
  formUserMessage,
} from "./prompt";
import {
  ChunkingResult,
  ConvergenceState,
  MessageEmbedding,
  MessageSpan,
  TurningPoint,
  turningPointCategories,
  TurningPointCategory,
  TurningPointDetectorConfig,
} from "./types";

// Cache for token counts to avoid recalculating - implements atomic memory concept
const tokenCountCache = new LRUCache<string, number>({
  max: 10000,
  ttl: 1000 * 60 * 60 * 24,
});

// -----------------------------------------------------------------------------
// Main Detector Class
// -----------------------------------------------------------------------------

export class SemanticTurningPointDetector {
  private config: TurningPointDetectorConfig;

  /**
   * For ease of use in llm requests, openai's client is used as it allows configurable endpoints. Further expoloration might be reasonable in leveraging other libaries, such as ollama, llmstudio, genai, etc, for more direct compatibility with other LLM providers. Though at this time, the OpenAI client is sufficient for requests done by this detector.
   */
  private openai: OpenAI;
  /**
   * This provides the array of the initial messages that were passed to the detector. This is noted as such as throughout the process, ARC involves analyzing subsets of the original messages, and the original messages are not modified.
   */
  private originalMessages: Message[] = [];
  /**
   * AN array of changes of state across iterations, used for convergence measurement.
   * This is used to track the evolution of turning points across iterations and dimensions.
   * This is used when returning the final results, to determine whether the turning points have converged.
   */
  private convergenceHistory: ConvergenceState[] = [];
  /**
   * Used to help mitigate repeat embedding requests for the same message content. And can be configured to avoid excessive RAM usage via `embeddingCacheRamLimitMB`.
   */
  private embeddingCache: LRUCache<string, Float32Array>;

  private endpointType: "ollama" | "openai" | "unknown";

  private ollama: Ollama | null = null;
  readonly logger: winston.Logger | Console;
  /**
   * Creates a new instance of the semantic turning point detector
   */
  constructor(config: Partial<TurningPointDetectorConfig> = {}) {
    // Default configuration (from your provided code)
    this.config = {
      apiKey: config.apiKey || process.env.OPENAI_API_KEY || "",
      classificationModel: config.classificationModel || "gpt-4o-mini",
      embeddingModel: config.embeddingModel || "text-embedding-3-small",
      embeddingEndpoint: config.embeddingEndpoint,
      semanticShiftThreshold: config.semanticShiftThreshold || 0.22,
      minTokensPerChunk: config.minTokensPerChunk || 250,
      maxTokensPerChunk: config.maxTokensPerChunk || 2000,
      concurrency: (config.concurrency ?? config?.endpoint) ? 1 : 4,
      embeddingConcurrency: config.embeddingConcurrency ?? 5,
      logger: config?.logger ?? undefined,
      embeddingCacheRamLimitMB: config.embeddingCacheRamLimitMB || 256,
      maxRecursionDepth: config.maxRecursionDepth || 3,
      onlySignificantTurningPoints: config.onlySignificantTurningPoints ?? true,
      significanceThreshold: config.significanceThreshold || 0.5,
      minMessagesPerChunk: config.minMessagesPerChunk || 3,
      maxTurningPoints: config.maxTurningPoints || 5,
      debug: config.debug || false,
      turningPointCategories:
        config?.turningPointCategories &&
          config?.turningPointCategories.length > 0
          ? config.turningPointCategories
          : turningPointCategories,
      endpoint: config.endpoint,

      temperature: config?.temperature ?? 0.6,
      top_p: config?.top_p ?? 0.95,
      complexitySaturationThreshold:
        config.complexitySaturationThreshold || 4.5,
      measureConvergence: config.measureConvergence ?? true,
    };

    this.endpointType = config?.endpoint
      ? config.endpoint.includes("api.openai.com")
        ? "unknown"
        : "unknown"
      : "openai";

    if (this.config.logger === undefined) {
      fs.ensureDirSync("results");
      this.logger = winston.createLogger({
        level: "info",
        format: winston.format.combine(
          winston.format.timestamp(),
          winston.format.json(),
        ),
        transports: [
          new winston.transports.Console({
            format: winston.format.combine(
              winston.format.colorize(),
              winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
              winston.format.printf(({ timestamp, level, message }) => {
                return `${timestamp} ${level}: ${message}`;
              }),
            ),
          }),
          new winston.transports.File({
            filename: "results/semanticTurningPointDetector.log",
            format: winston.format.json(),
          }),
        ],
      });
    }

    // now validate the turning point categories (that wil simply log warnings), and also after the logging is setup above.
    if (
      config?.turningPointCategories &&
      config?.turningPointCategories.length > 0
    ) {
      this.validateTurningPointCategories(config.turningPointCategories);
    }

    // Initialize OpenAI client
    this.openai = new OpenAI({
      apiKey:
        this.config.apiKey ??
        process.env.LLM_API_KEY ??
        process.env.OPENAI_API_KEY,
      baseURL: this.config.endpoint,
    });

    /**
     * Initialize the embedding cache with the specified RAM limit.
     */

    this.embeddingCache = createEmbeddingCache(
      this.config.embeddingCacheRamLimitMB,
    );

    if (this.config.debug) {
      this.logger.info("[TurningPointDetector] Initialized with config:", {
        ...this.config,
        apiKey: "[REDACTED]",
      });

      this.logger.info(
        `[TurningPointDetector] Embedding cache initialized with ${this.embeddingCache.max} max entries (${this.config.embeddingCacheRamLimitMB}MB limit)`,
      );
    }
  }

  public getModelName(): string {
    return this.config.classificationModel;
  }

  /**
   * Main entry point: Detect turning points in a conversation
   * Implements the full ARC/CRA framework
   */
  public async detectTurningPoints(messages: Message[]): Promise<{
    confidence: number;
    points: TurningPoint[];
  }> {
    this.logger.info(
      "Starting turning-point detection (ARC/CRA) on",
      messages.length,
      "messages",
    );
    this.convergenceHistory = [];

    const isEndpointOllamaBased = await this.isOllamaEndpoint(
      this.config.endpoint,
    );

    if (isEndpointOllamaBased) {
      this.endpointType = "ollama";

      // strip away paths from endpint to get host
      // e.g. https://loollama.liu.netwsdf.network/v1" to "https://ollama.liu.netwsdf.network"
      const url = new URL(this.config.endpoint);
      const host = `${url.protocol}//${url.hostname}`;
      //
      this.logger.info(
        `Detected Ollama endpoint: ${host}. Initializing Ollama client.`,
      );

      this.ollama = new Ollama({
        host,
      });
    }

    // ── cache original conversation for downstream helpers
    const totalTokens = await this.getMessageArrayTokenCount(messages);
    this.logger.info(`Total conversation tokens: ${totalTokens}`);
    this.originalMessages = messages.map((m) => ({ ...m }));

    // ── 1️⃣  full multi-layer detection (dim-0 entry)
    const turningPointsFound = await this.multiLayerDetection(messages, 0);
    this.logger.info(
      `Multi-layer detection returned ${turningPointsFound.length} turning points`,
    );

    // ── 2️⃣  compute a per-TP confidence score
    const confidenceScoresByPoint: number[] = new Array(
      turningPointsFound.length,
    ).fill(0);

    // helper to collapse per-message embeddings into a single mean vector
    const meanEmbedding = (embs: MessageEmbedding[]): Float32Array => {
      if (embs.length === 0) return new Float32Array(1536);

      const dim = embs[0].embedding.length;

      const softMax = (values: number[]): number[] => {
        const maxVal = Math.max(...values);
        const exps = values.map((v) => Math.exp(v - maxVal));
        const sumExps = exps.reduce((sum, v) => sum + v, 0);
        return exps.map((v) => v / sumExps);
      };

      const magnitudes = embs.map(({ embedding }) =>
        Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0)),
      );

      const attnWeights = softMax(magnitudes);

      const acc = new Float32Array(dim);
      for (let idx = 0; idx < embs.length; idx++) {
        const { embedding } = embs[idx];
        const weight = attnWeights[idx];
        for (let i = 0; i < dim; i++) {
          acc[i] += embedding[i] * weight;
        }
      }

      return acc;
    };

    await async.eachOfLimit(
      turningPointsFound,
      this.config.concurrency,
      async (tp, idxStr) => {
        const idx = Number(idxStr);

        // slice conversation around this TP
        const pre = messages.slice(0, tp.span.startIndex);
        const turn = messages.slice(tp.span.startIndex, tp.span.endIndex + 1);
        const post = messages.slice(tp.span.endIndex + 1);

        if (pre.length === 0 || post.length === 0) {
          this.logger.info(
            `TP ${tp.id} at edges of convo – skipping confidence`,
          );
          confidenceScoresByPoint[idx] = 0;
          return;
        }

        // generate *per message* embeddings for each slice
        const [preE, turnE, postE] = await Promise.all([
          this.generateMessageEmbeddings(pre, 0),
          this.generateMessageEmbeddings(turn, 0),
          this.generateMessageEmbeddings(post, 0),
        ]);

        // collapse to single vectors
        const vPre = meanEmbedding(preE);
        const vTurn = meanEmbedding(turnE);
        const vPost = meanEmbedding(postE);

        // distance (0-1)  – higher when meaning shifts
        const distPre = this.calculateSemanticDistance(vPre, vTurn);
        const distPost = this.calculateSemanticDistance(vTurn, vPost);

        // simple confidence: average outward semantic shift
        confidenceScoresByPoint[idx] = (distPre + distPost) / 2;

        this.logger.info(
          `TP ${tp.id}: distPre=${distPre.toFixed(3)}, distPost=${distPost.toFixed(
            3,
          )}, conf=${confidenceScoresByPoint[idx].toFixed(3)}`,
        );
      },
    );

    // ── 3️⃣  aggregate conversation-level confidence (mean of non-zero scores)
    const valid = confidenceScoresByPoint.filter((v) => v > 0);
    const aggregateConfidence =
      valid.length === 0 ? 0 : valid.reduce((s, v) => s + v, 0) / valid.length;

    this.logger.info(
      `Aggregate confidence for conversation: ${aggregateConfidence.toFixed(3)}`,
    );

    return {
      confidence: aggregateConfidence,
      points: turningPointsFound,
    };
  }

  /**
   * Multi-layer detection implementing the ARC/CRA dimensional processing
   * This is the primary implementation of the transition operator Ψ
   */
  private async multiLayerDetection(
    messages: Message[],
    dimension: number,
  ): Promise<TurningPoint[]> {
    this.logger.info(`Starting dimensional analysis at n=${dimension}`);

    // Check recursion depth - hard limit on dimensional expansion
    if (dimension >= this.config.maxRecursionDepth) {
      this.logger.info(
        `Maximum dimension (n=${dimension}) reached, processing directly without further expansion`,
      );
      // Pass originalMessages context only at dimension 0 if needed by detectTurningPointsInChunk->classifyTurningPoint
      return await this.detectTurningPointsInChunk(
        messages,
        dimension,
        0,
        this.originalMessages,
      );
    }

    // For very small conversations (or at deeper levels), use sliding window
    let localTurningPoints: TurningPoint[] = [];

    // Adjusted condition to handle small message counts more directly
    if (
      messages.length < this.config.minMessagesPerChunk * 2 &&
      dimension === 0
    ) {
      this.logger.info(
        `Dimension ${dimension}: Small conversation (${messages.length} msgs), processing directly`,
      );
      // Optionally adjust threshold for small conversations
      const originalThreshold = this.config.semanticShiftThreshold;
      this.config.semanticShiftThreshold = Math.max(
        0.3,
        originalThreshold * 1.1,
      ); // Slightly higher threshold

      localTurningPoints = await this.detectTurningPointsInChunk(
        messages,
        dimension,
        0,
        this.originalMessages,
      );

      // Restore config
      this.config.semanticShiftThreshold = originalThreshold;
    } else {
      // Chunk the conversation
      const { chunks } = await this.chunkConversation(messages, dimension);
      this.logger.info(
        `Dimension ${dimension}: Split into ${chunks.length} chunks`,
      );

      if (chunks.length === 0) {
        this.logger.info(
          `Dimension ${dimension}: No valid chunks created, returning empty.`,
        );
        return [];
      }

      // Process each chunk in parallel to find local turning points
      const chunkTurningPoints: TurningPoint[][] = new Array(chunks.length);
      const durationsSeconds: number[] = new Array(chunks.length).fill(-1);
      const limit = this.config.concurrency;

      await async.eachOfLimit(chunks, limit, async (chunk, indexStr) => {
        const index = Number(indexStr);
        const startTime = Date.now();

        if (index % 10 === 0 || limit < 10 || this.config.debug) {
          this.logger.info(
            ` - Dimension ${dimension}: Processing chunk ${index + 1}/${chunks.length} (${chunk.length} messages)`,
          );
        }

        // Pass originalMessages context only at dimension 0
        chunkTurningPoints[index] = await this.detectTurningPointsInChunk(
          chunk,
          dimension,
          index,
          this.originalMessages,
        );
        const durationSecs = (Date.now() - startTime) / 1000;
        durationsSeconds[index] = durationSecs;

        if (index % 10 === 0 || limit < 10 || this.config.debug) {
          const processedCount = durationsSeconds.filter((d) => d > 0).length;
          if (processedCount > 0) {
            const averageDuration =
              durationsSeconds.filter((d) => d > 0).reduce((a, b) => a + b, 0) /
              processedCount;
            const remainingChunks = durationsSeconds.length - processedCount;
            const remainingTime = (averageDuration * remainingChunks).toFixed(
              1,
            );
            const percentageComplete =
              (processedCount / durationsSeconds.length) * 100;
            this.logger.info(
              `    - Chunk ${index + 1} processed in ${durationSecs.toFixed(1)}s. Est. remaining: ${remainingTime}s (${percentageComplete.toFixed(1)}% complete)`,
            );
          } else {
            this.logger.info(
              `    - Chunk ${index + 1} processed in ${durationSecs.toFixed(1)}s.`,
            );
          }
        }
      });

      // Flatten all turning points from all chunks
      localTurningPoints = chunkTurningPoints.flat();
    }

    this.logger.info(
      `Dimension ${dimension}: Found ${localTurningPoints.length} raw turning points`,
    );

    // If we found zero or one turning point at this level, return it directly (after potential filtering if needed)
    if (localTurningPoints.length <= 1) {
      // Apply filtering even for single points if configured
      return this.config.onlySignificantTurningPoints
        ? this.filterSignificantTurningPoints(localTurningPoints)
        : localTurningPoints;
    }

    // First merge any similar turning points at this level
    const mergedLocalTurningPoints =
      this.mergeSimilarTurningPoints(localTurningPoints);
    this.logger.info(
      `Dimension ${dimension}: Merged similar TPs to ${mergedLocalTurningPoints.length}`,
    );

    // If merging resulted in 0 or 1 TP, return it (after filtering)
    if (mergedLocalTurningPoints.length <= 1) {
      return this.config.onlySignificantTurningPoints
        ? this.filterSignificantTurningPoints(mergedLocalTurningPoints)
        : mergedLocalTurningPoints;
    }

    // ------------------- CRITICAL ARC/CRA IMPLEMENTATION -------------------
    // Determine whether to expand dimension based on complexity saturation

    // Calculate the maximum complexity in this dimension
    const maxComplexity = Math.max(
      0,
      ...mergedLocalTurningPoints.map((tp) => tp.complexityScore),
    ); // Ensure non-negative

    // Implement Transition Operator Ψ
    const needsDimensionalEscalation =
      maxComplexity >= this.config.complexitySaturationThreshold;

    this.logger.info(
      `Dimension ${dimension}: Max complexity = ${maxComplexity.toFixed(2)}, Saturation threshold = ${this.config.complexitySaturationThreshold}`,
    );
    this.logger.info(
      `Dimension ${dimension}: Needs Escalation (Ψ)? ${needsDimensionalEscalation}`,
    );

    // Conditions to STOP escalation and finalize at this dimension:
    // 1. Max recursion depth reached
    // 2. Too few turning points to warrant higher-level analysis
    // 3. Complexity hasn't saturated (no need to escalate)
    if (
      dimension >= this.config.maxRecursionDepth - 1 ||
      mergedLocalTurningPoints.length <= 2 || // Adjusted slightly, maybe 2 TPs isn't enough to find meta-patterns
      !needsDimensionalEscalation
    ) {
      this.logger.info(
        `Dimension ${dimension}: Finalizing at this level. Applying final filtering.`,
      );
      // Track convergence for this dimension
      if (this.config.measureConvergence) {
        this.convergenceHistory.push({
          previousTurningPoints: [], // No previous state at the final level of processing
          currentTurningPoints: mergedLocalTurningPoints, // TPs before final filtering
          dimension,
          distanceMeasure: 0, // No comparison needed at final step
          hasConverged: true, // Considered converged as processing stops here
          didEscalate: false,
        });
      }
      // Filter the merged points before returning
      return this.filterSignificantTurningPoints(mergedLocalTurningPoints);
    }

    // ----- DIMENSIONAL ESCALATION (n → n+1) -----
    this.logger.info(
      `Dimension ${dimension}: Escalating to dimension ${dimension + 1}`,
    );

    // Create meta-messages from the merged turning points at this level
    // Pass originalMessages for context if needed by createMetaMessagesFromTurningPoints
    const metaMessages = this.createMetaMessagesFromTurningPoints(
      mergedLocalTurningPoints,
      this.originalMessages,
    );
    this.logger.info(
      `Dimension ${dimension}: Created ${metaMessages.length} meta-messages for dimension ${dimension + 1}`,
    );

    if (metaMessages.length < 2) {
      this.logger.info(
        `Dimension ${dimension}: Not enough meta-messages (${metaMessages.length}) to perform higher-level analysis. Finalizing with current TPs.`,
      );
      if (this.config.measureConvergence) {
        this.convergenceHistory.push({
          previousTurningPoints: mergedLocalTurningPoints, // State before attempted escalation
          currentTurningPoints: mergedLocalTurningPoints, // State after failed escalation
          dimension: dimension + 1, // Represents the attempted next dimension
          distanceMeasure: 0, // No change
          hasConverged: true, // Converged because escalation failed
          didEscalate: false, // Escalation attempted but yielded no processable result
        });
      }
      return this.filterSignificantTurningPoints(mergedLocalTurningPoints);
    }

    // Recursively process the meta-messages to find higher-dimensional turning points
    const higherDimensionTurningPoints = await this.multiLayerDetection(
      metaMessages,
      dimension + 1,
    );
    this.logger.info(
      `Dimension ${dimension + 1}: Found ${higherDimensionTurningPoints.length} higher-dimension TPs.`,
    );

    // Track convergence and dimension escalation
    if (this.config.measureConvergence) {
      const convergenceState: ConvergenceState = {
        previousTurningPoints: mergedLocalTurningPoints, // TPs from dim n
        currentTurningPoints: higherDimensionTurningPoints, // TPs found in dim n+1
        dimension: dimension + 1,
        distanceMeasure: this.calculateStateDifference(
          mergedLocalTurningPoints,
          higherDimensionTurningPoints,
        ),
        hasConverged: higherDimensionTurningPoints.length > 0, // Converged if TPs were found at higher level
        didEscalate: true,
      };
      this.convergenceHistory.push(convergenceState);
      this.logger.info(
        `Dimension ${dimension} → ${dimension + 1}: Convergence distance: ${convergenceState.distanceMeasure.toFixed(3)}. Converged: ${convergenceState.hasConverged}`,
      );
    }

    // Combine turning points from local (n) and higher (n+1) dimensions
    // The combine function will handle merging, prioritizing higher-dim, and filtering
    return this.combineTurningPoints(
      mergedLocalTurningPoints,
      higherDimensionTurningPoints,
    );
  }

  /**
   * Calculate a difference measure between two states (sets of turning points)
   * Used for convergence tracking. Considers significance and location.
   */
  private calculateStateDifference(
    state1: TurningPoint[],
    state2: TurningPoint[],
  ): number {
    // Handle empty states
    if (state1.length === 0 && state2.length === 0) return 0.0; // No difference
    if (state1.length === 0 || state2.length === 0) return 1.0; // Maximum difference

    // 1. Average Significance Difference
    const avgSig1 =
      state1.reduce((sum, tp) => sum + tp.significance, 0) / state1.length;
    const avgSig2 =
      state2.reduce((sum, tp) => sum + tp.significance, 0) / state2.length;
    const sigDiff = Math.abs(avgSig1 - avgSig2); // Range [0, 1]

    // 2. Structural Difference (using Jaccard index on span ranges)
    const spans1 = new Set(
      state1.map((tp) => `${tp.span.startIndex}-${tp.span.endIndex}`),
    );
    const spans2 = new Set(
      state2.map((tp) => `${tp.span.startIndex}-${tp.span.endIndex}`),
    );
    const intersection = new Set(
      [...spans1].filter((span) => spans2.has(span)),
    );
    const union = new Set([...spans1, ...spans2]);
    const jaccardDistance =
      union.size > 0 ? 1.0 - intersection.size / union.size : 0.0; // Range [0, 1]

    // Combine the measures (e.g., weighted average)
    const combinedDistance = sigDiff * 0.5 + jaccardDistance * 0.5;

    return Math.min(1.0, Math.max(0.0, combinedDistance)); // Ensure bounds [0, 1]
  }

  /**
   * Apply complexity function χ from the ARC/CRA framework
   */
  private calculateComplexityScore(
    significance: number,
    semanticShiftMagnitude: number,
  ): number {
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
    const embeddings = await this.generateMessageEmbeddings(
      messages,
      dimension,
    );

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

      this.logger.debug(
        `Anlyzing with dimensionAdjustedThreshold: ${dimensionAdjustedThreshold.toFixed(3)}, compared to original threshold: ${baseThreshold.toFixed(3)}`,
      );
      if (dimensionAdjustedThreshold <= distance) {
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

    this.logger.info(
      `For a total number of points: ${embeddings.length}, there were ${distances.length} distances found as being greater than the threshold of ${this.config.semanticShiftThreshold}. Across this span of messages of length ${messages.length}, the following distances were found:
        - The top 3 greatest distances are: ${allDistances
        .slice(0, 3)
        .sort((a, b) => b.distance - a.distance)
        .map((d) => d.distance.toFixed(3))
        .join(", ")}
      
      
      This means there were ${distances.length} potential turning points detected ${dimension === 0 ? "with valid user-assistant turn pairs" : "with valid meta-messages"}`,
    );
    if (distances.length === 0) {
      this.logger.info(
        `No significant semantic shifts detected in chunk ${chunkIndex}`,
      );
      return [];
    }
    await async.eachOfLimit(
      distances,
      this.config.concurrency,
      async (distanceObj, idxStr) => {
        const d = Number(idxStr);

        const i = distanceObj.current; // Current message index
        const current = embeddings[i]; // Current message embedding
        const next = embeddings[distanceObj.next]; // Next message embedding
        // If the distance exceeds our threshold, we've found a turning point
        // Use direct array indices to get the messages
        const distance = distanceObj.distance; // Semantic distance between current and next message
        const beforeMessage = messages[i];
        const afterMessage = messages[i + 1];
        if (beforeMessage == undefined || afterMessage == undefined) {
          this.logger.info(
            `detectTurningPointsInChunk: warning beforeMessage or afterMessage is undefined, beforeMessage: ${beforeMessage}, afterMessage: ${afterMessage}`,
          );
          return;
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

        this.logger.info(
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
      },
    );

    return turningPoints;
  }

  private async clearOllamaCache(modelName: string) {
    // 4. This is the bogey request. It tells the server to unload the model.
    // await fetch('http://your-ollama-host:11434/api/generate', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({
    //     model: modelName,
    //     keep_alive: 0,
    //   }),
    // });
    try {
      return await this.ollama.generate({
        model: modelName,
        keep_alive: 0, // Unload the model after use,
        prompt: "\\no_think",
        options: {
          num_predict: 0, // No predictions needed, just unload
        },
      });
    } catch (error) {
      this.logger.warn(
        `Error clearing Ollama cache for model ${modelName}: ${(error as any)?.message || error}`,
      );
    }
  }

  /**
   * Use LLM to classify a turning point and generate metadata.
   * *** MODIFIED to prioritize message.spanData over regex ***
   */
  /**
   * Use LLM to classify a turning point and generate metadata.
   * This implementation uses a highly modular prompt architecture with
   * multiple distinct user messages to ensure clarity. The payload consists of:
   * - A system message that sets the core identity and universal constraints.
   * - A static context user message containing framework and evaluation criteria.
   * - A dynamic data user message that provides conversation context and the specific messages to analyze.
   * - A final user instruction message that tells the model what to do with all this information.
   */
  private async classifyTurningPoint(
    beforeMessage: Message,
    afterMessage: Message,
    distance: number,
    dimension: number,
    originalMessages: Message[],
    index: number = 0,
  ): Promise<TurningPoint> {
    let span: MessageSpan;

    if (dimension > 0) {
      if (
        !(beforeMessage instanceof MetaMessage) ||
        !(afterMessage instanceof MetaMessage)
      ) {
        throw new Error(
          "Before or after message is not a MetaMessage at higher dimension",
        );
      }
      const beforeMessageMeta = beforeMessage as MetaMessage;
      const afterMessageMeta = afterMessage as MetaMessage;
      // For higher dimensions, extract the starting and ending message from within the meta-message's inner list
      span = {
        startId:
          beforeMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0]
            .id,
        endId:
          afterMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0].id,
        startIndex: this.originalMessages.findIndex(
          (candidateM) =>
            candidateM.id ===
            beforeMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0]
              .id,
        ),
        endIndex: this.originalMessages.findIndex(
          (candidateM) =>
            candidateM.id ===
            afterMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0]
              .id,
        ),
        originalSpan: {
          startId: beforeMessage.id,
          endId: afterMessage.id,
          startIndex: index,
          endIndex: index + 1,
        },
      };
    } else {
      // For base-level conversations, use the original message IDs and find their indices.
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

    // --- Constructing the Modular Prompt ---

    // 1. System Message: Core identity and immutable instructions.
    const systemMessage =
      this.config.customSystemInstruction &&
        this.config.customSystemInstruction.length > 0
        ? this.config.customSystemInstruction
        : `You are an expert conversation analyzer specializing in semantic turning point detection.
Your primary goal is to identify significant shifts in conversation flow and meaning.
Analyze semantic differences in the provided conversation context and provide a structured JSON output as described.`;

    // 2. Static Context User Message: Framework and evaluation criteria.
    const frameworkContextMessage = `<analysis_framework>
Turning points are significant shifts in conversation that indicate changes in subject, emotion, or decision-making.
A semantic distance of ${distance.toFixed(3)} has been detected between the messages.
You are analyzing dimension ${dimension} where ${dimension > 0 ? "each message represents a group of related turning points" : "messages are direct conversation exchanges"}.
Classification categories include:
 - Topic, Insight, Emotion, Meta-Reflection, Decision,
 - Question, Problem, Action, Clarification, Objection, Other.
Significance (0.0 to 1.0) reflects the impact of the turning point on the overall conversation.
</analysis_framework>

<output_format>
Your JSON response must include:
- label: (string, max 50 chars) a brief description,
- category: (string) one of the categories mentioned,
- keywords: (array of strings, max 4),
- quotes: (array of strings, max 3),
- emotionalTone: (string),
- sentiment: (one of "positive", "negative", "neutral"),
- significance: (number, 0.0 to 1.0),
- best_id: (string) the representative message ID.
</output_format>`;

    // 3. Dynamic Data User Message: Conversation context and messages to analyze.
    const contextualInfo = this.prepareContextualInfoMeta(
      beforeMessage,
      afterMessage,
      span,
      originalMessages,
      dimension,
      2,
      dimension > 0,
    );

    const dynamicDataMessage = `<conversation_context>
${contextualInfo}
</conversation_context>

<messages_to_analyze>
BEFORE MESSAGE:
  - Role: ${beforeMessage.author}
  - Content: ${returnFormattedMessageContent(this.config, beforeMessage, dimension)}
AFTER MESSAGE:
  - Role: ${afterMessage.author}
  - Content: ${returnFormattedMessageContent(this.config, afterMessage, dimension)}
</messages_to_analyze>`;

    // 4. Final Task Instruction User Message: Direct instruction to the LLM.
    const finalInstructionMessage =
      this.config.customUserInstruction &&
        this.config.customUserInstruction.length > 0
        ? this.config.customUserInstruction
        : `Using the criteria provided in <analysis_framework> and the detailed context in <conversation_context> along with the specific messages in <messages_to_analyze>, 
analyze whether the provided messages represent a turning point in the conversation.
Determine the category, significance, and other attributes as specified in <output_format>.
Return your answer as valid JSON.`;

    // Assemble all messages as a multi-message payload
    const messagesPayload: OpenAI.ChatCompletionMessageParam[] = [
      { role: "system", content: systemMessage },
      { role: "user", content: frameworkContextMessage },
      { role: "user", content: dynamicDataMessage },
      { role: "user", content: finalInstructionMessage },
    ];
    let classification: any = {};
    try {
      // Call the LLM using the assembled messages
      let classificationResponseStringContent: string | null = null;

      if (this.endpointType !== "ollama") {
        const response = await this.openai.chat.completions.create({
          model: this.config.classificationModel,
          messages: messagesPayload,
          temperature: this.config.temperature,
          response_format: formResponseFormatSchema(dimension, this.config),
          top_p: this.config.top_p,
        });

        classificationResponseStringContent =
          response.choices[0]?.message?.content || "{}";
      } else {
        const response = await this.ollama.chat({
          model: this.config.classificationModel,
          messages: messagesPayload.map((msg) => ({
            role: msg.role,
            content: String(msg.content),
          })),
          stream: false,
          format: formResponseFormatSchema(dimension, this.config).json_schema
            .schema,
          options: {
            temperature: this.config.temperature,
            top_p: this.config.top_p,
            top_k: 20,

            num_ctx: this.config.maxTokensPerChunk,
          },
        });

        // now try to json parse, if failure do the ame fallback
        classificationResponseStringContent = response?.message?.content ?? "";
      }

      if (classificationResponseStringContent) {
        classification = this.parseClassificationResponse(
          classificationResponseStringContent,
          span,
        );
      } else {
        // Fallback if no response content
        classification = {
          label: "No Response - Unclassified",
          category: "Other",
          keywords: [],
          emotionalTone: "neutral",
          sentiment: "neutral",
          significance: 0.0, // Lower significance for no response
          quotes: [],
          best_id: span.startId,
        };
      }

      // Validate and sanitize the LLM output.
      const validatedClassification = {
        label:
          typeof classification.label === "string"
            ? classification.label.substring(0, 50)
            : "Unknown Turning Point",
        category:
          typeof classification.category === "string"
            ? classification.category
            : "Other",
        keywords: Array.isArray(classification.keywords)
          ? classification.keywords.map(String).slice(0, 4)
          : [],
        emotionalTone:
          typeof classification.emotionalTone === "string"
            ? classification.emotionalTone
            : "neutral",
        sentiment: ["positive", "negative", "neutral"].includes(
          classification.sentiment,
        )
          ? classification.sentiment
          : "neutral",
        significance:
          typeof classification.significance === "number"
            ? Math.max(0, Math.min(1, classification.significance))
            : 0.5,
        quotes: Array.isArray(classification.quotes)
          ? classification.quotes.map(String).slice(0, 3)
          : [],
        best_id:
          typeof classification.best_id === "string"
            ? classification.best_id
            : span.startId,
      };

      // Calculate complexity score using the significance and the raw distance.
      const complexityScore = this.calculateComplexityScore(
        validatedClassification.significance,
        distance,
      );

      // Construct and return the final TurningPoint object.
      return {
        id: `tp-${dimension}-${span.startIndex}-${span.endIndex}`,
        label: validatedClassification.label,
        category: validatedClassification.category,
        span: span,
        semanticShiftMagnitude: distance,
        keywords: validatedClassification.keywords,
        quotes: validatedClassification.quotes,
        emotionalTone: validatedClassification.emotionalTone,
        sentiment: validatedClassification.sentiment,
        detectionLevel: dimension,
        significance: validatedClassification.significance,
        complexityScore: complexityScore,
      };
    } catch (err: any) {
      this.logger.info(
        `Error during LLM call for turning point classification: ${err.message}`,
      );
      if (this.config.throwOnError) {
        throw err;
      } else {
        return {
          id: `tp-err-${dimension}-${span.startId}`,
          label: "LLM Error - Unclassified",
          category: "Other",
          span: span,
          semanticShiftMagnitude: distance,
          keywords: [],
          quotes: [],
          emotionalTone: "neutral",
          sentiment: "neutral",
          detectionLevel: dimension,
          significance: 0.1,
          complexityScore: 1.0,
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

    this.logger.info(
      `Grouped categories:\n` + JSON.stringify(groupedByCategory, null, 2),
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
      console.info("created sectionMetageMessage");
      metaMessages.push(sectionMetaMessage);
    }

    this.logger.info(
      `Created ${metaMessages.length
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
  private filterSignificantTurningPoints(
    turningPoints: TurningPoint[],
  ): TurningPoint[] {
    if (
      !this.config.onlySignificantTurningPoints ||
      turningPoints.length === 0
    ) {
      // Ensure sorted return even if not filtering
      return turningPoints.sort(
        (a, b) => a.span.startIndex - b.span.startIndex,
      );
    }

    this.logger.info(
      `Filtering ${turningPoints.length} TPs based on significance >= ${this.config.significanceThreshold} and maxPoints = ${this.config.maxTurningPoints}`,
    );

    // Sort by significance, complexity, magnitude
    const sorted = [...turningPoints].sort((a, b) => {
      if (b.significance !== a.significance)
        return b.significance - a.significance;
      if (b.complexityScore !== a.complexityScore)
        return b.complexityScore - a.complexityScore;
      return b.semanticShiftMagnitude - a.semanticShiftMagnitude;
    });

    const result: TurningPoint[] = [];
    const coveredIndices: Set<number> = new Set(); // Use indices for overlap check
    const maxPoints = this.config.maxTurningPoints;

    for (const tp of sorted) {
      // Check significance threshold first
      if (tp.significance < this.config.significanceThreshold) {
        // Only consider points below threshold if we haven't found enough significant ones yet
        if (result.length >= Math.ceil(maxPoints / 2)) {
          // Heuristic: if we have half the max points, stop adding insignificant ones
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
        this.logger.info(
          `    TP ${tp.id} (Sig: ${tp.significance.toFixed(2)}) overlaps significantly (${(overlapRatio * 100).toFixed(0)}%) with existing TPs. Skipping.`,
        );
      } else if (result.length >= maxPoints) {
        this.logger.info(
          `    Reached max turning points (${maxPoints}). Skipping TP ${tp.id}.`,
        );
      }
    }

    // Ensure at least one TP is returned if any were found initially
    if (result.length === 0 && sorted.length > 0) {
      this.logger.info(
        "No TPs met significance/overlap criteria, returning the single most significant one.",
      );
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
            if (
              j >= result[0].span.startIndex &&
              j <= result[0].span.endIndex
            ) {
              overlapsFirst = true;
              break;
            }
          }
          if (!overlapsFirst) {
            this.logger.info(
              "Adding a second, non-overlapping TP for diversity.",
            );
            result.push(nextTp);
            break;
          }
        }
      }
    }

    this.logger.info(
      `Filtered down to ${result.length} significant turning points.`,
    );
    // Final sort by position in conversation
    return result.sort((a, b) => a.span.startIndex - b.span.startIndex);
  }

  /**
   * Combine turning points from different dimensions
   * (Using original logic from the second code block)
   */
  private combineTurningPoints(
    localTurningPoints: TurningPoint[],
    higherDimensionTurningPoints: TurningPoint[],
  ): TurningPoint[] {
    this.logger.info(
      `Combining ${localTurningPoints.length} local (dim ${localTurningPoints[0]?.detectionLevel ?? "N/A"}) and ${higherDimensionTurningPoints.length} higher (dim ${higherDimensionTurningPoints[0]?.detectionLevel ?? "N/A"}) TPs.`,
    );

    // Prioritize higher-dimensional turning points by boosting their significance (original logic)
    const boostedHigher = higherDimensionTurningPoints.map((tp) => ({
      ...tp,
      // Apply a boost, ensuring it doesn't exceed 1.0
      significance: Math.min(1.0, tp.significance * 1.2), // Adjusted boost factor slightly
      // Keep original detectionLevel for merging logic
    }));

    // Combine all turning points
    const allTurningPoints = [...localTurningPoints, ...boostedHigher];
    this.logger.info(
      `Total TPs before cross-level merge: ${allTurningPoints.length}`,
    );

    // Merge overlapping turning points across dimensions, prioritizing higher dimensions/significance
    const mergedTurningPoints = this.mergeAcrossLevels(allTurningPoints);
    this.logger.info(
      `Merged across levels to ${mergedTurningPoints.length} TPs.`,
    );

    // Filter the combined & merged list to keep the most significant ones overall
    const filteredTurningPoints =
      this.filterSignificantTurningPoints(mergedTurningPoints);

    this.logger.info(
      `Final combined and filtered TPs: ${filteredTurningPoints.length}`,
    );
    // Sort by position in conversation before returning
    return filteredTurningPoints.sort(
      (a, b) => a.span.startIndex - b.span.startIndex,
    );
  }

  /**
   * Merge similar or overlapping turning points *within* the same dimension
   * (Using original logic from the second code block)
   */
  private mergeSimilarTurningPoints(
    turningPoints: TurningPoint[],
  ): TurningPoint[] {
    if (turningPoints.length <= 1) return turningPoints;

    // Sort turning points by start index
    const sorted = [...turningPoints].sort(
      (a, b) => a.span.startIndex - b.span.startIndex,
    );
    const merged: TurningPoint[] = [];
    let currentTp = sorted[0]; // Use a more descriptive name

    for (let i = 1; i < sorted.length; i++) {
      const nextTp = sorted[i];

      // Check conditions for merging (original logic)
      const isOverlapping =
        nextTp.span.startIndex <= currentTp.span.endIndex + 2; // Allow small gap
      const isSimilarCategory = nextTp.category === currentTp.category;
      // Added closeness check from original code
      const hasCloseIndices =
        nextTp.span.startIndex - currentTp.span.endIndex <= 3;

      // Merge if overlapping OR close, AND same category
      if ((isOverlapping || hasCloseIndices) && isSimilarCategory) {
        this.logger.info(
          `    Merging similar TPs (Dim ${currentTp.detectionLevel}): ${currentTp.id} and ${nextTp.id}`,
        );
        // Merge the turning points
        const newLabel = this.createMergedLabel(currentTp.label, nextTp.label);

        // Create merged span (min start, max end)
        const mergedSpan = this.ensureChronologicalSpan({
          startId:
            currentTp.span.startIndex <= nextTp.span.startIndex
              ? currentTp.span.startId
              : nextTp.span.startId,
          endId:
            currentTp.span.endIndex >= nextTp.span.endIndex
              ? currentTp.span.endId
              : nextTp.span.endId,
          startIndex: Math.min(
            currentTp.span.startIndex,
            nextTp.span.startIndex,
          ),
          endIndex: Math.max(currentTp.span.endIndex, nextTp.span.endIndex),
        });

        // Update the deprecated span too (original logic, though less relevant now)
        // Note: deprecatedSpan might not exist if TPs came from meta-messages
        const mergedDeprecatedSpan =
          currentTp.deprecatedSpan && nextTp.deprecatedSpan
            ? {
              startIndex: Math.min(
                currentTp.deprecatedSpan.startIndex,
                nextTp.deprecatedSpan.startIndex,
              ),
              endIndex: Math.max(
                currentTp.deprecatedSpan.endIndex,
                nextTp.deprecatedSpan.endIndex,
              ),
              startMessageId:
                mergedSpan.startIndex === currentTp.deprecatedSpan.startIndex
                  ? currentTp.deprecatedSpan.startMessageId
                  : nextTp.deprecatedSpan.startMessageId,
              endMessageId:
                mergedSpan.endIndex === currentTp.deprecatedSpan.endIndex
                  ? currentTp.deprecatedSpan.endMessageId
                  : nextTp.deprecatedSpan.endMessageId,
            }
            : undefined; // Handle cases where deprecatedSpan might be missing

        // Combine keywords and quotes (unique, limited)
        const mergedKeywords = Array.from(
          new Set([...(currentTp.keywords || []), ...(nextTp.keywords || [])]),
        ).slice(0, 5);
        const mergedQuotes = Array.from(
          new Set([...(currentTp.quotes || []), ...(nextTp.quotes || [])]),
        ).slice(0, 3); // Limit quotes too

        // Update the current TP to be the merged version
        currentTp = {
          ...currentTp, // Keep most properties of the first TP
          id: `${currentTp.id}-merged-${nextTp.span.startIndex}`, // Indicate merge in ID
          label: newLabel,
          span: mergedSpan,
          // Only include deprecatedSpan if it was successfully merged
          ...(mergedDeprecatedSpan && { deprecatedSpan: mergedDeprecatedSpan }),
          semanticShiftMagnitude:
            (currentTp.semanticShiftMagnitude + nextTp.semanticShiftMagnitude) /
            2,
          keywords: mergedKeywords,
          quotes: mergedQuotes,
          // Boost significance slightly, cap at 1.0 (original logic)
          significance: Math.min(
            1.0,
            ((currentTp.significance + nextTp.significance) / 2) * 1.1,
          ),
          // Take max complexity (original logic)
          complexityScore: Math.max(
            currentTp.complexityScore,
            nextTp.complexityScore,
          ),
          // Combine emotional tone/sentiment logically (e.g., take the one from the more significant TP)
          emotionalTone:
            currentTp.significance >= nextTp.significance
              ? currentTp.emotionalTone
              : nextTp.emotionalTone,
          sentiment:
            currentTp.significance >= nextTp.significance
              ? currentTp.sentiment
              : nextTp.sentiment,
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
      if (b.detectionLevel !== a.detectionLevel)
        return b.detectionLevel - a.detectionLevel;
      // Add secondary sort by significance within the same level
      if (b.significance !== a.significance)
        return b.significance - a.significance;
      return a.span.startIndex - b.span.startIndex; // Tertiary sort by position
    });

    const merged: TurningPoint[] = [];
    // Use a Set of covered *indices* for more granular overlap checking
    const coveredIndices: Set<number> = new Set();

    this.logger.info(
      `    Merging across levels. Input count: ${sorted.length}. Prioritizing higher dimension/significance.`,
    );

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
        this.logger.info(
          `      Keeping TP ${tp.id} (Dim ${tp.detectionLevel}, Sig ${tp.significance.toFixed(2)}). Overlap: ${(overlapRatio * 100).toFixed(0)}%`,
        );
      } else {
        this.logger.info(
          `      Skipping TP ${tp.id} (Dim ${tp.detectionLevel}, Sig ${tp.significance.toFixed(2)}) due to significant overlap (${(overlapRatio * 100).toFixed(0)}%).`,
        );
      }
    }

    this.logger.info(
      `    Finished merging across levels. Output count: ${merged.length}.`,
    );
    // Sort the final result by position in conversation
    return merged.sort((a, b) => a.span.startIndex - b.span.startIndex);
  }

  /**
   * Create a merged label (Using original logic)
   */
  private createMergedLabel(label1: string, label2: string): string {
    if (label1 === label2) return label1;
    if (label1.includes("Unclassified")) return label2;
    if (label2.includes("Unclassified")) return label1;

    const commonWords = this.findCommonWords(label1, label2);
    if (commonWords.length > 0) {
      // Simple common word approach (original)
      return commonWords.join(" ") + " Discussion";
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
    const stopwords = new Set([
      "to",
      "the",
      "in",
      "of",
      "and",
      "a",
      "an",
      "on",
      "for",
      "with",
      "shift",
      "discussion",
      "about",
      "summary",
    ]); // Added more stopwords
    // Filter common words, exclude stopwords, ensure decent length
    return words1.filter(
      (word) =>
        word.length > 3 && words2.includes(word) && !stopwords.has(word),
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
  private isSpanOverlapping(
    tp: TurningPoint,
    coveredSpans: Set<string>,
  ): boolean {
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
  private async generateMessageEmbeddings(
    messages: Message[],
    dimension = 0,
  ): Promise<MessageEmbedding[]> {
    const embeddings: MessageEmbedding[] = new Array(messages.length);
    // Using original concurrency limit of 4
    // console.info(`Generating embeddings for ${messages.length} messages with dimension ${dimension}.`);
    await async.eachOfLimit(
      messages,
      this.config.embeddingConcurrency,
      async (message, indexStr) => {
        let candidateText = message.message;
        if (dimension > 0 && message instanceof MetaMessage) {
          // For meta-messages, use the original message text
          const metaMessage = message as MetaMessage;
          const messagesWithinMeta =
            metaMessage.getMessagesInTurningPointSpanToMessagesArray();
          // naiviely concatenate the last and first message
          candidateText = `${messagesWithinMeta[0].message} ${messagesWithinMeta[messagesWithinMeta.length - 1].message}`;
          console.info(
            `Meta message ${message.id} contains ${messagesWithinMeta.length} messages.`,
          );
        }
        const index = Number(indexStr);
        try {
          const embedding = await this.getEmbedding(candidateText);
          // Store the original index from the input 'messages' array
          embeddings[index] = {
            id: message.id,
            index: index, // Store the index within the current chunk/message list being processed
            embedding,
          };
        } catch (error: any) {
          this.logger.info(
            `Error generating embedding for msg ${message.id} at index ${index}: ${error.message}. Creating zero vector.`,
          );
          const embeddingSize = 1536; // Assuming text-embedding-3-small
          embeddings[index] = {
            id: message.id,
            index: index,
            embedding: new Float32Array(embeddingSize).fill(0),
          };
        }
      },
    );
    // Filter out any potential null/undefined entries if errors occurred
    return embeddings.filter((e) => e);
  }

  /**
   * Retrieves the embedding vector for a given text string using the configured embedding model.
   *
   * This method implements intelligent token management, crypto-hashed caching, and multi-provider
   * support for embedding generation within the ARC/CRA framework.
   *
   * @param text - The input text to generate embeddings for. Will be automatically truncated if exceeding token limits.
   * @param naiveTokenLimit - Maximum number of tokens allowed before truncation (default: 8192).
   *                          Uses a character-to-token ratio estimation for efficient preprocessing.
   *
   * @returns Promise<Float32Array> - The embedding vector as a Float32Array. Dimension depends on the model:
   *   - text-embedding-3-small: 1536 dimensions
   *   - text-embedding-3-large: 3072 dimensions
   *   - Other models: 1024 dimensions (fallback)
   *
   * @throws Will not throw errors directly, but logs warnings and returns zero vectors on API failures.
   *
   * @remarks
   * **Token Management:**
   * - Implements dynamic text truncation based on token counting to prevent API errors
   * - Uses character-to-token ratio estimation for efficient preprocessing
   * - Recalculates ratio iteratively until under the specified limit
   *
   * **Caching Strategy:**
   * - Uses SHA-256 crypto hashing for cache keys to avoid issues with special characters and long text
   * - Cache keys include both model name and (truncated) text content for uniqueness
   * - Leverages the class's LRU cache with configurable RAM limits via `embeddingCacheRamLimitMB`
   * - Cache operations are logged in debug mode for monitoring effectiveness
   *
   * **Multi-Provider Support:**
   * - Supports OpenAI API and OpenAI-compatible endpoints (Ollama, LM Studio, etc.)
   * - Uses EMBEDDINGS_API_KEY environment variable first, falls back to OPENAI_API_KEY
   * - Respects the configured `embeddingEndpoint` for custom embedding providers
   * - Maintains compatibility with standard OpenAI embedding response format
   *
   * **Error Handling:**
   * - Returns zero vectors instead of throwing on API failures
   * - Logs detailed error information for debugging
   * - Gracefully handles invalid API responses or network issues
   * - Maintains system stability during transient embedding service outages
   *
   * **Integration with ARC/CRA Framework:**
   * - Critical component for semantic distance calculations in turning point detection
   * - Supports both base message embeddings (dimension 0) and meta-message embeddings (dimension > 0)
   * - Cache performance directly impacts overall framework processing speed
   * - Embedding quality affects the accuracy of semantic shift detection
   */

  async getEmbedding(
    text: string,
    naiveTokenLimit = 8192,
  ): Promise<Float32Array> {
    // Ensure that the input text length is less than 8192 tokens
    let tokensCount = countTokens(text);
    let textCharToTokenRatio = text.length / tokensCount;

    while (tokensCount > naiveTokenLimit) {
      // Remove the exact number of characters to get the token count under limit
      const charsToRemove = Math.ceil(
        (tokensCount - naiveTokenLimit) * textCharToTokenRatio,
      );
      text = text.substring(0, text.length - charsToRemove);
      tokensCount = countTokens(text);
    }

    // Create crypto hash cache key AFTER text truncation (important!)
    const cacheKey = crypto
      .createHash("sha256")
      .update(`${this.config.embeddingModel}:${text}`)
      .digest("hex");

    // Check cache first
    const cachedEmbedding = this.embeddingCache.get(cacheKey);
    if (cachedEmbedding) {
      if (this.config.debug) {
        this.logger.debug(
          `Cache hit for embedding (${cacheKey.substring(0, 8)}...)`,
        );
      }
      return cachedEmbedding;
    }

    try {
      // Create OpenAI client with proper configuration
      const openai = new OpenAI({
        apiKey: process.env.EMBEDDINGS_API_KEY ?? process.env.OPENAI_API_KEY,
        baseURL: this.config.embeddingEndpoint,
      });

      const response = await openai.embeddings.create({
        model: this.config.embeddingModel,
        input: text,
        encoding_format: "float",
      });

      if (
        response.data &&
        response.data.length > 0 &&
        response.data[0].embedding
      ) {
        const embedding = new Float32Array(response.data[0].embedding);

        // Store in cache with crypto hash key
        this.embeddingCache.set(cacheKey, embedding);

        // if (this.config.debug) { no need for this its too muc hnoise
        //   this.logger.info(
        //     `Cache miss - stored new embedding (${cacheKey.substring(0, 8)}...)`,
        //   );
        // }

        return embedding;
      } else {
        throw new Error("Invalid embedding response structure from OpenAI");
      }
    } catch (err: any) {
      this.logger.info(
        `Error generating embedding: ${err.message}. Returning zero vector.`,
      );
      const embeddingSize =
        this.config.embeddingModel === "text-embedding-3-small"
          ? 1536
          : this.config.embeddingModel === "text-embedding-3-large"
            ? 3072
            : 1024;
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
  private async chunkConversation(
    messages: Message[],
    dimension = 0,
  ): Promise<ChunkingResult> {
    const chunks: Message[][] = [];
    // scale down the min messages based on each depth increase
    const baseMinMessages = this.config.minMessagesPerChunk; // Preserve original value
    const dimensionScaleFactor = Math.max(0.1, Math.pow(0.35, dimension));
    const minMessages = Math.max(
      2,
      Math.round(baseMinMessages * dimensionScaleFactor),
    );
    // But you need to scale maxTokensPerChunk too
    const tokenScaleFactor = Math.max(0.2, Math.pow(0.5, dimension));
    const maxTokens = Math.max(
      this.config.minTokensPerChunk,
      Math.round(this.config.maxTokensPerChunk * tokenScaleFactor),
    );

    // Handle case where input has fewer than minimum messages (original logic)
    if (messages.length < minMessages) {
      this.logger.info(
        `Input messages (${messages.length}) less than minMessagesPerChunk (${minMessages}). Returning as single chunk.`,
      );
      // Return single chunk only if it's not empty
      return {
        chunks: messages.length > 0 ? [[...messages]] : [],
        numChunks: messages.length > 0 ? 1 : 0,
        avgTokensPerChunk:
          messages.length > 0
            ? await this.getMessageArrayTokenCount(messages)
            : 0,
      };
    }

    let currentChunk: Message[] = [];
    let currentTokens = 0;
    let totalTokens = 0;
    const overlapSize = Math.max(0, Math.min(2, Math.floor(minMessages / 2))); // Keep overlap small, max 2

    // Ideal chunk size logic (original) - helps guide chunking beyond just token limits
    const idealMessageCount = Math.max(
      minMessages,
      Math.min(
        10,
        Math.ceil(
          messages.length / Math.max(1, Math.floor(messages.length / 10)),
        ),
      ), // Aim for ~10 chunks max?
    );
    this.logger.info(
      `    Chunking ${messages.length} messages. MinMsgs: ${minMessages}, MaxTokens: ${maxTokens}, IdealMsgCount: ${idealMessageCount}, Overlap: ${overlapSize}`,
    );

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
      const approachingMaxTokens = currentTokens >= maxTokens * 0.9; // Increase threshold slightly
      const isLastMessage = i === messages.length - 1;
      const significantlyOverMaxTokens = currentTokens > maxTokens * 1.1; // Check if significantly over

      // Close chunk conditions (refined slightly)
      // 1. Last message: always close.
      // 2. Min messages met AND (approaching/over max tokens OR reached ideal size)
      if (
        isLastMessage ||
        (hasMinMessages &&
          (approachingMaxTokens || hasIdealSize || significantlyOverMaxTokens))
      ) {
        // Add the chunk
        chunks.push([...currentChunk]);
        this.logger.info(
          `      Created chunk ${chunks.length} with ${currentChunk.length} messages, ${currentTokens} tokens. Ends at index ${i}.`,
        );

        // If not the last message, start next chunk with overlap
        if (!isLastMessage) {
          const startIndexForNextChunk = Math.max(
            0,
            currentChunk.length - overlapSize,
          );
          currentChunk = currentChunk.slice(startIndexForNextChunk);
          currentTokens = await this.getMessageArrayTokenCount(currentChunk);
          this.logger.info(
            `      Starting next chunk with ${currentChunk.length} overlapping messages, ${currentTokens} tokens.`,
          );
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
      if (
        midPointIndex >= minMessages &&
        singleChunk.length - midPointIndex >= minMessages
      ) {
        this.logger.info(
          "    Single chunk detected for large conversation, attempting to split.",
        );
        const firstChunk = singleChunk.slice(0, midPointIndex);
        // Apply overlap when splitting
        const secondChunkStartIndex = Math.max(0, midPointIndex - overlapSize);
        const secondChunk = singleChunk.slice(secondChunkStartIndex);

        chunks.splice(0, 1, firstChunk, secondChunk); // Replace single chunk with two
        this.logger.info(
          `    Successfully split into two chunks: ${firstChunk.length} msgs and ${secondChunk.length} msgs.`,
        );
      }
    }

    // Ensure all chunks meet minimum message requirement, merging small trailing chunks if necessary
    if (chunks.length > 1) {
      const lastChunk = chunks[chunks.length - 1];
      if (lastChunk.length < minMessages) {
        this.logger.info(
          `    Last chunk (${lastChunk.length} msgs) is smaller than min size (${minMessages}). Merging with previous.`,
        );
        const secondLastChunk = chunks[chunks.length - 2];
        // Combine, avoiding duplicates from overlap if possible
        const combinedChunk = [...secondLastChunk];
        const lastIdInSecondLast =
          secondLastChunk[secondLastChunk.length - 1]?.id;
        let appendStartIndex = 0;
        if (overlapSize > 0 && lastChunk[0]?.id === lastIdInSecondLast) {
          appendStartIndex = 1; // Skip first element of last chunk if it's the same as end of previous
        }
        combinedChunk.push(...lastChunk.slice(appendStartIndex));

        // Replace the last two chunks with the merged one
        chunks.splice(chunks.length - 2, 2, combinedChunk);
        this.logger.info(
          `    Merged last two chunks. New chunk count: ${chunks.length}.`,
        );
      }
    }
    // --- End Post-processing ---

    const numChunks = chunks.length;
    const avgTokens = numChunks > 0 ? totalTokens / numChunks : 0; // Avoid division by zero
    const avgMessagesPerChunk = numChunks > 0 ? messages.length / numChunks : 0;

    this.logger.info(
      `    Finished chunking. Created ${numChunks} chunks. Avg Tokens: ${avgTokens.toFixed(0)}, Avg Msgs: ${avgMessagesPerChunk.toFixed(1)}`,
    );

    return {
      chunks,
      numChunks,
      avgTokensPerChunk: avgTokens,
    };
  }

  /**
   * Ensures a message span is in chronological order by index
   * (Using original logic)
   */
  private ensureChronologicalSpan(span: MessageSpan): MessageSpan {
    if (span.startIndex > span.endIndex) {
      this.logger.info(
        `Warning: Correcting reversed span indices (${span.startIndex} > ${span.endIndex}) for IDs ${span.startId}/${span.endId}.`,
      );
      // Create a new span with swapped values to maintain immutability
      return {
        startId: span.endId, // Swap IDs
        endId: span.startId,
        startIndex: span.endIndex, // Swap indices
        endIndex: span.startIndex,
        originalSpan: span.originalSpan ?? span, // Store original if needed
      };
    }
    return span; // Return original if already chronological
  }

  /**
   * Get token count for a message with caching (Using original logic)
   */
  private async getMessageTokenCount(text: string): Promise<number> {
    const hash = crypto.createHash("sha256").update(text).digest("hex");
    if (tokenCountCache.has(hash)) {
      return tokenCountCache.get(hash)!;
    }

    let count: number;
    try {
      // Use external countTokens function (original logic)
      count = countTokens(text);
    } catch (err: any) {
      this.logger.info(
        `Error counting tokens: ${err.message}. Falling back to length/4.`,
      );
      /**
       * Fallback (original logic) based on a naive approach that uses a ratio of four characters per token. This method is inaccurate because the token count is also influenced by certain special strings or characters.
       * - The ratio can vary significantly depending on the type of text content; for example, JSON data may yield a slightly different ratio. However, this four-character ratio is generally reasonable for semantic text.
       */
      count = Math.ceil(text.length / 4);
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
    if (
      dimension > 0 ||
      _beforeMessage instanceof MetaMessage ||
      _afterMessage instanceof MetaMessage
    ) {
      throw new Error(
        "Contextual information preparation is not supported for dimensions greater than 0.",
      );
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

  /**
   * Validates turning point categories configuration
   * Ensures each category has required fields and logs warnings for issues
   */
  private validateTurningPointCategories(
    categories?: TurningPointCategory[],
  ): TurningPointCategory[] {
    // If no categories provided, use defaults
    if (!categories || categories.length === 0) {
      return turningPointCategories;
    }

    // Check if too many categories
    if (categories.length > 15) {
      this.logger.warn(
        `Warning: ${categories.length} turning point categories provided. ` +
        `Maximum recommended is 15. Consider reducing for better LLM performance.`,
      );
    }

    const validatedCategories: TurningPointCategory[] = [];
    const seenCategories = new Set<string>();
    const warningClose = `Proceeding anyway - if this was intentional for testing purposes, you can ignore this warning.`;
    categories.forEach((categoryConfig, index) => {
      // Check if category config exists and is an object
      if (!categoryConfig || typeof categoryConfig !== "object") {
        this.logger.warn(
          `Warning: Invalid category configuration at index ${index}. ` +
          `Expected object with 'category' and 'description' properties but found ${JSON.stringify(categoryConfig)}. ${warningClose}`,
        );
      }

      const { category, description } = categoryConfig;

      // Validate category field exists and is a string
      if (
        !category ||
        typeof category !== "string" ||
        category.trim().length === 0
      ) {
        this.logger.warn(
          `Warning: Missing or invalid 'category' field at index ${index}. ` +
          `Expected non-empty string. Using anyway with fallback value "unknown". ${warningClose}`,
        );
      }

      // Validate description field exists and is a string
      if (
        !description ||
        typeof description !== "string" ||
        description.trim().length === 0
      ) {
        this.logger.warn(
          `Warning: Missing or invalid 'description' field for category "${category}" at index ${index}. ` +
          `Expected non-empty string, but got ${JSON.stringify(description)}. Using anyway with fallback. ${warningClose}`,
        );
      }

      const trimmedCategory = category?.trim() || "unknown";
      const trimmedDescription =
        description?.trim() || "[no description provided]";

      // Check for duplicate categories (case-insensitive)
      const categoryLower = trimmedCategory.toLowerCase();
      if (seenCategories.has(categoryLower)) {
        this.logger.warn(
          `Warning: Duplicate category "${trimmedCategory}" found at index ${index}. ` +
          `Categories should be unique. Using anyway. ${warningClose}`,
        );
      }

      // Check if category name has more than two words
      const wordCount = trimmedCategory.split(/\s+/).length;
      if (wordCount > 2) {
        this.logger.warn(
          `Warning: Category "${trimmedCategory}" at index ${index} has ${wordCount} words. ` +
          `Consider using 1-2 words for better categorization. Using anyway. ${warningClose}`,
        );
      }

      // Add to seen categories set (even if duplicate)
      seenCategories.add(categoryLower);

      // ALWAYS add the category, even if it's fucked up
      validatedCategories.push({
        category: trimmedCategory,
        description: trimmedDescription,
      });
    });
    // Log what we're using
    if (this.config?.debug) {
      this.logger.info(
        `Using ${validatedCategories.length} turning point categories: ${validatedCategories.map((c) => c.category).join(", ")}`,
      );
    }

    return validatedCategories;
  }
  /**
   * Check if an endpoint is running Ollama by checking the root response
   */
  private async isOllamaEndpoint(endpoint: string): Promise<boolean> {
    try {
      // Remove trailing /v1 or other paths to get base URL
      const baseUrl = endpoint.replace(/\/v1\/?$/, "").replace(/\/$/, "");

      const response = await fetch(baseUrl, {
        method: "GET",
      });

      if (response.ok) {
        const text = await response.text();
        return text.includes("Ollama is running");
      }

      return false;
    } catch (error) {
      // If fetch fails, assume it's not Ollama
      return false;
    }
  }

  /**
   * Processes the raw string content received from an LLM classification call
   * and attempts to parse it into a structured JSON object.
   *
   * This method implements a robust parsing strategy:
   * 1. It first attempts a direct `JSON.parse()` on the input string.
   * 2. If direct parsing fails, it looks for a JSON object embedded within
   *    a markdown-style code block (e.g., ```json\n{...}\n```) and tries to parse that.
   * 3. If that also fails, it attempts a more lenient match for any string
   *    that starts with `{` and ends with `}` and tries to parse that.
   * 4. If all parsing attempts fail, or if the resulting object is empty,
   *    it returns a default "Parsing Error - Unclassified" object. This ensures
   *    that the subsequent processing steps always receive an object with expected
   *    (though potentially default) properties.
   *
   * The `span` parameter is used to provide a default `best_id` in the
   * fallback classification object if parsing fails, ensuring some contextual
   * link back to the original messages being analyzed.
   *
   * @param content The raw string content from the LLM response, expected to contain JSON.
   * @param span The MessageSpan object corresponding to the messages being classified.
   *             Used for providing a fallback `best_id` if parsing fails.
   * @returns An `any` object representing the parsed classification. This object
   *          will have a defined structure if parsing is successful, or a default
   *          error structure if all parsing attempts fail.
   */
  parseClassificationResponse(content: string, span: MessageSpan): any {
    let classification = {};
    try {
      classification = JSON.parse(content);
    } catch (err: any) {
      this.logger.info("Error parsing LLM response as JSON:", err.message);
      const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
      if (jsonMatch && jsonMatch[1]) {
        try {
          classification = JSON.parse(jsonMatch[1]);
          this.logger.info("Successfully extracted JSON from markdown block.");
        } catch (parseErr: any) {
          this.logger.info("Failed to parse extracted JSON:", parseErr.message);
          classification = {};
        }
      } else {
        const plainJsonMatch = content.match(/\{[\s\S]*\}/);
        if (plainJsonMatch) {
          try {
            classification = JSON.parse(plainJsonMatch[0]);
            this.logger.info("Successfully extracted JSON using simple match.");
          } catch (parseErr: any) {
            this.logger.info(
              "Failed to parse JSON using simple match:",
              parseErr.message,
            );
            classification = {};
          }
        } else {
          this.logger.info("Could not extract JSON from response:", content);
          classification = {};
        }
      }
      if (Object.keys(classification).length === 0) {
        classification = {
          label: "Parsing Error - Unclassified",
          category: "Other",
          keywords: [],
          emotionalTone: "neutral",
          sentiment: "neutral",
          significance: 0.1,
          quotes: [],
          best_id: span.startId, // Use span from the calling context
        };
      }
    }
    return classification;
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
  const conversationPariah = fs.readJsonSync("src/conversationPariah.json", {
    encoding: "utf-8",
  }) as Message[];

  // Load conversation data from a JSON file
  // Calculate adaptive recursion depth based on conversation length
  // This directly implements the ARC concept of adaptive dimensional analysis
  const determineRecursiveDepth = (messages: Message[]) => {
    return Math.floor(messages.length / thresholdForMinDialogueShift);
  };

  const startTime = new Date().getTime();

  // Create detector with configuration based on the ARC/CRA framework
  const detector = new SemanticTurningPointDetector({
    /**
     * If you wish to not pass variables, you can set the environment variable LLM_API_KEY in case you are using a different endpoint that is external, or set the OPENAI_API_KEY environment variable to use the default OpenAI API endpoint, by default if a new OpenAI client is created without apiKey set, it looks for one as the `OPENAI_API_KEY` environment variable.
     */
    apiKey: process.env.OPENAI_API_KEY || "",

    // Dynamic configuration based on conversation complexity
    semanticShiftThreshold: 0.94,
    minTokensPerChunk: 1024,
    maxTokensPerChunk: 8192,

    // ARC framework: dynamic recursion depth based on conversation complexity
    maxRecursionDepth: Math.min(determineRecursiveDepth(conversationPariah), 5),

    /**
     * Setting this to false means that maxTurningPoints will have little effect as a boundary, but still influence the size of results.
     * - To ensure maxTurningPoints is respected, set this to true. To see more than just significant turning points, set it to false.
     */
    onlySignificantTurningPoints: true,
    significanceThreshold: 0.95,

    // ARC framework: chunk size scales with complexity
    minMessagesPerChunk: Math.ceil(
      determineRecursiveDepth(conversationPariah) * 3.5,
    ),

    // ARC framework: number of turning points scales with conversation length
    // maxTurningPoints: Math.max(6, Math.round(conversationPariah.length / 20)),

    // for sake of demostration, between llm models, we set it hardcoded to 16
    maxTurningPoints: 16,

    // CRA framework: explicit complexity saturation threshold for dimensional escalation
    complexitySaturationThreshold: 4.1,

    // if using openai set it to 4 or more.
    concurrency: 1,
    max_character_length: 4000,

    // Enable convergence measurement for ARC analysis
    measureConvergence: true,
    customSystemInstruction: `You are an expert conversation analyzer specializing in semantic turning point detection.
Your primary goal is to identify significant shifts in conversation flow and meaning.
Analyze semantic differences in the provided conversation context and provide a structured JSON output as described\n  `,

    classificationModel: 'qwen3:4b',


    /**
     * Setting a custom endpoint overrides the default`api.openai.com/v1` endpoint. Allowing for usage of other llm providers that follow the same API structure. Since the Semantic TUrning Point does utilize advanced parameters, namely `format`, or response format, in which instructs the response to be returned as a JSON Schema object, not all openai compatible methods will support this. The examples below all support formatted responses. Semantic Turning Point also does not utilize tool calls.
     *
     *  Some examples include:
     * - Ollama
     * - OpenRouter
     * - vLLM
     * - LM Studio
     * - Text Generation API
     *
     */
    endpoint: 'http://localhost:11434/v1', // e.g. ' (Ollama).

    // embeddingEndpoint: "http://10.3.28.33:7756/v1", // this one points to a LMStudio instance, Apple silicon laptops/macs are quite powerful for this.
    // embeddingModel: "text-embedding-snowflake-arctic-embed-l-v2.0",
    embeddingModel: 'text-embedding-3-large',
    debug: true,
  });

  try {
    // Detect turning points using the ARC/CRA framework
    const tokensInConvoFile =
      await detector.getMessageArrayTokenCount(conversationPariah);
    const turningPointResult =
      await detector.detectTurningPoints(conversationPariah);

    const turningPoints = turningPointResult.points;
    const confidenceScore = turningPointResult.confidence;
    const endTime = new Date().getTime();
    const difference = endTime - startTime;
    const formattedTimeDateDiff = new Date(difference)
      .toISOString()
      .slice(11, 19);

    console.info(
      `\nTurning point detection took as MM:SS: ${formattedTimeDateDiff} for ${tokensInConvoFile} tokens in the conversation`,
    );

    // Display results with complexity scores from the ARC framework
    console.info("\n=== DETECTED TURNING POINTS (ARC/CRA Framework) ===\n");
    console.info(
      `Detected ${turningPoints.length} turning points with a confidence score of ${confidenceScore.toFixed(2)} using model ${detector.getModelName()}.`,
    );

    turningPoints.forEach((tp, i) => {
      detector.logger.info(`${i + 1}. ${tp.label} (${tp.category})`);
      detector.logger.info(
        `   Messages: "${tp.span.startId}" → "${tp.span.endId}"`,
      );
      detector.logger.info(`   Dimension: n=${tp.detectionLevel}`);
      detector.logger.info(
        `   Complexity Score: ${tp.complexityScore.toFixed(2)} of 5`,
      );
      detector.logger.info(
        `   Emotional Tone: ${tp.emotionalTone || "unknown"}`,
      );
      detector.logger.info(
        `   Semantic Shift Magnitude: ${tp.semanticShiftMagnitude.toFixed(2)}`,
      );
      detector.logger.info(`   Sentiment: ${tp.sentiment || "unknown"}`);
      detector.logger.info(`   Significance: ${tp.significance.toFixed(2)}`);
      detector.logger.info(`   Keywords: ${tp.keywords?.join(", ") || "none"}`);
      detector.logger.info(`   Quotes: ${tp.quotes?.join(", ") || "none"}`);
    });

    // Get and display convergence history to demonstrate the ARC framework
    const convergenceHistory = detector.getConvergenceHistory();

    detector.logger.info("\n=== ARC/CRA FRAMEWORK CONVERGENCE ANALYSIS ===\n");
    convergenceHistory.forEach((state, i) => {
      detector.logger.info(`Iteration ${i + 1}:`);
      detector.logger.info(`  Dimension: n=${state.dimension}`);
      detector.logger.info(
        `  Convergence Distance: ${state.distanceMeasure.toFixed(3)}`,
      );
      detector.logger.info(
        `  Dimensional Escalation: ${state.didEscalate ? "Yes" : "No"}`,
      );
      detector.logger.info(
        `  Turning Points: ${state.currentTurningPoints.length}`,
      );
    });

    // Save turning points to file
    fs.writeJSONSync("results/turningPoints.json", turningPoints, {
      spaces: 2,
      encoding: "utf-8",
    });

    // Also save convergence analysis
    fs.writeJSONSync("results/convergence_analysis.json", convergenceHistory, {
      spaces: 2,
      encoding: "utf-8",
    });

    detector.logger.info("Results saved to files.");
  } catch (err) {
    detector.logger.error("Error detecting turning points:", err);
  }
}

if (require.main === module) {
  runTurningPointDetectorExample().finally(() => {
    process.exit(0);
  });
}
