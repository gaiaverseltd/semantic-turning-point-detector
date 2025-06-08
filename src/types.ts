// -----------------------------------------------------------------------------
// Embedding Generation
// -----------------------------------------------------------------------------

import type winston from "winston";
import type { Message } from "./Message";

// -----------------------------------------------------------------------------
// Core Interfaces
// -----------------------------------------------------------------------------

/**
 * Message span identifies a range of messages
 * Used for tracking dimensional representations across recursion levels
 */
export interface MessageSpan {
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
  category: string;
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
 * Type defintion for a single category utilized as part of instructions for the LLM in analyzing turning points.
 */
export type TurningPointCategory = {
  /**
   * A single word, recommended to be one-word, that describes the category of the turning point, and is distinct from any other category.
   */
  category: string;
  /**
   * A short one sentence description of the category of what this represents, usually a definition suffices.
   */
  description: string;
};
/**
 * Configuration options for the turning point detector.
 * - Detailed descriptions for each option are provided below.
 */
export interface TurningPointDetectorConfig {
  /**
   * Configurable turning point categories with descriptions.
   * Each category should have a name and description to help the LLM classify turning points.
   *
   * @remarks
   * - Maximum of 15 categories allowed (default categories count: 11)
   * - If more than 15 categories are provided, excess categories will be ignored with a warning
   * - Categories should be descriptive and distinct to ensure accurate classification
   * - Default categories include: Topic, Insight, Emotion, Meta-Reflection, Decision, Question, Problem, Action, Clarification, Objection, Other
   *
   * @example
   * ```typescript
   * turningPointCategories: [
   *   { category: "Topic", description: "A shift to a new subject or theme" },
   *   { category: "Insight", description: "A realization, discovery, or moment of understanding" },
   *   { category: "Custom", description: "A custom category for specific use cases" }
   * ]
   * ```
   */
  turningPointCategories: TurningPointCategory[];

  /**
   * API Key for LLM requests.
   *
   * If you prefer not to pass the API key as a variable, you can set the environment variable
   * `LLM_API_KEY` for a different external endpoint. Alternatively, set the `OPENAI_API_KEY`
   * environment variable to use the default OpenAI API endpoint.
   *
   * By default, if a new OpenAI client is created without the `apiKey` set, it will look for
   * the `OPENAI_API_KEY` environment variable.
   */
  apiKey: string;

  /**
   * The llm model used in analyzing turning points. Must be a model that exists on the configured endpoint, and thus if no custom endpoint is configured, must be a model that exists on the OpenAI API available on your apiKey.
   */

  classificationModel: string;

  /**
   * The temperature setting for the LLM model.
   * - Defaults to 0.6, not recommended to set higher than 1, as it may lead to incorrect responses that are not in a proper response format, since semantic turnign point relies on the LLM to return a JSON Schema object.
   */
  temperature?: number;

  /**
   * The top probability setting for the LLM model.
   * - Lower means probabilities must be higher, and higher means more diverse responses, max is 1.0. Not recommended to go lower than 0.8 as it may lead to incorrect responses that have wrong syntax in JSON Schema format, given that in the case of those responses, lowering the diversity, may inadvertnly limit the LLM to provide the write json syntax, which is of course completely unrelated to the actual response, etc.
   */
  top_p?: number;

  /** Model for generating embeddings, e.g 'text-embedding-3-small', or a custom embedding model if embeddingEndpoint is set and is an OpenAI-compatible endpoint. */
  embeddingModel: string;

  /**
   * You can also choose to utilize a custom endpoint, that also follows the OpenAI API format for embeddings, to utilize other embedding providers and models. This includes such as LM Studio, Ollama, and most other providers that support OpenAI format via chat requests. If left undefined, it will default to the OpenAI API endpoint, in which you can set an embedding [model](https://platform.openai.com/docs/guides/embeddings/embedding-models) offered by OpenAI.
   * - If you use an external commercial provider for embeddings, you can configure the api key via setting the environment variable `EMBEDDINGS_API_KEY`.
   */
  embeddingEndpoint?: string;

  /**
   * RAM limit for embedding cache in MB (default: 256MB).
   * - Note: On Node.js, the default memory limit for a process is 512MB on 32-bit systems and approximately 1GB on 64-bit systems.
   *   This limit can be increased if needed. For TypeScript projects, compile your script to JavaScript and run it with:
   *   `node --max-old-space-size=4096 yourScript.js` to set the RAM limit to 4GB.
   */
  embeddingCacheRamLimitMB?: number;

  /**
   * Threshold that determines when semantic changes between messages constitute a turning point.
   *
   * @remarks
   * Range: 0.0-1.0 (typically 0.2-0.8)
   * - Higher values (>0.5) detect only major semantic shifts, resulting in fewer but more significant turning points
   * - Lower values (<0.3) capture subtle changes in conversation flow, but may produce numerous minor turning points
   * - At dimension > 0, this threshold is automatically scaled down to accommodate meta-message analysis
   * - Adjust based on conversation density: use higher values for technical/focused discussions,
   *   lower values for casual/meandering conversations
   */
  semanticShiftThreshold: number;

  /**
   * Minimum token count when dividing conversations into processable chunks.
   *
   * @remarks
   * - Prevents creation of chunks that are too small for meaningful semantic analysis
   * - Lower values allow finer-grained chunking but may miss broader context
   * - Typically set between 200-500 tokens for balanced processing
   * - For highly technical content, consider higher minimum (400+) to maintain context
   * - For conversations with short messages, lower values (150-250) may be appropriate
   * - This setting works in conjunction with minMessagesPerChunk
   */
  minTokensPerChunk: number;

  /**
   * Maximum token count allowed for conversation chunks before splitting.
   *
   * @remarks
   * - Limits chunk size to prevent context window overflows when processing with LLMs
   * - Higher values provide more context but consume more computational resources
   * - Recommended to set below the classification model's context window, accounting for prompt overhead
   *   (e.g., 4000-8000 for most models, 12000-16000 for larger models)
   * - If set too low relative to minTokensPerChunk, many chunks will be exactly at the minimum size
   * - At deeper dimensions, this value is automatically scaled down proportionally
   */
  maxTokensPerChunk: number;

  /**
   * Maximum dimension level for recursive analysis in the ARC framework.
   *
   * @remarks
   * - Controls how many levels of meta-analysis are performed on the conversation
   * - Dimension 0: Direct message analysis
   * - Dimension 1: Analysis of turning point patterns
   * - Dimension 2+: Higher-order pattern recognition
   * - Higher values (3-5) enable detection of complex narrative arcs and subtle theme progressions,
   *   but significantly increase processing time
   * - For most conversations, 2-3 levels are sufficient
   * - Very long conversations may benefit from higher values (4-5)
   * - Actual escalation to higher dimensions only occurs when complexity saturation is reached
   */
  maxRecursionDepth: number;

  /**
   * Minimum significance score required for a turning point to be included in final results.
   *

   * Range: 0.0-1.0
   * - Acts as a quality filter to exclude minor or low-confidence turning points
   * - Only applied when onlySignificantTurningPoints is true
   * - Higher thresholds (>0.7) produce fewer, higher-quality turning points
   * - Lower thresholds (<0.4) include more subtle conversation shifts
   * - Typical setting: 0.5-0.7 for balanced results
   * - Significance scores are determined by the classification model based on semantic
   *   importance, not just embedding distance
   * - Consider using lower values for technical/educational content where subtle shifts matter
* Please note that the "scoring," such as confidence levels, will vary in scale depending on the embedding model used. It is important to understand these differences. When measuring across dialogues, ensure that you record the embedding model and the significance threshold used, as both factors will impact the results.
  * 
  * @remarks
  * **Why Different Models Need Different Thresholds:**
  * 
  * Each embedding model has distinct similarity score distributions that require calibrated thresholds:
  * 
  * **OpenAI Models (text-embedding-3-large/small):**
  * - Similarity range: 0.3-0.9 (compressed distribution)
  * - Threshold: `0.7` - Works well due to higher baseline similarities
  * - Characteristics: More conservative, tends to cluster similar concepts tightly
  * 
  * **Snowflake Arctic Embed v2 (Recommended: 0.5-0.6):**
  * - Similarity range: 0.1-0.7 (expanded distribution) 
  * - Threshold: `0.5` - Optimal for capturing genuine semantic shifts
  * - **Why Arctic May Be Superior:**
  *   - **Better Discrimination**: Spreads similarity scores across full range (0.1-0.7 vs 0.3-0.9)
  *   - **Retrieval-Optimized**: Specifically trained for semantic search and distinction tasks
  *   - **Semantic Sensitivity**: More responsive to subtle meaning changes in conversations
  *   - **Real-World Performance**: Often outperforms larger models in semantic turning point detection
  * 
  * **Quality Indicators (Arctic vs OpenAI):**
  * ```
  * Arctic Results:  0.70, 0.82, 0.89, 0.94, 0.99  ← Wide discrimination range
  * OpenAI Results:  0.75, 0.75, 0.85, 0.85, 0.85  ← Compressed, less nuanced
  * ```
  * 
  * **Arctic's Advantages for Conversation Analysis:**
  * - **Granular Significance**: Provides 0.70→0.99 range vs OpenAI's 0.75→0.85 clustering
  * - **Better Context Understanding**: Trained on diverse text for retrieval tasks
  * - **Semantic Shift Detection**: More sensitive to conversational flow changes
  * - **Efficiency**: Smaller model with specialized training often outperforms general-purpose large models
  * 
  * **Impact of Miscalibrated Thresholds:**
  * - **Too High (0.7+ for Arctic)**: Filters out genuinely significant turning points
  * - **Too Low (0.3- for any model)**: Includes noise and minor variations
  * - **Model Mismatch**: Causes artificially low confidence scores despite quality detection
  * 
  * **Auto-Calibration Pattern:**
  * ```typescript
  * significanceThreshold: this.config.embeddingModel.includes("arctic") 
  *   ? 0.5  // Arctic: Lower threshold captures quality discrimination
  *   : 0.75  // OpenAI: Higher threshold compensates for compressed range
   * 
   */
  significanceThreshold: number;
  /**
   * Controls turning point filtering strategy and result prioritization.
   *
   * @remarks
   * This parameter determines how turning points are filtered and returned:
   *
   * When `true` (focused analysis):
   * - Enforces filtering based on `significanceThreshold`
   * - Strictly limits results to `maxTurningPoints`
   * - Prioritizes results by significance score over chronological order
   * - Ideal for comparative analysis across different conversations or configurations
   *
   * When `false` (comprehensive analysis):
   * - Returns all detected turning points regardless of significance
   * - Ignores the `maxTurningPoints` limit
   * - Orders results chronologically by position in conversation
   * - Preferred for detailed conversation analysis and identifying all semantic shifts
   *
   * @example
   * // For detailed exploration of a single conversation:
   * detector.onlySignificantTurningPoints = false;
   *
   * // For comparing key shifts across multiple conversations:
   * detector.onlySignificantTurningPoints = true;
   */
  onlySignificantTurningPoints: boolean;

  /** Minimum messages per chunk */
  minMessagesPerChunk: number;
  /** Maximum turning points in final results */
  maxTurningPoints: number;
  /** Enable verbose logging */
  debug: boolean;
  /**
   * Setting a custom endpoint overrides the default `api.openai.com/v1` endpoint, allowing for the use of other LLM providers that adhere to the same API structure. The Semantic Turning Point utilizes advanced parameters, specifically `format`, which instructs the response to be returned as a JSON Schema object. However, not all OpenAI-compatible methods will support this feature. The examples below all support formatted responses, and it's important to note that Semantic Turning Point does not utilize tool calls.
   *
   * Some examples include:
   * - [Ollama](https://github.com/ollama/ollama/tree/main/docs)
   * - [OpenRouter](https://openrouter.ai/models)
   * - [vLLM](https://github.com/vllm-project/vllm)
   * - [LM Studio](https://lmstudio.ai/)
   * - [Text Generation API](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API)
   *
   */
  endpoint?: string;

  /** Complexity saturation threshold (dimension escalation trigger) */
  complexitySaturationThreshold: number;
  /** Enable convergence measurement across iterations */
  measureConvergence: boolean;

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

  logger?: winston.Logger | Console;

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

  /**
* Configures the number of parallel requests for LLM analysis. The ARC process involves recursively breaking down the conversation into separate, independent segments for analysis. This allows for parallel processing, which can increase both costs and resource usage. The default value is 1 when using a custom endpoint, or 4 if none is set (thus defaulting to OpenAI). If you are using an external service, such as a commercial API like OpenAI or OpenRouter.ai, you may increase this number as desired. However, setting it higher than 20 may risk reaching rate limits, depending on the constraints imposed by the service.


   */
  concurrency?: number;

  /**
   * Concurrency to use when doing embeddings, is default to 5 as embeddings do not require much resources and can be done in parallel.
   */
  embeddingConcurrency?: number;
}

/**
 * Default turning point categories with descriptions
 */
export const turningPointCategories: TurningPointCategory[] = [
  {
    category: "Topic",
    description:
      "This category is for content that is primarily focused on a specific area, domain, or subject. Use this when the content warrants categorization by topic.",
  },
  {
    category: "Insight",
    description:
      "This category applies to content that provides a unique insight or perspective. Use this when the content warrants categorization by insight.",
  },
  {
    category: "Emotion",
    description:
      "This category is for content that holds significant emotional impact. Use this when the content warrants categorization by emotion.",
  },
  {
    category: "Meta-Reflection",
    description:
      "This category applies to content that reflects on the conversation or interaction. Use this when the content warrants categorization by meta-reflection.",
  },
  {
    category: "Decision",
    description:
      "This category is for content that involves a decision or choice that has been made. Use this when the content warrants categorization by decision.",
  },
  {
    category: "Question",
    description:
      "This category applies to content that poses a question or inquiry. Use this when the content warrants categorization by question.",
  },
  {
    category: "Problem",
    description:
      "This category is for content that presents a problem or issue. Use this when the content warrants categorization by problem.",
  },
  {
    category: "Action",
    description:
      "This category applies to content that involves an action or activity, or serves as a call to action. Use this when the content warrants categorization by action.",
  },
  {
    category: "Clarification",
    description:
      "This category is for content that seeks or provides clarification. Use this when the content warrants categorization by clarification.",
  },
  {
    category: "Objection",
    description:
      "This category applies to content that expresses an objection or disagreement. Use this when the content warrants categorization by objection.",
  },
  {
    category: "Other",
    description:
      "This category applies to any other significant conversational shift that doesn't fit the above categories.",
  },
];

/**
 * Chunking result with message segments and metrics
 */
export interface ChunkingResult {
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
export interface MessageEmbedding {
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
export interface ConvergenceState {
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
