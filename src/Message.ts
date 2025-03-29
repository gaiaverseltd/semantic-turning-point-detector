import { returnFormattedMessageContent } from './stripContent';
import type { TurningPoint } from './semanticTurningPointDetector';

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
 * BaseMessage interface - foundation for all message types
 */
export interface Message {
  /** Unique identifier for this message */
  id: string;
  /** The sender of the message */
  author: string;
  /** The message content */
  message: string;
  /** Optional span data for dimensional tracking */
  spanData?: MessageSpan;


  // Below are MetaMessage specific methods (need to fix TODO make Message a class too so archaic check via does function exist or instance of is not needed)

  /** Get index method - implementations must handle both cases */
  getIndex?(originalMessages?: any[], isStart?: boolean): number;




  /** Get turning points */
  getTurningPoints?(): TurningPoint[];


  /** Get getMessagesContentContextualAid  */
  getMessagesContentContextualAid?(props: {
    dimension?: number,
    contextualType?: 'before-and-after' | 'within',
    messagesToUse?: number,
    maxContentLengthChar?: number
  }): string;

  getMessagesInTurningPointSpan?(): Map<string, Message[]>;
  getMessagesInTurningPointSpanToMessagesArray?(): Message[];
}


/**
 * MetaMessage class provides structured representation of higher dimensional messages
 * with guaranteed span information and proper indexing
 */
export class MetaMessage implements Message {
  public readonly id: string;
  public readonly author: string = 'meta';
  public readonly message: string;
  public readonly spanData: MessageSpan;
  private readonly representedTurningPoints?: TurningPoint[];
  // maps a turning points id to the messages that are within the span of the turning point
  private readonly messagesByTurningPoint: Map<string, Message[]> = new Map();

  readonly dimension: number;
  constructor(
    id: string,
    content: string,
    spanData: MessageSpan,
    representedTurningPoints: TurningPoint[],
    originalMessages: Message[],
    dimension: number = 0,
  ) {
    this.id = id;
    this.message = content;
    this.spanData = spanData;
    this.representedTurningPoints = representedTurningPoints;
    this.dimension = dimension;
 
    for (const tp of representedTurningPoints || []) {
      // Store the original messages that are associated with this turning point
      // based on the turning point's indices which reference the ORIGINAL message array
      const messages: Message[] = [];

      // Check if indices are valid - they might reference positions in a larger array
      // than what we currently have
      const startIndex = Math.min(tp.span.startIndex, originalMessages.length - 1);
      const endIndex = Math.min(tp.span.endIndex, originalMessages.length - 1);

      // If the indices are valid for our current array, use them
      if (startIndex >= 0 && startIndex < originalMessages.length &&
        endIndex >= 0 && endIndex < originalMessages.length) {
        messages.push(...originalMessages.slice(startIndex, endIndex + 1));
      }
      // Otherwise, try to find messages by ID
      else {
        const startMessage = originalMessages.find(m => m.id === tp.span.startId);
        const endMessage = originalMessages.find(m => m.id === tp.span.endId);

        if (startMessage) messages.push(startMessage);
        if (endMessage && startMessage !== endMessage) messages.push(endMessage);
      }

      if (messages.length === 0) {
        console.warn(`Warning: No messages found for turning point ${tp.id}. Using indirect reference.`);
        // As a last resort, just use the original messages directly
        // This at least prevents crashes, though contextual information might be limited
        this.messagesByTurningPoint.set(tp.id, originalMessages);
      } else {
        this.messagesByTurningPoint.set(tp.id, messages);
      }
    }
  }

  /**
   * Reliably returns the index of this meta-message in the original conversation
   * - Meta-messages always include spanData with reliable indices, making originalMessages unnecessary. Originally, originalMessages was used in a format that managed containing messages within a single interface. It relied on creating regex IDs to extract indices and determine if a message was a meta-message. However, this approach failed due to design flaws, as more complex origin message string IDs could not reliably be converted into a meta ID. As a result, this new class was developed to encapsulate a meta-message, which can encompass a group of turning points. This is distinct from a baseMessage, which represents a single turning point between two actual messages.

   */
  getIndex(originalMessages?: any[], isStart: boolean = true): number {
    return isStart ? this.spanData.startIndex : this.spanData.endIndex;
  }

  getMessagesInTurningPointSpan(): Map<string, Message[]> {
    return this.messagesByTurningPoint;
  }
  getMessagesInTurningPointSpanToMessagesArray(): Message[] {
    // convert the map values to a single array
    return Array.from(this.messagesByTurningPoint.values()).flat();
  }

  /**
   * Creates a string representation with embedded span information for debugging
   */
  toString(): string {
    return `MetaMessage(id=${this.id}, span=${this.spanData.startIndex}-${this.spanData.endIndex})`;
  }

  /**
   * Like `getMessagesContentContextualAid`, but for a single turning point rather than a meta message group
   * - does not include any header, only a 4thlevel header for the message(s) content
   * @param  dimension - the dimension of the message
   * @param  messagesToUse - the number of messages to use
   * @param  maxContentLengthChar - the maximum content length in characters 
   * @param beforeMessage 
   */
  static getMessagesContentContextualAidFromJustProvidedBeforeAndAfterMessages(
    beforeMessage: MetaMessage,
    afterMessage: MetaMessage,
    dimension: number = 0,
    messagesToUse: number = 2,
    maxContentLengthChar: number = 8000,
    originalMessages: Message[] = [],
    type: 'before-and-after' | 'within' = 'within',
  ) {

    if (originalMessages.some(m => m instanceof MetaMessage)) {
      throw new Error(`Error: Original messages should not contain any meta-messages. Found: ${originalMessages.filter(m => m instanceof MetaMessage).map(m => m.id).join(', ')}`);
    }

    // create a span id for this before and after message (simialr to how the meta message is created)
    const prefix = beforeMessage instanceof MetaMessage && afterMessage instanceof MetaMessage ? 'meta-' : 'base-';
    const spanId = `${prefix}${beforeMessage.id}-${afterMessage.id}`;


    if (originalMessages.length === 0) {
      throw new Error(`No messages found for turning point ${spanId} in span ${beforeMessage.spanData.startIndex}-${afterMessage.spanData.endIndex}, originalMessages length: ${originalMessages.length}\n- tp.span data: ${JSON.stringify(beforeMessage.spanData)}, tp. after span data: ${JSON.stringify(afterMessage.spanData)}`);
    }

    // const messageContent = MetaMessage.getMessagesContentContextualAid(
    //   metaMessage.getTurningPoints(),
    //   messages,
    //   metaMessage,
    //   dimension,
    //   type,
    //   0,
    //   maxContentLengthChar
    // );
    // use new param props 
    if (beforeMessage instanceof MetaMessage && afterMessage instanceof MetaMessage) {


      const beforeMessageContextual = beforeMessage.getMessagesContentContextualAid({
        dimension,
        contextualType: type,
        messagesToUse: messagesToUse ?? 1,
        maxContentLengthChar
      });

      const afterMessageContextual = afterMessage.getMessagesContentContextualAid({
        dimension,
        contextualType: type,
        messagesToUse: messagesToUse ?? 1,
        maxContentLengthChar
      });

      return `## Below is contextual content of the actual converation message content concerning the meta message group of turning points that are being analyzed, and the messages that are before and after the turning point to analyze\n\n` +
        `### These are messages of the conversation  are ${type === 'within' ? 'are at the start of the messages within' : 'before'} the group of turning points that are being analyzed\n` +
        `${beforeMessageContextual.split('\n').map(line => `   ${line}`).join('\n')}` +
        `\n### These are messages that end at this group of turning points that are ${type === 'within' ? 'within' : 'after'
        } the group of turning points that are being analyzed\n` +
        `${afterMessageContextual.split('\n').map(line => `   ${line}`).join('\n')}` +
        `\n---- end of messages content before and after the turning point to analyze ----\n\n\n`;
    } else {
      throw new Error(`Before and after messages must be instances of MetaMessage`);
    }





  }


  /**
   * Finds the index of a specific message content element (baseMessage), from a given provided id string that is either a MetaMessage or a BaseMessage
   * - determines the instance of the message (meta or base) and returns the index of the message in the original messages array
   * @param param0 
   * @returns 
   */
  static findIndexOfMessageFromId = ({
    id,
    beforeMessage,
    afterMessage,
    messages,
    consoleLogger = console,
  }: {
    /** The id of the meta/ormessge to find index */
    id: string;
    /** The message before the turning point (may be a meta) */
    beforeMessage?: Message | null | undefined | MetaMessage;
    /** The message after the turning point (may be a meta) */
    afterMessage?: Message | null | undefined | MetaMessage;
    /** The original array of messages for lookup these are the original messages (not meta) */
    messages: Message[] | MetaMessage[];

    consoleLogger?: Console;
  }): number => {

    // Check if message has getIndex method (MetaMessage instances)
    // - if so, check if the beforeMessage (MetaMessage) has the same id as the one we are looking for, if so use that for faster lookup
    if (
      beforeMessage &&
      typeof beforeMessage?.getIndex === "function" &&
      beforeMessage.id === id
    ) {
      return beforeMessage.getIndex(messages);
    }
    // Check if message has getIndex method (MetaMessage instances)
    // - if so, check if the afterMessage (MetaMessage) has the same id as the one we are looking for, if so use that for faster lookup
    if (
      afterMessage &&
      typeof afterMessage.getIndex === "function" && afterMessage.id === id) {
      return afterMessage.getIndex(messages);
    }

    // IMPORTANT FIX: Check if ID is a numeric string (an index from meta-message parsing)
    // This handles the case where we extract "4" from "SpanIndices: 4-10"
    if (/^\d+$/.test(id)) {
      // It's a numeric index from meta-message content, use it directly
      const numericIndex = parseInt(id, 10);
      if (numericIndex >= 0 && numericIndex < messages.length) {
        return numericIndex;
      }
      // If it's outside valid range, log but continue to other checks
      consoleLogger.info(
        `Warning: Numeric ID ${id} is outside valid range for original messages originalMessages possible id list: ${messages
          .map((msg) => msg.id)
          .join(", ")}`
      );
      throw new Error(
        `Numeric ID ${id} is outside valid range for original messages`
      );
    }

    // Special handling for meta-message IDs
    if (id.startsWith("meta-")) {
      const messagesArray = beforeMessage && afterMessage ? [beforeMessage, afterMessage] : messages;
      // For meta-messages, use their spanData directly if available
      const metaMessage = messagesArray.find(
        (msg) => msg.id === id
      );
      if (metaMessage?.spanData) {
        if (messages[metaMessage.spanData.startIndex] === undefined) {
          throw new Error(`Meta-message ${id} has spanData with startId ${metaMessage.spanData.startId} that is not found in original messages.`);
        }
        return metaMessage.spanData.startIndex;
      }

      // Still need the fallback parsing for legacy meta-messages
      const msgWithSpan = messagesArray.find(
        (m) => m.id === id
      );
      if (msgWithSpan && msgWithSpan.author === "meta") {
        // const spanMatch = msgWithSpan.message.match(/SpanIndices: (\d+)-(\d+)/);
        // if (spanMatch && spanMatch.length >= 2) {
        //   return parseInt(spanMatch[1], 10);
        // }
        throw new Error(
          `Incorrect meta-message format for ID ${id}. Expected spanData to be available but found none. Message: ${msgWithSpan.message}, some code is still using old messages, check to ensure new classes are being used.`
        );
      }

      consoleLogger.error(
        `Error: Meta-message ${id} missing required spanData metameasge:${JSON.stringify(
          metaMessage,
          null,
          2
        )}`
      );
      throw new Error(
        `Meta-message ${id} missing required spanData. All meta-messages should have spanData.`
      );
    }

    // Regular lookup for non-meta messages
    const index = messages.findIndex((msg) => msg.id === id);
    if (index === -1) {
      console.log(`Error: Message ID ${id} not found in original messages`);
      throw new Error(
        `Message with ID ${id} not found in original messages array.`
      );
    }

    return index;
  };
/**
 * Retrieves and formats message content from turning points to provide contextual analysis.
 * 
 * This method extracts messages from the first and last turning points in the group,
 * formats them according to the specified parameters, and returns a structured
 * representation that can be used for analysis or display.
 * 
 * @param options Configuration options for content retrieval and formatting
 * @param options.dimension - Dimensional level of analysis (0 = base conversation, 1+ = meta-analysis of turning point groups)
 * @param options.contextualType - How to present message context:
 *   - "within": Shows messages within the turning point group (first and last messages)
 *   - "before-and-after": Shows messages that appear before and after the turning point group
 * @param options.messagesToUse - Number of messages to include in each context section (default: 2)
 * @param options.maxContentLengthChar - Maximum length in characters for each message content (default: 8000)
 * 
 * @returns Formatted string containing structured message content with appropriate headers and context
 * 
 * @example
 * // Get messages within a turning point group
 * const withinContent = metaMessage.getMessagesContentContextualAid({
 *   dimension: 1,
 *   contextualType: "within"
 * });
 * 
 * @example
 * // Get messages before and after a turning point group with custom limits
 * const surroundingContent = metaMessage.getMessagesContentContextualAid({
 *   contextualType: "before-and-after",
 *   messagesToUse: 3,
 *   maxContentLengthChar: 5000
 * });
 */
public getMessagesContentContextualAid({
  dimension = 0,
  contextualType = "within",
  messagesToUse = 2,
  maxContentLengthChar = 8000
}: {
  /** 
   * Dimensional level of analysis:
   * - 0: Base conversation analysis (individual messages)
   * - 1+: Meta-analysis of turning point groups (higher abstraction)
   */
  dimension?: number,
  
  /**
   * Context presentation strategy:
   * - "within": Shows messages within the turning point span (first and last)
   * - "before-and-after": Shows messages surrounding the turning point
   */
  contextualType?: "before-and-after" | "within",
  
  /**
   * Number of messages to include in each context section
   * (beginning/end of turning point or before/after turning point)
   */
  messagesToUse?: number,
  
  /**
   * Maximum character length for individual message content before truncation
   */
  maxContentLengthChar?: number
}): string {
    // Get turning points and original messages
    const turningPoints = this.getTurningPoints();
    const originalMessages = this.getMessagesInTurningPointSpanToMessagesArray();

    console.info(
      `getMessagesContentContextualAid: ${this.id} - ${turningPoints.length} turning points, ` +
      `original messages length: ${originalMessages.length}, ` +
      `org ids: ${turningPoints.map(tp => tp.id).join(', ')}`
    );

    // Find turning points with extreme indices (first/last)
    const getTurningPointWithExtremeIndex = (turningPoints: TurningPoint[], isStart = true) => {
      let extremeIndex = isStart ? turningPoints[0].span.startIndex : turningPoints[0].span.endIndex;
      let extremeTurningPoint = turningPoints[0];

      for (let i = 1; i < turningPoints.length; i++) {
        const currentTurningPoint = turningPoints[i];
        const currentIndex = isStart ? currentTurningPoint.span.startIndex : currentTurningPoint.span.endIndex;
        const isMoreExtreme = isStart ? currentIndex < extremeIndex : currentIndex > extremeIndex;

        if (isMoreExtreme) {
          extremeIndex = currentIndex;
          extremeTurningPoint = currentTurningPoint;
        }
      }

      return extremeTurningPoint;
    }

    // Get first and last turning points
    const firstTurningPoint = getTurningPointWithExtremeIndex(turningPoints, true);
    const lastTurningPoint = getTurningPointWithExtremeIndex(turningPoints, false);

    // Get associated messages
    const startMessagesContext = this.messagesByTurningPoint.get(firstTurningPoint.id) || [];
    const endMessagesContext = this.messagesByTurningPoint.get(lastTurningPoint.id) || [];

    // Validate we have messages
    if (startMessagesContext.length === 0 && endMessagesContext.length === 0) {
      throw new Error(
        `No messages found for turning point IDs ${firstTurningPoint.id}-${lastTurningPoint.id} ` +
        `in span ${firstTurningPoint.span.startIndex}-${lastTurningPoint.span.endIndex}. ` +
        `Original messages length: ${originalMessages?.length}.\n` +
        `- First turning point span data: ${JSON.stringify(firstTurningPoint.span)}\n` +
        `- Last turning point span data: ${JSON.stringify(lastTurningPoint.span)}`
      );
    }

    // Format the message content from start and end turning points
    const startMessages = startMessagesContext
      .slice(0, messagesToUse)
      .map(m => returnFormattedMessageContent({
        max_character_length: maxContentLengthChar,
      }, m, dimension))
      .join('\n');

    const endMessages = endMessagesContext
      .slice(-1 * messagesToUse)
      .map(m => returnFormattedMessageContent({
        max_character_length: maxContentLengthChar,
      }, m, dimension))
      .join('\n');

    // Format for "within" context type
    if (contextualType === 'within') {
      return this.formatWithinContextOutput(startMessages, endMessages);
    }
    // Format for "before-and-after" context type
    else {
      return this.formatBeforeAfterContextOutput({
        firstTurningPoint,
        lastTurningPoint,
        originalMessages,
        dimension,
        messagesToUse,
        maxContentLengthChar
      });
    }
  }

  /**
   * Formats the "within" context output
   */
  private formatWithinContextOutput(startMessages: string, endMessages: string): string {
    return [
      `## Messages Within This Turning Point Group (ID: "${this.id}")`,
      `------ Begin of messages within grouping of turning points id="${this.id}" ------`,

      `### First Messages in This Turning Point Group`,
      startMessages.split('\n').map(line => `   ${line}`).join('\n'),

      `### Last Messages in This Turning Point Group`,
      endMessages.split('\n').map(line => `   ${line}`).join('\n'),

      `------ End of messages within grouping of turning points id="${this.id}" ------\n\n`,
    ].join('\n');
  }

  /**
   * Formats the "before-and-after" context output
   */
  private formatBeforeAfterContextOutput({
    firstTurningPoint,
    lastTurningPoint,
    originalMessages,
    dimension,
    messagesToUse,
    maxContentLengthChar
  }: {
    firstTurningPoint: TurningPoint,
    lastTurningPoint: TurningPoint,
    originalMessages: Message[],
    dimension: number,
    messagesToUse: number,
    maxContentLengthChar: number
  }): string {
    // Get messages for these turning points
    const beforeTPMessages = this.messagesByTurningPoint.get(lastTurningPoint.id) || [];
    const afterTPMessages = this.messagesByTurningPoint.get(firstTurningPoint.id) || [];

    // Find messages that come before the first message in the turning points
    const beforeMessages = beforeTPMessages.length > 0
      ? originalMessages
        .filter(m => originalMessages.indexOf(m) < originalMessages.indexOf(beforeTPMessages[0]))
        .slice(-messagesToUse)
      : [];

    // Find messages that come after the last message in the turning points
    const afterMessages = afterTPMessages.length > 0
      ? originalMessages
        .filter(m => originalMessages.indexOf(m) > originalMessages.indexOf(afterTPMessages[afterTPMessages.length - 1]))
        .slice(0, messagesToUse)
      : [];

    // Format the content
    const dimensionDescription = dimension === 0
      ? 'paired messages forming a potential turning point'
      : 'group of related turning points';

    const beforeMessagesContent = beforeMessages.length > 0
      ? beforeMessages
        .map(m => returnFormattedMessageContent({
          max_character_length: maxContentLengthChar,
        }, m, dimension))
        .join('\n')
      : `No messages exist before this ${dimensionDescription}.`;

    const afterMessagesContent = afterMessages.length > 0
      ? afterMessages
        .map(m => returnFormattedMessageContent({
          max_character_length: maxContentLengthChar,
        }, m, dimension))
        .join('\n')
      : `No messages exist after this ${dimensionDescription}.`;

    return [
      `## Context Surrounding This Turning Point`,
      `- These messages provide context for analyzing the turning point but are NOT part of the turning point itself.`,
      `- The turning point consists of ${dimensionDescription} that represent a significant shift in the conversation.`,
      `- This contextual information helps with analysis but should not be the primary basis for classification.`,

      `### Messages Before This Turning Point Group`,
      beforeMessagesContent.split('\n').map(line => `   ${line}`).join('\n'),

      `### Messages After This Turning Point Group`,
      afterMessagesContent.split('\n').map(line => `   ${line}`).join('\n'),

      `---- End of contextual messages surrounding this turning point ----\n\n`
    ].join('\n');
  }
  /**
   * Factory method to create a category meta-message from turning points
   */
  static createCategoryMetaMessage(
    category: string,
    points: TurningPoint[],
    index: number,
    originalMessages: Message[],
    dimension: number = 0
  ): MetaMessage {
    if (originalMessages.some(m => m instanceof MetaMessage)) {
      throw new Error(`Error: Original messages should not contain any meta-messages. Found: ${originalMessages.filter(m => m instanceof MetaMessage).map(m => m.id).join(', ')}`);
    }
    // Find the overall span of all turning points in this category
    const minStartIndex = Math.min(...points.map(p => p.span.startIndex));
    const maxEndIndex = Math.max(...points.map(p => p.span.endIndex));

    // Find corresponding message IDs
    const startMsgId = points.find(p => p.span.startIndex === minStartIndex)?.span.startId || '';
    const endMsgId = points.find(p => p.span.endIndex === maxEndIndex)?.span.endId || '';

    // Generate content
    const quotes = points.flatMap(tp => tp.quotes || []).filter(Boolean).sort((a, b) => a.length - b.length).filter(q => q.length > 5 && q.length < 1000).slice(0, 3);
    const keywords = points.flatMap(tp => tp.keywords || []).filter(Boolean);

    const categoryContent = `
### ${category} Turning Points (within this Meta Grouping)
- The point here is to form a higher level Turning Point based on this list of turning points.
Significance: ${Math.max(...points.map(p => p.significance)).toFixed(2)}
Complexity: ${Math.max(...points.map(p => p.complexityScore)).toFixed(2)}
Keywords: ${Array.from(new Set(keywords)).slice(0, 10).join(', ')}
Quotes: ${quotes.map(q => `"${q.replace(
      /\n/g,
      ' '
    )}"`).join(', ')}
SpanIndices: ${minStartIndex}-${maxEndIndex}
SpanMessageIds: ${startMsgId}-${endMsgId}
Emotional Tones: ${Array.from(new Set(points.flatMap(tp => tp.emotionalTone || []))).slice(0, 5).join(', ')}
Sentimentality: ${Math.max(...points.map(p => (p.sentiment?.toLocaleLowerCase()?.includes('positive') ? 1 : -1) || 0)) >= 1 ? 'positive' : 'negative'}

`;

    // Add contextual information
    let builtContext = ``;
    const startMessagesContext = originalMessages.slice(Math.max(0, minStartIndex - 3), minStartIndex).filter(Boolean);
    const endMessagesContext = originalMessages.slice(maxEndIndex, maxEndIndex + 3).filter(Boolean);

    if (startMessagesContext.length > 0 || endMessagesContext.length > 0) {
      builtContext = `\n\n## Contextual Aid\n- The following text provides broader context to showcase a truncated view of the messages within this span in the turning point.`;

      if (startMessagesContext.length > 0) {
        builtContext += `\n### Messages of the start of turning points of this grouping of turning point(s) that are within span as Context of the message content within this group of turning points\n` +
          startMessagesContext.map(m => `Author: ${m.author}\nID: "${m.spanData?.startId ?? m.id}"\nContent:\n\n${returnFormattedMessageContent({
            max_character_length: 5000,
          }, m, 0)
            })
            }`).join('\n\n');
      }
      if (endMessagesContext.length > 0) {
        builtContext += `\n### The messages in between the turning points have been omitted for brevity\n`;
      }
      if (endMessagesContext.length > 0) {
        builtContext += `\n### Messages near the end of the span of this grouping of turning point(s) span as Context of the message content within this group of turning points\n` +
          endMessagesContext.map(m => `Author: ${m.author}\nID: "${m.spanData?.startId ?? m.id}"\nContent:\n\n${returnFormattedMessageContent({
            max_character_length: 5000,
          }, m, 0)
            }`).join('\n\n');
      }
    }

    // Create span data with guaranteed indices
    const span: MessageSpan = {
      startId: startMsgId,
      endId: endMsgId,
      startIndex: minStartIndex,
      endIndex: maxEndIndex
    };

    return new MetaMessage(`meta-cat-${index}`, categoryContent + builtContext, span, points, originalMessages, dimension);
  }

  /**
   * Factory method to create a section meta-message
   */
  static createSectionMetaMessage(
    sectionPoints: TurningPoint[],
    sectionIndex: number,
    originalMessages: Message[]
  ): MetaMessage {
    // Find the overall span of all turning points in this section
    const minStartIndex = Math.min(...sectionPoints.map(p => p.span.startIndex));
    const maxEndIndex = Math.max(...sectionPoints.map(p => p.span.endIndex));

    // Find corresponding message IDs
    const startMsgId = sectionPoints.find(p => p.span.startIndex === minStartIndex)?.span.startId || '';
    const endMsgId = sectionPoints.find(p => p.span.endIndex === maxEndIndex)?.span.endId || '';

    // Create section meta-message content
    const sectionContent = `
# Conversation Section ${sectionIndex + 1}
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

    // Create with guaranteed span data
    const span: MessageSpan = {
      startId: startMsgId,
      endId: endMsgId,
      startIndex: minStartIndex,
      endIndex: maxEndIndex
    };

    return new MetaMessage(`meta-section-${sectionIndex}`, sectionContent, span, sectionPoints, originalMessages);
  }

  getTurningPoints() {
    return this.representedTurningPoints || [];
  }
}

/**
 * Type guard to check if a message is a MetaMessage instance
 */
export function isMetaMessage(message: any): message is MetaMessage {
  return message instanceof MetaMessage ||
    (message && message.author === 'meta' && typeof message.getIndex === 'function');
}