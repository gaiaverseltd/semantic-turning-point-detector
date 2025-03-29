import { Message } from "./Message";
import { TurningPointDetectorConfig } from "./semanticTurningPointDetector";

/**
 * Defines the formatting style for replaced headings.
 * - 'plain': Just the heading text (removes '#' markers only).
 * - 'bold': Surrounds the heading text with '**'.
 * - 'italic': Surrounds the heading text with '*'.
 * - 'bold-italic': Surrounds the heading text with '***'.
 * - 'prefix': Prepends a specific string (defined in `headingPrefix`) to the heading text.
 */
export type HeadingStyle = 'plain' | 'bold' | 'italic' | 'bold-italic' | 'prefix';

/**
 * Configuration options for selectivelyStripMarkdown function.
 */
export type StripMarkdownOptions = {
  /**
   * If true, removes list markers (*, -, +, 1.) while keeping the item text.
   * @default false
   */
  removeLists?: boolean;

  /**
   * Defines how heading syntax (#) should be replaced.
   * @default 'bold'
   */
  headingStyle?: HeadingStyle;

  /**
   * The prefix string to use when `headingStyle` is 'prefix'.
   * @default 'heading: '
   */
  headingPrefix?: string;
}

/**
 * Selectively removes or reformats Markdown elements like headings and optionally lists.
 * Headings (#) are replaced based on the specified `headingStyle`.
 * Lists (*, -, +, 1.) can optionally be stripped to plain text (controlled by `removeLists`).
 * Content remains on the same line, and overall newlines are preserved.
 *
 * @param markdown The input Markdown string.
 * @param options Configuration options for stripping and formatting.
 * @returns The processed string.
 */
export function selectivelyStripMarkdown(
  markdown: string,
  options?: StripMarkdownOptions
): string {
  let result = markdown;

  // --- Configuration Defaults ---
  const shouldRemoveLists = options?.removeLists ?? false;
  const headingStyle = options?.headingStyle ?? 'bold'; // Default to 'bold'
  const headingPrefix = options?.headingPrefix ?? 'heading: '; // Default prefix

  // --- Heading Replacement ---
  // Use a replacer function to dynamically format the heading text
  result = result.replace(/^#{1,6}\s+(.*)/gm, (match, headingText) => {
    // 'match' is the full matched string, e.g., "## Heading Title"
    // 'headingText' is the captured group (.*), e.g., "Heading Title"
    switch (headingStyle) {
      case 'italic':
        return `*${headingText}*`;
      case 'bold-italic':
        return `***${headingText}***`;
      case 'prefix':
        return `${headingPrefix}${headingText}`;
      case 'plain':
        return headingText;
      case 'bold': // Fallthrough for default 'bold'
      default:
        return `**${headingText}**`;
    }
  });

  // --- List Removal (Optional) ---
  if (shouldRemoveLists) {
    // Remove unordered list markers (*, -, +), preserving indentation
    result = result.replace(/^(\s*)(?:[-*+])\s+(.*)/gm, '$1$2');
    // Remove ordered list markers (1., 2.), preserving indentation
    result = result.replace(/^(\s*)(?:\d+\.)\s+(.*)/gm, '$1$2');
  }

  return result;
}


 
/**
 * A helper function that formats a given message in a form that ensures the content is not long and easily distinguishable as part of contextual information when requesting a llm or nlp model to process it.
 * @param semanticSettings 
 * @param m 
 * @param dimension 
 * @param addHeader 
 * @param sliceId 
 * @returns 
 */
export function returnFormattedMessageContent(semanticSettings: Partial<TurningPointDetectorConfig>, m: Message, dimension: number = 0, addHeader = false, sliceId = true): string {

  const messageContent = selectivelyStripMarkdown(
    m.message)
  const header = addHeader ? `##### Message - (${dimension === 0 ? 'Author\'s name' : 'Source of Turning Point (this is a turning point comprising of messages (2-or-more) that are part of a larger single conversation)'
    }): ${m.author}\nID: "${m.id.slice(
      sliceId ? 37 :  0

    )}"` : '';
  return `${header}\n` +
    `--- start of message content for id="${m.id.slice(sliceId ? 37 : 0)}" author="${m.author}" ---\n` +
    `\n  <content id="${m.id.slice(
      
      sliceId ? 37 : 0
    )}"  author="${m.author}" dimension="${dimension}">
                ${messageContent
      ?.slice(
        0,
        dimension === 0 ? Math.min(
          semanticSettings?.max_character_length != undefined
            ? semanticSettings?.max_character_length / 2
            : 20000,
          8000
        ) : messageContent.length
      )
      .split("\n")
      .map((line) => `    ${line}`)
      .join("\n")}\n[content may be truncated, original length: ${m.message.length}]\n  </content>\n--- end of message content for id="${m.id.slice(sliceId ? 37 : 0)
    }" author="${m.author}" ---\n`;



}