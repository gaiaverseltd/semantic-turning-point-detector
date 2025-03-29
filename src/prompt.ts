import { ResponseFormatJSONSchema } from "openai/resources/shared";
import { Message, MetaMessage } from "./Message";
import { returnFormattedMessageContent } from "./stripContent";
import { TurningPointDetectorConfig } from "./semanticTurningPointDetector";

export const emotionalTones: string[] = [
  "delight",
  "love",
  "gratitude",
  "pride",
  "anger",
  "fear",
  "sadness",
  "disgust",
  "surprise",
  "curiosity",
];

export const categories = {
  topic:
    "This category is for content that is primarily focused on a specific area, domain, or subject. Use this when the content warrants categorization by topic.",
  insight:
    "This category applies to content that provides a unique insight or perspective. Use this when the content warrants categorization by insight.",
  emotion:
    "This category is for content that holds significant emotional impact. Use this when the content warrants categorization by emotion.",
  "meta-reflection":
    "This category applies to content that reflects on the conversation or interaction. Use this when the content warrants categorization by meta-reflection.",
  decision:
    "This category is for content that involves a decision or choice that has been made. Use this when the content warrants categorization by decision.",
  question:
    "This category applies to content that poses a question or inquiry. Use this when the content warrants categorization by question.",
  problem:
    "This category is for content that presents a problem or issue. Use this when the content warrants categorization by problem.",
  action:
    "This category applies to content that involves an action or activity, or serves as a call to action. Use this when the content warrants categorization by action.",
  clarification:
    "This category is for content that seeks or provides clarification. Use this when the content warrants categorization by clarification.",
  objection:
    "This category applies to content that expresses an objection or disagreement. Use this when the content warrants categorization by objection.",
};
export function formSystemMessage({
  distance,
  dimension,
}: {
  /** The semantic distance (via embeddings) between two messages */
  distance: number;
  /** The dimensionlity of the currentl potential turning point, where if dimension > 0, it means that it contains a group of turning points, recursively. */
  dimension: number;
}): string {
  const systemPrompt = `
You are an expert conversation analyst agent tasked with identifying and classifying semantic turning points.
- Within a conversation chain of messages ${dimension > 0
      ? `( provided as a grouping of turning points (partially a full conversation), to analyze from the overall turning point (akin to formulate a single turning point from a list of turning points), or a full conversation)`
      : ` (provided as two messages to analyze to assess if it is a turning point or not, as well as contextual messages before and after these two messages to help you better assess if these specific two mesages are in fact a turning point or not)`
    } 
- As a background, on what we define as a turning point: A potential turning point has been detected between the two messages provided below based on semantic distance analysis, your task is to ensure you understand this to help you ${dimension === 0
      ? "formulate a single meta turning point from the group of turning points that partially encapuslate an entire conversation, with provided contextual messages as aid"
      : "formulate a turning point from the provided messages"
    }.

**Your Goal:** Accurately classify the *type* of shift occurring between ${dimension === 0
      ? "these two specific turning points provided in the user message as text content"
      : "these messages provided in the user message as text content"
    }.
}, considering the surrounding context provided in the 'Contextual Aid' section (if available), provided wherein this system prompt is a guide to help you understand the task at hand.

# Contextual Aid and Background Info

Use these content as context to help aid you in your analysis of the content provided in the user message content, and to help you understand the task at hand.

**Semantic Distance:** The calculated cosine distance is ${distance.toFixed(
      2,
    )}. 
- While this distance indicates a shift, focus primarily on the *content and interaction* to determine the significance.`;

  return systemPrompt;
}

export function formUserMessage({
  beforeMessage,
  afterMessage,
  dimension,
  config,
}: {
  /** The dimensionlity of the currentl potential turning point, where if dimension > 0, it means that it contains a group of turning points, recursively. */
  dimension: number;
  /** The configuration settings for the turning point detector */
  config: TurningPointDetectorConfig;
  /** The first message to be analyzed */
  beforeMessage: Message | MetaMessage;
  /** The second message to be analyzed */
  afterMessage: Message | MetaMessage;
}): string {
  const userMessage =
    `
Analyze the content below to determine the classification and labels as instructed in the previous message. Carefully scrutinize the information provided, and ${
    // When the dimension is greater than 0, we are analyzing the classification of a grouping of two turning points. This scenario provides more relevant contextual information because it consolidates the message content related to the two assessed turning points into a meta-turning point. In contrast, if the dimension is 0, the system context only includes message content from neighboring messages that are outside the two being evaluated. Therefore, the content below will only reference the two turning points without including their associated message content, which is provided in the user message.

    dimension === 0
      ? "Refer to the contextual information from the system message to assist with your analysis. This information is designed to enhance your understanding of the surrounding messages, but it should not be used as a basis for formulating your response. The relevant content is included below (within this message), which lists the two messages being assessed as potential turning points within the broader conversation."
      : "Refer to the contextual information from the system message to assist with your analysis. This information aims to enhance your understanding of the surrounding turning points outside the two being assessed. Furthermore, the contextual information from the system prompt provides the actual content of messages related to those turning points for your analysis.\n- Use the system context solely to aid in comprehending the task at hand, rather than as a foundation for your response.\n\nThe relevant content below (pertaining to the two turning points) should primarily guide your analysis. You may refer to the system context for the actual conversation messages and their content associated with the two turning points if the provided content below is insufficient for formulating your response:"
    }\n\n` + // further content below truncated but implemented as added dynamic string
    `[${dimension === 0
      ? "First Message of the two messages being analyzed as a potential turning point in the entire conversation - The content below for this message is all from the author: 'Ziping Liu'"
      : 'First Turning Point within the Group of Turning Points that encapsulate a single conversation being assessed into a Single, "Meta" Turning Point'
    } ${dimension === 0 ? "Author" : "Source"}: ${beforeMessage.author}, ID: "${beforeMessage.id}"]
${dimension === 0
      ? returnFormattedMessageContent(config, beforeMessage, dimension)
      : beforeMessage.message
        .split("\n")
        .map((line) => `   ${line}`)
        .join("\n")
    }

[${dimension === 0
      ? "Second Message of the two messages being analyzed as a potential turning point in the entire conversation - The content below is all from the author 'ChatGPT'"
      : 'Last Turning Point within the Group of Turning Points that encapsulate a single conversation being assessed into a Single, "Meta" Turning Point'
    } ${dimension === 0 ? "Author" : "Source"}: ${afterMessage.author}, ID: "${afterMessage.id}"]
${dimension === 0
      ? returnFormattedMessageContent(config, afterMessage, dimension)
      : afterMessage.message
        .split("\n")
        .map((line) => `   ${line}`)
        .join("\n")
    }` +

    `\n\n------ end of content to analyze, now see below for you response instructions and task as a reminder ------\n\n# Response Format and Task Reminder\n- With the given content above, as well as contextual info and in accordance with the system instructions, please ensure your respond only with the instructed response foramt json stringified ojbect, for example:\n\n{ 
    "label": "<YOUR CREATIVE TITLE LABEL HERE> e.g 'Progressing from General Budgets to Budget Concerns', or 'Analysis on the notions of Leadership and Teamwork'",
    "keywords": ["<KEYWORD1>", "<KEYWORD2>", "<KEYWORD3>"],
    "sentiment": "<SENTIMENT HERE>, as one of the following: 'positive', 'negative'",
    "significance": <SIGNIFICANCE SCORE HERE, from 0-100>,   
    "category": "<Insert your categorization here using one of the following options: ${Object.keys(
      categories
    )
      .sort(() => Math.random() - 0.5)
      .map((c) => `\`${c}\``)
      .join(", ")}>",\n` +
    `   "quotes": [\n      "Some Author name: Some quote here 1",\n      "Some Author name: Some quote here 2"\n   ],\n` +
    `   "emotionalTone": "<Select your emotional tone from these values: ${emotionalTones
      .sort(() => Math.random() - 0.5)
      .map((c) => `'${c}'`)
      .join(", ")}>"\n` +
    `}\n`;

  return userMessage;
}

export const formResponseFormatSchema = (dimension: number) => ({
  type: "json_schema",
  json_schema: {
    name: "turning_point_classification_format",
    strict: true,
    schema: {
      type: "object",
      properties: {

        label: {
          type: "string",
          description: `A short, creative labeling, used as a title, meant to encompass this potential turning point ${dimension === 0
            ? "between these two messages, and utilizing the contextual information fromthe system message which contains neighboring messages ot help better formulate this creative title"
            : "between these two turning points, and utilizing the contextual information from the system message which contains neighboring messages to help better formulate this creative title"
            }. This should clearly describe the potential turning point that occurred.`,
        },

        emotionalTone: {
          type: "string",
          description: `An emotional tone that best describes the content presented in the user message. You must choose a value from this fixed list, else your response will be reported and flagged: ${emotionalTones.sort(
            // random
            () => Math.random() - 0.5,
          ).join(", ")}.`


        },

        keywords: {

          "type": "array",
          "description": "A list of entities, or keywords, phrases that are relevant to the presented content to analyze form the user message, as well as the presented contextual information from the system message.",
          "items": {
            "type": "string",
           }
        },
 
        significance: {
          type: "number",
          description:
            "A significance score from (0-100) representing how important this turning point is to the overall conversation, or the aspect of the turning point's level of shift that it represents. Higher values indicate a more significant turning point, assess this carefully, and no lower than 0 or higher than 100. This should be a decimal number.",
        },
        category: {
          type: "string",
          description: `A value from one of the following values only: ${Object.keys(
            categories,
          )
            .sort(() => Math.random() - 0.5)
            .map((c) => `\`${c}\``)
            .join(", ")}`,
        },
      },
      required: [
        "label",
        "sentiment",
        "significance",
        "category",
        "keywords",
        
        "emotionalTone",
      ],
      additionalProperties: false,
    },
  },
} as ResponseFormatJSONSchema);


export const formSystemPromptEnding = (dimension: number) => {

  const ending = `# **Analysis Steps and Response Format:**\n\nTo clarify your task and the expected output, please follow these steps:\n
1. Review the 'Contextual Aid' messages (if provided) to grasp the conversational flow.
2. Carefully compare the content and interaction between the two ${dimension === 0 ? "messages" : "turning points"
    }, provided in the user mesage content.
3. Identify the *primary nature* of the changes between the two ${dimension === 0 ? "messages" : "turning points"
    }, provided in the user message content.
4. A short, specific creative, title label for this turning point (e.g., "Shift from Discussion of General Budgets to Argument over Budget Concerns")
  - This should infact utilize not only the content provided in the following user message, but also the contextual information provided above which provides the content of neighboring ${dimension === 0 ? "messages" : "turning points"
    } that occur before and/or after the two ${dimension === 0 ? "messages" : "turning points"} being analyzed to creatively capture the essence of this potential turning point being analyzed (as provided of the two ${dimension === 0 ? "messages" : "turning points"} in the user message content).
6. A significance score (0-100) representing how important this turning point is to the overall conversation
7. A category, one of the following values: ${Object.keys(
      categories
    ).sort(() => Math.random() - 0.5)}, which categorizes the ocntent presented in the user message content.
8. Respond with only one exact categorization for the emotionalTone of the content, from the provided list of emotional tones above. DO NOT USE ANY OTHER VALUES, ONLY ONES FROM THE LISTED EMOTIONAL TONES, DO NOT CONFUSE THE NOTION OF emotionalTone to sentiment, emotionalTone MAY NEVER BE associated with a value of 'postiive' or 'negative', please refer to the list of emotional tones above for the correct values to use.
Respond with a JSON object containing these fields. Do not include any text outside the JSON object.`
  return ending;
}