// src/tokenUtils.ts
import GPTTokenizer from 'gpt-tokenizer';
import OpenAI from 'openai';
import dotenv from 'dotenv';
dotenv.config();
/**
 * Count tokens in a given string using OpenAI-compatible tokenization.
 * 
 * @param text - The text to tokenize
 * @returns Number of tokens
 */
export function countTokens(text: string): number {
  
  return GPTTokenizer.encode(text).length;
}


/**
 * Generates an embedding for a given text using the OpenAI API
 * This provides the vector representation for semantic distance calculation
 */
export async function generateEmbedding(
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
