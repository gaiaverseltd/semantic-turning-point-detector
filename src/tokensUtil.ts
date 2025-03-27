// src/tokenUtils.ts
import GPTTokenizer from 'gpt-tokenizer';

/**
 * Count tokens in a given string using OpenAI-compatible tokenization.
 * 
 * @param text - The text to tokenize
 * @returns Number of tokens
 */
export function countTokens(text: string): number {
  
  return GPTTokenizer.encode(text).length;
}