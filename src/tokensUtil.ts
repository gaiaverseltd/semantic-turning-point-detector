// src/tokenUtils.ts
import GPTTokenizer from 'gpt-tokenizer';
import OpenAI from 'openai';
import dotenv from 'dotenv';
import crypto from 'crypto';
import {LRUCache} from 'lru-cache';

 

dotenv.config();
/**
 * Count tokens in a given string using OpenAI-compatible tokenization.
 * 
 * @param text - The text to tokenize
 * @param modelName - Optional model name to specify the tokenizer variant, though not used as the default is sufficient.
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
  model?: string,
  cache?: LRUCache<string, Float32Array>
): Promise<Float32Array> {
  // Create a hash-based cache key instead of using raw text
  const cacheKey = cache ? crypto
    .createHash('sha256')
    .update(`${model || 'default'}:${text}`)
    .digest('hex') : '';
  
  // Check cache if provided
  if (cache && cacheKey) {
    const cached = cache.get(cacheKey);
    if (cached) return cached;
  }

  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const response = await openai.embeddings.create({
    model: model,
    input: text,
    encoding_format: "float",
  });
  
  const embedding = new Float32Array(response.data?.[0].embedding);
  
  // Store in cache if provided
  if (cache && cacheKey) {
    cache.set(cacheKey, embedding);
  }
  
  return embedding;
}



/**
 * Creates a new LRU cache for embeddings with RAM limit
 */
export function createEmbeddingCache(ramLimitMB: number = 100, ttlSeconds = 600): LRUCache<string, Float32Array> {
  const embeddingSize = 3072 * 4; // text-embedding-3-large: 3072 dimensions * 4 bytes per float
  const maxEntries = Math.floor((ramLimitMB * 1024 * 1024) / embeddingSize);
  
  return new LRUCache<string, Float32Array>({
    max: maxEntries,
    ttl: ttlSeconds * 1000, // Convert seconds to milliseconds
  });
}