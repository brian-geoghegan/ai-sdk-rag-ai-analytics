import { embed, embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import { cosineDistance, desc, gt, sql } from "drizzle-orm";
import { embeddings } from "../db/schema/embeddings";
import { db } from "../db";

const embeddingModelName = 'text-embedding-ada-002'
const embeddingModel = openai.embedding(embeddingModelName);

const generateChunks = (input: string, chunkSize: number = 1000): string[] => {
  const chunks: string[] = [];
  let index = 0;

  while (index < input.length) {
    chunks.push(input.slice(index, index + chunkSize));
    index += chunkSize;
  }

  return chunks;
};

export const generateEmbeddings = async (
  value: string,
): Promise<Array<{ embedding: number[]; content: string }>> => {
  const chunks = generateChunks(value);
  const { embeddings } = await embedMany({
    model: embeddingModel,
    values: chunks,
  });
  return embeddings.map((e, i) => ({ content: chunks[i], embedding: e }));
};

export const generateEmbedding = async (value: string): Promise<number[]> => {
  console.log('Embeddings model: ', embeddingModelName)

  console.log('Generating an embedding for text: ', value)
  const input = value.replaceAll("\n", " ");
  const { embedding } = await embed({
    model: embeddingModel,
    value: input,
  });
  console.log('Embedding: [', embedding.slice(0, 6).join(''), ',...]')
  return embedding;
};

export const findRelevantContent = async (userQuery: string) => {
  const userQueryEmbedded = await generateEmbedding(userQuery);
  const similarity = sql<number>`1 - (${cosineDistance(embeddings.embedding, userQueryEmbedded)})`;

  const similarGuides = await db
    .select({ name: embeddings.content, similarity })
    .from(embeddings)
    .where(gt(similarity, 0.3))
    .orderBy((t) => desc(t.similarity))
    .limit(4);
  return similarGuides;
};
