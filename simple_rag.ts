/**
 * Concise RAG pipeline in TypeScript (LangChain + Hugging Face embeddings)
 * - Reads 3 local Word docs
 * - Chunks text
 * - Embeds using HuggingFaceEmbeddings
 * - Stores in FAISS (in-memory) or Chroma (persistent)
 * - Runs a sample retrieval query using OpenAI LLM
 */

import * as fs from "fs";
import * as path from "path";
import "dotenv/config";

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface";
import { OpenAI } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { load } from "docx-parser";

const FILES = ["doc1.docx", "doc2.docx", "doc3.docx"]; // Local docx files
const VECTOR_STORE: "faiss" | "chroma" = "faiss"; // change to "chroma" if needed
const CHROMA_DIR = "./chroma_store";
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2"; // Hugging Face model
const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;

(async () => {
  console.log("Starting RAG pipeline...");

  // --- 1. Read documents ---
  const docs: Document[] = [];
  for (const file of FILES) {
    const filePath = path.resolve(file);
    const text = await load(filePath);
    docs.push(new Document({ pageContent: text }));
  }

  // --- 2. Chunk documents ---
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
  });
  const chunks = await splitter.splitDocuments(docs);
  console.log(`Total chunks: ${chunks.length}`);

  // --- 3. Embeddings ---
  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: EMBEDDING_MODEL,
  });

  // --- 4. Vector store ---
  let vectorStore;
  if (VECTOR_STORE === "faiss") {
    console.log("Using FAISS (in-memory) vector store...");
    vectorStore = await FaissStore.fromDocuments(chunks, embeddings);
  } else {
    console.log("Using Chroma (persistent) vector store...");
    vectorStore = await Chroma.fromDocuments(chunks, embeddings, {
      collectionName: "rag_docs",
      url: undefined,
      collectionMetadata: {},
      persistDirectory: CHROMA_DIR,
    });
  }

  // --- 5. LLM (OpenAI) ---
  const llm = new OpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0,
  });

  // --- 6. Create retrieval chain ---
  const retriever = vectorStore.asRetriever({ k: 4 });
  const chain = RetrievalQAChain.fromLLM(llm, retriever);

  // --- 7. Example query ---
  const query = "Summarize the main ideas from these documents.";
  console.log("\nQuery:", query);
  const result = await chain.call({ query });
  console.log("\nAnswer:\n", result.text);

  console.log("\n✅ RAG pipeline complete.");
})();
