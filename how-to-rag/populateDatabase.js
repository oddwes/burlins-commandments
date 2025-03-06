import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChromaClient } from "chromadb";
import path from "node:path";
import { CHROMA_CONFIG } from "./chromaConfig.js";

const PDF_DIRECTORY = "materials";

async function loadDocuments() {
  console.log("Loading PDFs...");
  const directoryLoader = new DirectoryLoader(
    path.join(process.cwd(), PDF_DIRECTORY),
    {
      ".pdf": (path) => new PDFLoader(path),
    }
  );

  const documents = await directoryLoader.load();
  console.log(`Loaded ${documents.length} pages from PDF files`);
  return documents;
}

async function splitDocuments(documents) {
  console.log("Splitting documents...");
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 800,
    chunkOverlap: 80,
  });

  const texts = await textSplitter.splitDocuments(documents);
  console.log(`Split into ${texts.length} chunks`);
  return texts;
}

async function addToChroma(chunks) {
  const embedding = new OpenAIEmbeddings({
    batchSize: 512,
    model: "text-embedding-3-large",
  });

  try {
    // First verify we can create embeddings
    console.log("Testing embedding creation...");
    const testEmbedding = await embedding.embedQuery("test");
    console.log(
      `✓ Successfully created test embedding of length ${testEmbedding.length}`
    );

    // Create a direct ChromaDB client
    const client = new ChromaClient({ path: CHROMA_CONFIG.url });

    // Delete collection if it exists
    try {
      await client.deleteCollection({ name: CHROMA_CONFIG.collectionName });
      console.log("Deleted existing collection");
    } catch (e) {
      console.log("No existing collection to delete");
    }

    // Create new collection
    const collection = await client.createCollection({
      name: CHROMA_CONFIG.collectionName,
      metadata: CHROMA_CONFIG.collectionMetadata,
    });

    console.log("Created new collection");

    // Add documents in batches
    const batchSize = 50;
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batchChunks = chunks.slice(i, i + batchSize);

      console.log(
        `Processing batch ${i / batchSize + 1} of ${Math.ceil(
          chunks.length / batchSize
        )}`
      );

      // Create embeddings for this batch
      const batchEmbeddings = await embedding.embedDocuments(
        batchChunks.map((chunk) => chunk.pageContent)
      );

      // Generate IDs
      const batchIds = batchChunks.map((_, idx) => `chunk_${i + idx}`);

      // Add to collection
      await collection.add({
        ids: batchIds,
        embeddings: batchEmbeddings,
        metadatas: batchChunks.map((chunk, idx) => ({
          ...chunk.metadata,
          id: batchIds[idx], // Include the ID in metadata
          source: chunk.metadata.source || "unknown",
          page: chunk.metadata.page || 0,
        })),
        documents: batchChunks.map((chunk) => chunk.pageContent),
      });

      console.log(`Added batch ${i / batchSize + 1}`);
    }

    console.log("All documents added. Testing queries...");

    // Test query with first chunk
    const sampleChunk = chunks[0];
    const queryEmbedding = await embedding.embedQuery(sampleChunk.pageContent);

    const results = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: 1,
    });

    console.log("Query results:", {
      nResults: results.documents[0].length,
      firstResult: results.documents[0][0]?.substring(0, 100),
    });

    // Create LangChain Chroma instance for return
    return new Chroma(embedding, CHROMA_CONFIG);
  } catch (error) {
    console.error("Error in addToChroma:", error);
    if (error.response?.data) {
      console.error("Response data:", error.response.data);
    }
    throw error;
  }
}

async function main() {
  try {
    const documents = await loadDocuments();
    if (!documents || documents.length === 0) {
      throw new Error("No documents loaded from PDF directory");
    }
    console.log(`Loaded ${documents.length} documents`);

    const chunks = await splitDocuments(documents);
    if (!chunks || chunks.length === 0) {
      throw new Error("No chunks created from documents");
    }
    console.log(`Created ${chunks.length} chunks`);

    await addToChroma(chunks);
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main().catch(console.error);

export { loadDocuments, splitDocuments, addToChroma };
