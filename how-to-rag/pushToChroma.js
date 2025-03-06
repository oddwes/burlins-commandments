import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChromaClient } from "chromadb";
import { CHROMA_CONFIG } from "./chromaConfig.js";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import fs from "fs/promises";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

// Get the equivalent of __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables from .env file (one directory up)
dotenv.config({ path: path.resolve(__dirname, ".env") });

// Parse command line arguments
const args = process.argv.slice(2);
const inputFilePath = args[0];

if (!inputFilePath) {
  console.error("Error: Please provide a file path as an argument");
  console.error("Usage: node pushToChroma.js <file_path>");
  process.exit(1);
}

async function loadContent() {
  console.log("Loading content from input file...");
  const documents = [];

  try {
    // Resolve the file path (handle both absolute and relative paths)
    const resolvedPath = path.isAbsolute(inputFilePath)
      ? inputFilePath
      : path.resolve(process.cwd(), inputFilePath);

    console.log(`Reading file from: ${resolvedPath}`);

    // Check file extension to determine how to load it
    const fileExtension = path.extname(resolvedPath).toLowerCase();

    if (fileExtension === ".pdf") {
      // Handle PDF files using PDFLoader
      console.log("Detected PDF file, using PDFLoader...");
      const pdfLoader = new PDFLoader(resolvedPath);
      const pdfDocuments = await pdfLoader.load();

      // Add source metadata to each document
      const fileName = path.basename(resolvedPath);
      pdfDocuments.forEach((doc) => {
        doc.metadata.source = fileName;
        doc.metadata.filePath = resolvedPath;
      });

      documents.push(...pdfDocuments);
      console.log(
        `✓ Loaded ${pdfDocuments.length} pages from PDF: ${fileName}`
      );
    } else {
      // Handle text-based files (JSON, TXT, etc.)
      const content = await fs.readFile(resolvedPath, "utf8");

      if (content) {
        const fileName = path.basename(resolvedPath);
        documents.push({
          pageContent: content,
          metadata: {
            source: fileName,
            filePath: resolvedPath,
          },
        });
        console.log(`✓ Loaded content from ${fileName}`);
      } else {
        console.warn(`No content found in file: ${resolvedPath}`);
      }
    }
  } catch (error) {
    console.error(`Error reading file ${inputFilePath}:`, error);
    process.exit(1);
  }

  console.log(`Loaded content from ${documents.length} documents/pages`);
  return documents;
}

async function splitDocuments(documents) {
  console.log("Splitting documents...");

  const allChunks = [];

  for (const document of documents) {
    try {
      console.log(
        `Splitting document from source: ${document.metadata.source}`
      );

      // For PDF files or non-JSON content, use the text splitter directly
      if (
        document.metadata.source.toLowerCase().endsWith(".pdf") ||
        !tryParseAsJson(document.pageContent)
      ) {
        console.log(
          `Processing as text document (${
            document.metadata.source.toLowerCase().endsWith(".pdf")
              ? "PDF"
              : "non-JSON text"
          })`
        );

        const textSplitter = new RecursiveCharacterTextSplitter({
          chunkSize: 1000,
          chunkOverlap: 100,
        });

        // Use the document with its original metadata
        const textChunks = await textSplitter.splitDocuments([document]);

        // Ensure each chunk has the correct source metadata from the original document
        textChunks.forEach((chunk, index) => {
          chunk.metadata = {
            ...document.metadata,
            chunk_id: index,
            total_chunks: textChunks.length,
          };
        });

        allChunks.push(...textChunks);
        continue;
      }

      // Process JSON content
      let jsonContent;
      try {
        jsonContent = JSON.parse(document.pageContent);

        // Check if it's an array
        if (!Array.isArray(jsonContent)) {
          console.log(
            "JSON content is not an array, treating as a single object"
          );
          jsonContent = [jsonContent];
        }
      } catch (error) {
        // This should not happen as we already checked with tryParseAsJson
        console.warn(`Unexpected error parsing JSON content: ${error.message}`);
        continue;
      }

      // Process each object in the JSON array as a separate chunk
      jsonContent.forEach((item, index) => {
        allChunks.push({
          pageContent: JSON.stringify(item, null, 2),
          metadata: {
            ...document.metadata,
            chunk_id: index,
            total_chunks: jsonContent.length,
          },
        });
      });

      console.log(
        `Split JSON array into ${jsonContent.length} individual object chunks`
      );
    } catch (error) {
      console.error(
        `Error splitting document from ${document.metadata.source}:`,
        error
      );
    }
  }

  console.log(`Split into ${allChunks.length} total chunks`);
  return allChunks;
}

// Helper function to safely check if content is valid JSON
function tryParseAsJson(content) {
  try {
    JSON.parse(content);
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Reconstructs original content from document chunks grouped by source
 * @param {Object} allDocuments - Documents retrieved from ChromaDB
 * @returns {Object} - Map of source names to reconstructed content
 */
async function reconstructContentBySource(allDocuments) {
  console.log("Reconstructing content by source...");
  const contentBySource = {};

  // Group documents by source
  for (let i = 0; i < allDocuments.documents.length; i++) {
    const source = allDocuments.metadatas[i].source;
    const content = allDocuments.documents[i];

    if (!contentBySource[source]) {
      contentBySource[source] = [];
    }

    contentBySource[source].push(content);
  }

  // Join chunks for each source
  Object.keys(contentBySource).forEach((source) => {
    contentBySource[source] = contentBySource[source].join(" ");
  });

  console.log(
    `Reconstructed content for ${Object.keys(contentBySource).length} sources`
  );
  return contentBySource;
}

async function addToChroma(chunks) {
  const embedding = new OpenAIEmbeddings({
    batchSize: 512,
    model: "text-embedding-3-large",
  });

  try {
    console.log("Testing embedding creation...");
    const testEmbedding = await embedding.embedQuery("test");
    console.log(`✓ Created test embedding of length ${testEmbedding.length}`);

    const client = new ChromaClient({ path: CHROMA_CONFIG.url });
    let collection;

    // Check if collection exists and delete it
    try {
      console.log(
        `Checking for existing collection: ${CHROMA_CONFIG.collectionName}`
      );
      await client.deleteCollection({
        name: CHROMA_CONFIG.collectionName,
      });
      console.log("✓ Deleted existing collection");
    } catch (e) {
      console.log("No existing collection to delete");
    }

    // Create a new collection
    console.log("Creating new collection");
    collection = await client.createCollection({
      name: CHROMA_CONFIG.collectionName,
      metadata: CHROMA_CONFIG.collectionMetadata,
      embeddingFunction: {
        dimensionality: testEmbedding.length,
      },
      hnsw: {
        ef_construction: 200,
        M: 16,
      },
    });

    // Add new chunks in batches
    const batchSize = 50;
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batchChunks = chunks.slice(i, i + batchSize);

      console.log(
        `Processing batch ${i / batchSize + 1} of ${Math.ceil(
          chunks.length / batchSize
        )}`
      );

      const batchEmbeddings = await embedding.embedDocuments(
        batchChunks.map((chunk) => chunk.pageContent)
      );

      const batchIds = batchChunks.map(
        (chunk, idx) => `${chunk.metadata.source}_chunk_${i + idx}`
      );

      await collection.add({
        ids: batchIds,
        embeddings: batchEmbeddings,
        metadatas: batchChunks.map((chunk, idx) => ({
          ...chunk.metadata,
          id: batchIds[idx],
          source: chunk.metadata.source || "unknown",
          timestamp: new Date().toISOString(),
        })),
        documents: batchChunks.map((chunk) => chunk.pageContent),
      });

      console.log(`Added batch ${i / batchSize + 1}`);
    }

    // Add improved verification step
    console.log("Verifying all content was properly added...");

    // Get all documents from the collection
    const allDocuments = await collection.get({
      include: ["metadatas", "documents"],
    });

    // Check if we have the expected number of documents
    if (allDocuments.ids.length !== chunks.length) {
      console.warn(
        `Document count mismatch: Expected ${chunks.length} documents but found ${allDocuments.ids.length} in ChromaDB`
      );
    }

    // Reconstruct content by source
    const reconstructedContentBySource = await reconstructContentBySource(
      allDocuments
    );

    // Verify content by source
    try {
      // Read the original file content again
      const resolvedPath = path.isAbsolute(inputFilePath)
        ? inputFilePath
        : path.resolve(process.cwd(), inputFilePath);

      const originalContent = await fs.readFile(resolvedPath, "utf8");
      const fileName = path.basename(resolvedPath);

      // Get reconstructed content for this source
      const reconstructedContent = reconstructedContentBySource[fileName] || "";
      if (!reconstructedContent) {
        console.warn(`No reconstructed content found for source: ${fileName}`);
      } else {
        // Check if all key parts of the original content are in the reconstructed content
        // We're checking for inclusion rather than exact match because of chunking
        const contentVerified = originalContent
          .split("\n")
          .filter((line) => line.trim().length > 20) // Only check substantial lines
          .slice(0, 10) // Check a sample of lines for efficiency
          .every((line) => reconstructedContent.includes(line.trim()));

        if (contentVerified) {
          console.log(`✓ Verified content for source: ${fileName}`);
        } else {
          console.warn(
            `⚠️ Content verification failed for source: ${fileName}`
          );
        }
      }
    } catch (error) {
      console.error(`Error verifying content for input file:`, error);
    }

    console.log("Content verification completed");

    return true;
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
    console.log(`Processing input file: ${inputFilePath}`);
    const documents = await loadContent();
    if (!documents || documents.length === 0) {
      throw new Error("No content loaded from input file");
    }
    console.log(`Loaded ${documents.length} documents`);

    const chunks = await splitDocuments(documents);
    if (!chunks || chunks.length === 0) {
      throw new Error("No chunks created from documents");
    }
    console.log(`Created ${chunks.length} chunks`);

    await addToChroma(chunks);
    console.log("✅ Successfully added all content to ChromaDB");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main().catch(console.error);

export { loadContent, splitDocuments, addToChroma, reconstructContentBySource };
