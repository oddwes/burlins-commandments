## Setup

```
cp .env.example .env
docker run -d --name chroma -p 8000:8000 chromadb/chroma
npm i
```

Packages used: `@langchain/anthropic @langchain/core @langchain/openai @langchain/textsplitters @langchain/community pdf-parse path url dotenv fs chromadb`

## Execution

### Example 1

```
node pushToChroma.js "docs/ai-agent-store-data.json"
```

this command can be run multiple times. The DB will be cleared and re-populated on every run

```
node queryChroma.js "show me accounting agents" --nResults 2
```

### Example 2

```
node pushToChroma.js "docs/monopoly.pdf"
```

this command can be run multiple times. The DB will be cleared and re-populated on every run

```
node queryChroma.js "how to buy property"
```
