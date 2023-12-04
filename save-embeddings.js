import { pipeline } from '@xenova/transformers';
import fs from 'fs';

const extractor = await pipeline(
  'feature-extraction',
  'Xenova/bge-small-en-v1.5'
);

const raw = fs.readFileSync('tv.txt', 'utf-8');
const shows = raw.split(/\n+/);

const outputJSON = { embeddings: [] };

for (const show of shows) {
  console.log(`Extracting embedding for: ${show}`);
  const embedding = await extractor(show, {
    pooling: 'mean',
    normalize: true,
  });
  outputJSON.embeddings.push({
    show,
    embedding: embedding.tolist()[0],
  });
}

const output = JSON.stringify(outputJSON, null, 2);
fs.writeFileSync('tv-embeddings.json', output);

// const source = 'Choo choo!';
// console.log(embeddings.data[0]);
