import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { env, pipeline } from '@huggingface/transformers';
import { buildScheduleSemanticTexts } from '../src/shared/semantic.js';
import {
  SEMANTIC_LOCAL_MODEL_ROOT,
  SEMANTIC_MODEL_DTYPE,
  SEMANTIC_MODEL_ID,
  SEMANTIC_MODEL_SUBFOLDER,
} from '../src/shared/semantic-config.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');

const publicRoot = path.join(appRoot, 'public');
const scheduleManifestPath = path.join(publicRoot, 'data', 'schedule_manifest.json');
const qualityReportPath = path.join(publicRoot, 'data', 'semantic_int8_quality_report.json');
const legacyOutBinPath = path.join(publicRoot, 'data', 'schedule_semantic_embeddings_q4.bin');
const legacyOutMetaPath = path.join(publicRoot, 'data', 'schedule_semantic_embeddings_q4.json');

const BATCH_SIZE = 48;
const INT8_MAX = 127;
const SINGLE_QUERY_COUNT = 128;
const CENTROID_QUERY_COUNT = 48;
const CENTROID_SIZE = 5;
const QUALITY_THRESHOLDS = {
  minAvgTop10Overlap: 0.97,
  minAvgTop50Overlap: 0.985,
  minTop1ExactRate: 0.9,
  minTop1InTop5Rate: 0.985,
  maxMeanAbsError: 0.0025,
  maxAbsError: 0.014,
};

function resolveLocalModelRoot() {
  const trimmed = String(SEMANTIC_LOCAL_MODEL_ROOT || '/models').replace(/^\/+/, '');
  return path.resolve(appRoot, 'public', trimmed);
}

function assertFileExists(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found at ${filePath}`);
  }
}

function publicPathFromUrl(urlPath) {
  const normalized = String(urlPath || '').replace(/^\/+/, '');
  return path.join(publicRoot, normalized);
}

function rowsFromTensor(output) {
  const data = output?.data;
  const dims = output?.dims;
  if (!data || !Array.isArray(dims) || dims.length < 2) {
    throw new Error('Unexpected embedding tensor output.');
  }
  const count = dims[0];
  const dim = dims[dims.length - 1];
  const vectors = new Array(count);
  for (let i = 0; i < count; i += 1) {
    const start = i * dim;
    const end = start + dim;
    const vec = new Float32Array(dim);
    vec.set(data.subarray(start, end));
    vectors[i] = vec;
  }
  return { vectors, dim };
}

function quantizeVectorsPerRowInt8(vectors, dimension) {
  const rowCount = vectors.length;
  const scales = new Float32Array(rowCount);
  const quantized = new Int8Array(rowCount * dimension);

  for (let row = 0; row < rowCount; row += 1) {
    const vector = vectors[row];
    let maxAbs = 0;
    for (let dim = 0; dim < dimension; dim += 1) {
      maxAbs = Math.max(maxAbs, Math.abs(vector[dim]));
    }
    const scale = maxAbs > 0 ? maxAbs / INT8_MAX : 1 / INT8_MAX;
    scales[row] = scale;

    const offset = row * dimension;
    for (let dim = 0; dim < dimension; dim += 1) {
      const scaled = Math.round(vector[dim] / scale);
      quantized[offset + dim] = Math.max(-INT8_MAX, Math.min(INT8_MAX, scaled));
    }
  }

  return {
    scales,
    quantized,
    buffer: Buffer.concat([
      Buffer.from(scales.buffer, scales.byteOffset, scales.byteLength),
      Buffer.from(quantized.buffer, quantized.byteOffset, quantized.byteLength),
    ]),
  };
}

function normalizeVector(vec) {
  let normSq = 0;
  for (let idx = 0; idx < vec.length; idx += 1) {
    normSq += vec[idx] * vec[idx];
  }
  const norm = Math.sqrt(normSq);
  if (!norm) return vec;
  for (let idx = 0; idx < vec.length; idx += 1) {
    vec[idx] /= norm;
  }
  return vec;
}

function makeSeed(input) {
  let seed = 2166136261;
  for (const char of String(input)) {
    seed ^= char.charCodeAt(0);
    seed = Math.imul(seed, 16777619);
  }
  return seed >>> 0;
}

function nextRandom(state) {
  state.seed = (Math.imul(state.seed, 1664525) + 1013904223) >>> 0;
  return state.seed / 0x100000000;
}

function buildQualityQueries(vectors, dimension, conferenceKey) {
  const queries = [];
  const rowCount = vectors.length;
  const singleCount = Math.min(SINGLE_QUERY_COUNT, rowCount);
  for (let idx = 0; idx < singleCount; idx += 1) {
    const row = singleCount === 1 ? 0 : Math.floor((idx * (rowCount - 1)) / (singleCount - 1));
    queries.push(vectors[row]);
  }

  const randomState = { seed: makeSeed(conferenceKey) };
  for (let query = 0; query < CENTROID_QUERY_COUNT; query += 1) {
    const centroid = new Float32Array(dimension);
    for (let item = 0; item < CENTROID_SIZE; item += 1) {
      const row = Math.floor(nextRandom(randomState) * rowCount);
      const vector = vectors[row];
      for (let dim = 0; dim < dimension; dim += 1) {
        centroid[dim] += vector[dim];
      }
    }
    for (let dim = 0; dim < dimension; dim += 1) {
      centroid[dim] /= CENTROID_SIZE;
    }
    queries.push(normalizeVector(centroid));
  }
  return queries;
}

function dotFloat32(vector, query, dimension) {
  let dot = 0;
  for (let dim = 0; dim < dimension; dim += 1) {
    dot += vector[dim] * query[dim];
  }
  return dot;
}

function dotInt8Normalized(quantized, scales, row, query, dimension) {
  let dot = 0;
  const offset = row * dimension;
  const scale = scales[row];
  let normSq = 0;
  for (let dim = 0; dim < dimension; dim += 1) {
    const value = quantized[offset + dim] * scale;
    dot += value * query[dim];
    normSq += value * value;
  }
  const norm = Math.sqrt(normSq) || 1;
  return dot / norm;
}

function topKIndexes(scores, k) {
  return Array.from(scores.keys())
    .sort((left, right) => scores[right] - scores[left])
    .slice(0, k);
}

function overlapRate(left, right) {
  const rightSet = new Set(right);
  let overlap = 0;
  for (const item of left) {
    if (rightSet.has(item)) overlap += 1;
  }
  return overlap / Math.max(1, left.length);
}

function evaluateInt8Quality({ conference, vectors, quantized, dimension }) {
  const rowCount = vectors.length;
  const queries = buildQualityQueries(vectors, dimension, conference.key);
  const totals = {
    avgTop10Overlap: 0,
    avgTop50Overlap: 0,
    top1Exact: 0,
    top1InTop5: 0,
    meanAbsError: 0,
    maxAbsError: 0,
  };
  let scoredPairs = 0;

  for (const query of queries) {
    const floatScores = new Float32Array(rowCount);
    const int8Scores = new Float32Array(rowCount);
    for (let row = 0; row < rowCount; row += 1) {
      const floatScore = dotFloat32(vectors[row], query, dimension);
      const int8Score = dotInt8Normalized(quantized.quantized, quantized.scales, row, query, dimension);
      floatScores[row] = floatScore;
      int8Scores[row] = int8Score;
      const absError = Math.abs(floatScore - int8Score);
      totals.meanAbsError += absError;
      if (absError > totals.maxAbsError) totals.maxAbsError = absError;
      scoredPairs += 1;
    }

    const floatTop10 = topKIndexes(floatScores, 10);
    const int8Top10 = topKIndexes(int8Scores, 10);
    const floatTop50 = topKIndexes(floatScores, 50);
    const int8Top50 = topKIndexes(int8Scores, 50);
    totals.avgTop10Overlap += overlapRate(floatTop10, int8Top10);
    totals.avgTop50Overlap += overlapRate(floatTop50, int8Top50);
    if (floatTop10[0] === int8Top10[0]) totals.top1Exact += 1;
    if (int8Top10.slice(0, 5).includes(floatTop10[0])) totals.top1InTop5 += 1;
  }

  totals.avgTop10Overlap /= queries.length;
  totals.avgTop50Overlap /= queries.length;
  totals.top1Exact /= queries.length;
  totals.top1InTop5 /= queries.length;
  totals.meanAbsError /= scoredPairs;

  const rawBytes = rowCount * dimension * 4;
  const int8Bytes = quantized.buffer.byteLength;
  return {
    key: conference.key,
    label: conference.label,
    rowCount,
    dimension,
    queries: queries.length,
    rawBytes,
    int8Bytes,
    ...totals,
  };
}

function aggregateQualityReports(reports) {
  let totalQueries = 0;
  let totalRowsScored = 0;
  let rawBytes = 0;
  let int8Bytes = 0;
  const aggregate = {
    avgTop10Overlap: 0,
    avgTop50Overlap: 0,
    top1Exact: 0,
    top1InTop5: 0,
    meanAbsError: 0,
    maxAbsError: 0,
  };

  for (const report of reports) {
    totalQueries += report.queries;
    totalRowsScored += report.queries * report.rowCount;
    rawBytes += report.rawBytes;
    int8Bytes += report.int8Bytes;
    aggregate.avgTop10Overlap += report.avgTop10Overlap * report.queries;
    aggregate.avgTop50Overlap += report.avgTop50Overlap * report.queries;
    aggregate.top1Exact += report.top1Exact * report.queries;
    aggregate.top1InTop5 += report.top1InTop5 * report.queries;
    aggregate.meanAbsError += report.meanAbsError * report.queries * report.rowCount;
    aggregate.maxAbsError = Math.max(aggregate.maxAbsError, report.maxAbsError);
  }

  aggregate.avgTop10Overlap /= totalQueries;
  aggregate.avgTop50Overlap /= totalQueries;
  aggregate.top1Exact /= totalQueries;
  aggregate.top1InTop5 /= totalQueries;
  aggregate.meanAbsError /= totalRowsScored;

  const passed =
    aggregate.avgTop10Overlap >= QUALITY_THRESHOLDS.minAvgTop10Overlap &&
    aggregate.avgTop50Overlap >= QUALITY_THRESHOLDS.minAvgTop50Overlap &&
    aggregate.top1Exact >= QUALITY_THRESHOLDS.minTop1ExactRate &&
    aggregate.top1InTop5 >= QUALITY_THRESHOLDS.minTop1InTop5Rate &&
    aggregate.meanAbsError <= QUALITY_THRESHOLDS.maxMeanAbsError &&
    aggregate.maxAbsError <= QUALITY_THRESHOLDS.maxAbsError;

  return {
    version: 1,
    generated_at: new Date().toISOString(),
    query_plan: {
      single_query_count: SINGLE_QUERY_COUNT,
      centroid_query_count: CENTROID_QUERY_COUNT,
      centroid_size: CENTROID_SIZE,
    },
    thresholds: QUALITY_THRESHOLDS,
    passed,
    aggregate: {
      ...aggregate,
      scoredPairs: totalRowsScored,
      rawBytes,
      int8Bytes,
    },
    conferences: reports,
  };
}

async function embedScheduleIndex({ extractor, conference }) {
  const scheduleIndexPath = publicPathFromUrl(conference.index_url);
  assertFileExists(scheduleIndexPath, `${conference.label || conference.key} schedule index`);
  const scheduleIndex = JSON.parse(fs.readFileSync(scheduleIndexPath, 'utf8'));
  const rows = Array.isArray(scheduleIndex.rows) ? scheduleIndex.rows : [];
  if (!rows.length) {
    throw new Error(`No schedule rows found for ${conference.key || scheduleIndexPath}.`);
  }

  const outBinPath = publicPathFromUrl(conference.semantic_embeddings_bin_url);
  const outMetaPath = publicPathFromUrl(conference.semantic_embeddings_meta_url);

  const texts = buildScheduleSemanticTexts(rows);
  console.log(
    `Embedding ${texts.length} rows for ${conference.label || conference.key} in batches of ${BATCH_SIZE}...`
  );

  const allVectors = [];
  let dimension = 0;
  for (let cursor = 0; cursor < texts.length; cursor += BATCH_SIZE) {
    const batch = texts.slice(cursor, cursor + BATCH_SIZE);
    const output = await extractor(batch, { pooling: 'mean', normalize: true });
    const { vectors, dim } = rowsFromTensor(output);
    if (!dimension) {
      dimension = dim;
    } else if (dimension !== dim) {
      throw new Error(`Embedding dimension mismatch (${dimension} vs ${dim}).`);
    }
    for (const vec of vectors) {
      allVectors.push(vec);
    }
    const processed = Math.min(cursor + batch.length, texts.length);
    const percent = Math.round((processed / texts.length) * 100);
    console.log(`${conference.label || conference.key}: processed ${processed}/${texts.length} (${percent}%)`);
  }

  const quantized = quantizeVectorsPerRowInt8(allVectors, dimension);
  const qualityReport = evaluateInt8Quality({
    conference,
    vectors: allVectors,
    quantized,
    dimension,
  });

  fs.mkdirSync(path.dirname(outBinPath), { recursive: true });
  fs.writeFileSync(outBinPath, quantized.buffer);

  const metadata = {
    version: 3,
    sharded: true,
    conference_key: conference.key || scheduleIndex.shard_key || null,
    conference_label: conference.label || null,
    model_id: SEMANTIC_MODEL_ID,
    dtype: SEMANTIC_MODEL_DTYPE,
    row_count: allVectors.length,
    dimension,
    format: 'int8_symmetric_per_row',
    layout: 'scales_float32le_then_int8',
    scale_count: quantized.scales.length,
    quantization: {
      type: 'symmetric_per_row_absmax',
      quantized_min: -INT8_MAX,
      quantized_max: INT8_MAX,
      scale_dtype: 'float32le',
      vector_dtype: 'int8',
      dequantize: 'value = quantized_value * row_scale; row vectors are normalized after dequantization',
    },
    generated_at: new Date().toISOString(),
    source_schedule_version: scheduleIndex.version || null,
    source_schedule_index_url: conference.index_url || null,
  };
  fs.writeFileSync(outMetaPath, `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');

  const mb = (quantized.buffer.byteLength / (1024 * 1024)).toFixed(2);
  console.log(`Wrote embeddings: ${outBinPath} (${mb} MB)`);
  console.log(`Wrote metadata: ${outMetaPath}`);

  return {
    rowCount: allVectors.length,
    byteLength: quantized.buffer.byteLength,
    qualityReport,
  };
}

async function main() {
  assertFileExists(scheduleManifestPath, 'schedule_manifest.json');
  const manifest = JSON.parse(fs.readFileSync(scheduleManifestPath, 'utf8'));
  const conferences = Array.isArray(manifest.conferences) ? manifest.conferences : [];
  if (!conferences.length) {
    throw new Error('No conferences found in schedule manifest.');
  }

  const localModelRoot = resolveLocalModelRoot();
  assertFileExists(path.join(localModelRoot, SEMANTIC_MODEL_ID, 'config.json'), 'Local semantic model');

  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = localModelRoot;

  console.log(`Loading embedding model '${SEMANTIC_MODEL_ID}' from ${localModelRoot}...`);
  const extractor = await pipeline('feature-extraction', SEMANTIC_MODEL_ID, {
    local_files_only: true,
    dtype: SEMANTIC_MODEL_DTYPE,
    subfolder: SEMANTIC_MODEL_SUBFOLDER,
  });

  let totalRows = 0;
  let totalBytes = 0;
  const qualityReports = [];
  for (const conference of conferences) {
    const result = await embedScheduleIndex({ extractor, conference });
    totalRows += result.rowCount;
    totalBytes += result.byteLength;
    qualityReports.push(result.qualityReport);
  }

  const legacyMetadata = {
    version: 2,
    sharded: true,
    model_id: SEMANTIC_MODEL_ID,
    dtype: SEMANTIC_MODEL_DTYPE,
    row_count: totalRows,
    conference_count: conferences.length,
    format: 'float32le',
    generated_at: new Date().toISOString(),
    manifest_url: 'data/schedule_manifest.json',
    note: 'Schedule semantic embeddings are stored per conference under data/conferences/<conference-key>/.',
  };
  fs.writeFileSync(legacyOutMetaPath, `${JSON.stringify(legacyMetadata, null, 2)}\n`, 'utf8');
  fs.writeFileSync(legacyOutBinPath, Buffer.alloc(0));

  const qualityReport = aggregateQualityReports(qualityReports);
  fs.writeFileSync(qualityReportPath, `${JSON.stringify(qualityReport, null, 2)}\n`, 'utf8');
  if (!qualityReport.passed) {
    throw new Error('Generated int8 semantic quality report failed threshold.');
  }

  const mb = (totalBytes / (1024 * 1024)).toFixed(2);
  console.log(`Wrote ${totalRows} sharded embeddings across ${conferences.length} conferences (${mb} MB total)`);
  console.log(`Wrote int8 quality report: ${qualityReportPath}`);
  console.log(`Wrote lightweight legacy metadata: ${legacyOutMetaPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
