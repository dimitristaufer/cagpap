import fs from 'node:fs';
import path from 'node:path';

const publicRoot = path.resolve('public');
const manifestPath = path.join(publicRoot, 'data', 'schedule_manifest.json');
const reportPath = path.join(publicRoot, 'data', 'semantic_int8_quality_report.json');
const INT8_MAX = 127;
const TOP_KS = [10, 50];
const SINGLE_QUERY_COUNT = 128;
const CENTROID_QUERY_COUNT = 48;
const CENTROID_SIZE = 5;
const THRESHOLDS = {
  minAvgTop10Overlap: 0.97,
  minAvgTop50Overlap: 0.985,
  minTop1ExactRate: 0.9,
  minTop1InTop5Rate: 0.985,
  maxMeanAbsError: 0.0025,
  maxAbsError: 0.014,
};

function publicPathFromUrl(urlPath) {
  return path.join(publicRoot, String(urlPath || '').replace(/^\/+/, ''));
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function loadFloat32Vectors(conference) {
  const metaPath = publicPathFromUrl(conference.semantic_embeddings_meta_url);
  const binPath = publicPathFromUrl(conference.semantic_embeddings_bin_url);
  const meta = readJson(metaPath);
  if (meta.format !== 'float32le') {
    return null;
  }
  const rowCount = Number(meta.row_count);
  const dimension = Number(meta.dimension);
  const buffer = fs.readFileSync(binPath);
  const flat = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  if (flat.length !== rowCount * dimension) {
    throw new Error(`${conference.key} vector length mismatch.`);
  }
  return { flat, rowCount, dimension };
}

function quantizeSymmetricInt8PerRow(flat, rowCount, dimension) {
  const quantized = new Int8Array(flat.length);
  const scales = new Float32Array(rowCount);
  const norms = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row += 1) {
    const offset = row * dimension;
    let maxAbs = 0;
    for (let dim = 0; dim < dimension; dim += 1) {
      maxAbs = Math.max(maxAbs, Math.abs(flat[offset + dim]));
    }
    const scale = maxAbs > 0 ? maxAbs / INT8_MAX : 1 / INT8_MAX;
    scales[row] = scale;

    let normSq = 0;
    for (let dim = 0; dim < dimension; dim += 1) {
      const scaled = Math.round(flat[offset + dim] / scale);
      const quantizedValue = Math.max(-INT8_MAX, Math.min(INT8_MAX, scaled));
      quantized[offset + dim] = quantizedValue;
      const value = quantizedValue * scale;
      normSq += value * value;
    }
    norms[row] = Math.sqrt(normSq) || 1;
  }
  return { quantized, scales, norms };
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

function rowVectorCopy(flat, row, dimension) {
  const offset = row * dimension;
  const vec = new Float32Array(dimension);
  vec.set(flat.subarray(offset, offset + dimension));
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

function buildQueryVectors(flat, rowCount, dimension, conferenceKey) {
  const queries = [];
  const singleCount = Math.min(SINGLE_QUERY_COUNT, rowCount);
  for (let idx = 0; idx < singleCount; idx += 1) {
    const row = singleCount === 1 ? 0 : Math.floor((idx * (rowCount - 1)) / (singleCount - 1));
    queries.push(rowVectorCopy(flat, row, dimension));
  }

  const randomState = { seed: makeSeed(conferenceKey) };
  for (let query = 0; query < CENTROID_QUERY_COUNT; query += 1) {
    const centroid = new Float32Array(dimension);
    for (let item = 0; item < CENTROID_SIZE; item += 1) {
      const row = Math.floor(nextRandom(randomState) * rowCount);
      const offset = row * dimension;
      for (let dim = 0; dim < dimension; dim += 1) {
        centroid[dim] += flat[offset + dim];
      }
    }
    for (let dim = 0; dim < dimension; dim += 1) {
      centroid[dim] /= CENTROID_SIZE;
    }
    queries.push(normalizeVector(centroid));
  }
  return queries;
}

function dotFloat32(flat, row, query, dimension) {
  let dot = 0;
  const offset = row * dimension;
  for (let dim = 0; dim < dimension; dim += 1) {
    dot += flat[offset + dim] * query[dim];
  }
  return dot;
}

function dotInt8Normalized(quantized, scales, norms, row, query, dimension) {
  let dot = 0;
  const offset = row * dimension;
  const scale = scales[row];
  for (let dim = 0; dim < dimension; dim += 1) {
    dot += (quantized[offset + dim] * scale) * query[dim];
  }
  return dot / norms[row];
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

function evaluateConference(conference) {
  const loaded = loadFloat32Vectors(conference);
  if (!loaded) {
    return null;
  }
  const { flat, rowCount, dimension } = loaded;
  const { quantized, scales, norms } = quantizeSymmetricInt8PerRow(flat, rowCount, dimension);
  const queries = buildQueryVectors(flat, rowCount, dimension, conference.key);

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
      const floatScore = dotFloat32(flat, row, query, dimension);
      const int8Score = dotInt8Normalized(quantized, scales, norms, row, query, dimension);
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

  const rawBytes = flat.byteLength;
  const int8Bytes = quantized.byteLength + scales.byteLength;
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

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function mb(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function main() {
  const manifest = readJson(manifestPath);
  const conferences = Array.isArray(manifest.conferences) ? manifest.conferences : [];
  if (!conferences.length) {
    throw new Error('No conferences found in schedule manifest.');
  }

  const results = conferences.map(evaluateConference).filter(Boolean);
  if (!results.length) {
    if (!fs.existsSync(reportPath)) {
      throw new Error('No float32 semantic baseline found, and no stored int8 quality report exists.');
    }
    const report = readJson(reportPath);
    if (!report.passed) {
      throw new Error('Stored int8 quality report did not pass threshold.');
    }
    console.log(
      `Stored int8 quality report passed: top10=${percent(report.aggregate.avgTop10Overlap)}, ` +
        `top50=${percent(report.aggregate.avgTop50Overlap)}, top1=${percent(report.aggregate.top1Exact)}, ` +
        `mae=${report.aggregate.meanAbsError.toFixed(6)}, max=${report.aggregate.maxAbsError.toFixed(6)}`
    );
    return;
  }
  let totalQueries = 0;
  let totalRowsScored = 0;
  let totalRawBytes = 0;
  let totalInt8Bytes = 0;
  const aggregate = {
    avgTop10Overlap: 0,
    avgTop50Overlap: 0,
    top1Exact: 0,
    top1InTop5: 0,
    meanAbsError: 0,
    maxAbsError: 0,
  };

  for (const result of results) {
    totalQueries += result.queries;
    totalRowsScored += result.queries * result.rowCount;
    totalRawBytes += result.rawBytes;
    totalInt8Bytes += result.int8Bytes;
    aggregate.avgTop10Overlap += result.avgTop10Overlap * result.queries;
    aggregate.avgTop50Overlap += result.avgTop50Overlap * result.queries;
    aggregate.top1Exact += result.top1Exact * result.queries;
    aggregate.top1InTop5 += result.top1InTop5 * result.queries;
    aggregate.meanAbsError += result.meanAbsError * result.queries * result.rowCount;
    aggregate.maxAbsError = Math.max(aggregate.maxAbsError, result.maxAbsError);
  }

  aggregate.avgTop10Overlap /= totalQueries;
  aggregate.avgTop50Overlap /= totalQueries;
  aggregate.top1Exact /= totalQueries;
  aggregate.top1InTop5 /= totalQueries;
  aggregate.meanAbsError /= totalRowsScored;

  for (const result of results) {
    console.log(
      `${result.label || result.key}: rows=${result.rowCount}, queries=${result.queries}, ` +
        `top10=${percent(result.avgTop10Overlap)}, top50=${percent(result.avgTop50Overlap)}, ` +
        `top1=${percent(result.top1Exact)}, top1_in_top5=${percent(result.top1InTop5)}, ` +
        `mae=${result.meanAbsError.toFixed(6)}, max=${result.maxAbsError.toFixed(6)}, ` +
        `${mb(result.rawBytes)} -> ${mb(result.int8Bytes)}`
    );
  }

  console.log(
    `Aggregate: scored=${totalRowsScored.toLocaleString()} row-query pairs, ` +
      `top10=${percent(aggregate.avgTop10Overlap)}, top50=${percent(aggregate.avgTop50Overlap)}, ` +
      `top1=${percent(aggregate.top1Exact)}, top1_in_top5=${percent(aggregate.top1InTop5)}, ` +
      `mae=${aggregate.meanAbsError.toFixed(6)}, max=${aggregate.maxAbsError.toFixed(6)}, ` +
      `${mb(totalRawBytes)} -> ${mb(totalInt8Bytes)}`
  );

  const ok =
    aggregate.avgTop10Overlap >= THRESHOLDS.minAvgTop10Overlap &&
    aggregate.avgTop50Overlap >= THRESHOLDS.minAvgTop50Overlap &&
    aggregate.top1Exact >= THRESHOLDS.minTop1ExactRate &&
    aggregate.top1InTop5 >= THRESHOLDS.minTop1InTop5Rate &&
    aggregate.meanAbsError <= THRESHOLDS.maxMeanAbsError &&
    aggregate.maxAbsError <= THRESHOLDS.maxAbsError;

  if (!ok) {
    throw new Error('Int8 quality check failed threshold.');
  }
  const report = {
    version: 1,
    generated_at: new Date().toISOString(),
    query_plan: {
      single_query_count: SINGLE_QUERY_COUNT,
      centroid_query_count: CENTROID_QUERY_COUNT,
      centroid_size: CENTROID_SIZE,
    },
    thresholds: THRESHOLDS,
    passed: ok,
    aggregate: {
      ...aggregate,
      scoredPairs: totalRowsScored,
      rawBytes: totalRawBytes,
      int8Bytes: totalInt8Bytes,
    },
    conferences: results,
  };
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`Wrote quality report: ${reportPath}`);
  console.log('Int8 quality check passed threshold.');
}

main();
