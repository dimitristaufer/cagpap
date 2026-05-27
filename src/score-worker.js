import { scoreSchedule } from './shared/scoring.js';
import {
  buildScheduleSemanticTexts,
  buildWorkChunkPlan,
  embedTexts,
  preloadSemanticModel,
  reduceChunkVectorsToWorkVectors,
} from './shared/semantic.js';
import {
  baseUrlPath,
  SEMANTIC_LOCAL_MODEL_ROOT,
  SEMANTIC_MODEL_DTYPE,
  SEMANTIC_MODEL_ID,
  SEMANTIC_MODEL_SUBFOLDER,
} from './shared/semantic-config.js';

let scheduleIndex = null;
let semanticScheduleCache = null;

const HYBRID_SEMANTIC_WEIGHT = 0.55;
const VALID_MATCHING_MODES = new Set(['tfidf', 'semantic', 'hybrid']);
const SEMANTIC_OPTIONS = {
  modelId: SEMANTIC_MODEL_ID,
  localModelRoot: SEMANTIC_LOCAL_MODEL_ROOT,
  allowRemoteModels: false,
  dtype: SEMANTIC_MODEL_DTYPE,
  subfolder: SEMANTIC_MODEL_SUBFOLDER,
};

const SEMANTIC_PREFETCH_OPTIONS = {
  ...SEMANTIC_OPTIONS,
  allowRemoteModels: true,
};

function normalizeMatchingMode(rawMode) {
  const normalized = String(rawMode || 'tfidf')
    .trim()
    .toLowerCase();
  return VALID_MATCHING_MODES.has(normalized) ? normalized : 'tfidf';
}

function normalizeConferenceKey(rawKey) {
  return String(rawKey || '')
    .trim()
    .toLowerCase();
}

function scheduleDocFreqForRows(rows) {
  const docFreq = Object.create(null);
  for (const row of rows) {
    const seen = new Set();
    for (const [term] of row?.tokens || []) {
      if (!term || seen.has(term)) continue;
      seen.add(term);
      docFreq[term] = (docFreq[term] || 0) + 1;
    }
  }
  return docFreq;
}

function scheduleIndexForScope() {
  const allRows = Array.isArray(scheduleIndex?.rows) ? scheduleIndex.rows : [];
  return {
    ...scheduleIndex,
    rows: allRows.map((row, idx) => ({ ...row, row_index: idx })),
    row_count: allRows.length,
    schedule_doc_freq: scheduleIndex?.schedule_doc_freq || scheduleDocFreqForRows(allRows),
  };
}

function postProgress(requestId, message, extra = {}) {
  self.postMessage({
    type: 'progress',
    requestId,
    message,
    ...extra,
  });
}

function modelLoadProgressMessage(info) {
  const status = String(info?.status || '');
  if (status === 'progress_total') {
    const percent = Number(info.progress);
    if (Number.isFinite(percent)) {
      return `Downloading semantic model (${Math.round(percent)}%)`;
    }
    return 'Downloading semantic model...';
  }
  if (status === 'download' && info.file) {
    return `Downloading model file: ${info.file}`;
  }
  if (status === 'initiate' && info.file) {
    return `Preparing model file: ${info.file}`;
  }
  if (status === 'done' && info.file) {
    return `Cached model file: ${info.file}`;
  }
  if (status === 'ready') {
    return 'Semantic model ready.';
  }
  return 'Preparing semantic model...';
}

function toFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function relayModelLoadProgress(requestId, info) {
  const total = toFiniteNumber(info?.total);
  const loaded = toFiniteNumber(info?.loaded);
  let percent = toFiniteNumber(info?.progress);
  if (percent == null && total && loaded != null) {
    percent = (loaded / total) * 100;
  }
  if (percent != null) {
    percent = Math.max(0, Math.min(100, percent));
  }

  postProgress(requestId, modelLoadProgressMessage(info), {
    stage: 'model_download',
    rawStatus: info?.status || '',
    file: info?.file || '',
    loaded: loaded ?? undefined,
    total: total ?? undefined,
    percent: percent ?? undefined,
  });
}

function createBatchProgressReporter(requestId, label) {
  let lastPercent = -1;
  return ({ processed, total }) => {
    if (!total) return;
    const percent = Math.max(0, Math.min(100, Math.round((processed / total) * 100)));
    if (percent === lastPercent) return;
    if (percent !== 100 && percent - lastPercent < 3) return;
    lastPercent = percent;
    postProgress(requestId, `${label} (${percent}%)`, {
      processed,
      total,
      percent,
    });
  };
}

function dotProduct(left, right) {
  let dot = 0;
  const len = Math.min(left.length, right.length);
  for (let i = 0; i < len; i += 1) {
    dot += left[i] * right[i];
  }
  return dot;
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

function centroidFromWorkVectors(workVectors) {
  if (!workVectors.length) {
    return new Float32Array(384);
  }
  const dim = workVectors[0]?.length || 384;
  const centroid = new Float32Array(dim);
  for (const vector of workVectors) {
    for (let i = 0; i < dim; i += 1) {
      centroid[i] += vector[i];
    }
  }
  const inv = 1 / workVectors.length;
  let normSq = 0;
  for (let i = 0; i < dim; i += 1) {
    centroid[i] *= inv;
    normSq += centroid[i] * centroid[i];
  }
  const norm = Math.sqrt(normSq);
  if (!norm) return centroid;
  for (let i = 0; i < dim; i += 1) {
    centroid[i] /= norm;
  }
  return centroid;
}

function computeSemanticScores(scheduleVectors, workVectors) {
  const centroid = centroidFromWorkVectors(workVectors);
  const scoresByIndex = new Array(scheduleVectors.length);

  for (let rowIndex = 0; rowIndex < scheduleVectors.length; rowIndex += 1) {
    const scheduleVec = scheduleVectors[rowIndex];
    const centroidSim = dotProduct(scheduleVec, centroid);

    let maxWorkSim = 0;
    for (const workVec of workVectors) {
      const sim = dotProduct(scheduleVec, workVec);
      if (sim > maxWorkSim) {
        maxWorkSim = sim;
      }
    }

    const semanticScore = 100 * (0.7 * centroidSim + 0.3 * maxWorkSim);
    scoresByIndex[rowIndex] = {
      semanticScore,
      centroidSim,
      maxWorkSim,
    };
  }

  return scoresByIndex;
}

function normalizeTopN(value, maxRows) {
  if (Number.isFinite(value)) {
    return Math.max(1, Math.min(maxRows, Math.floor(value)));
  }
  return maxRows;
}

function rankRows(rows, topN) {
  rows.sort((a, b) => b.relevance_score - a.relevance_score);
  rows.forEach((row, idx) => {
    row.relevance_rank = idx + 1;
  });
  return rows.slice(0, normalizeTopN(topN, rows.length));
}

function scheduleAssetUrl(urlPath) {
  const raw = String(urlPath || '').trim();
  if (!raw) return '';
  if (/^https?:\/\//i.test(raw) || raw.startsWith('/')) return raw;
  return baseUrlPath(raw);
}

async function fetchPrecomputedVectorShard({ requestId, metaUrl, binUrl, expectedRows, label }) {
  const [metaResp, binResp] = await Promise.all([
    fetch(scheduleAssetUrl(metaUrl)),
    fetch(scheduleAssetUrl(binUrl)),
  ]);
  if (!metaResp.ok) {
    throw new Error(`Missing precomputed semantic metadata for ${label || 'schedule shard'} (${metaResp.status}).`);
  }
  if (!binResp.ok) {
    throw new Error(`Missing precomputed semantic vectors for ${label || 'schedule shard'} (${binResp.status}).`);
  }

  const meta = await metaResp.json();
  const rowCount = Number(meta?.row_count);
  const dimension = Number(meta?.dimension);
  if (!Number.isInteger(rowCount) || !Number.isInteger(dimension) || rowCount <= 0 || dimension <= 0) {
    throw new Error('Invalid precomputed semantic metadata.');
  }
  if (Number.isInteger(expectedRows) && rowCount !== expectedRows) {
    throw new Error(`Precomputed semantic row_count mismatch for ${label || 'schedule shard'} (${rowCount} vs ${expectedRows}).`);
  }

  const arrayBuffer = await binResp.arrayBuffer();
  const format = meta?.format || 'float32le';
  const vectors = new Array(rowCount);

  if (format === 'float32le') {
    if (arrayBuffer.byteLength % 4 !== 0) {
      throw new Error('Invalid precomputed semantic vector binary format.');
    }
    const flat = new Float32Array(arrayBuffer);
    if (flat.length !== rowCount * dimension) {
      throw new Error(
        `Precomputed semantic vector length mismatch (${flat.length} vs expected ${rowCount * dimension}).`
      );
    }

    for (let idx = 0; idx < rowCount; idx += 1) {
      const start = idx * dimension;
      vectors[idx] = flat.subarray(start, start + dimension);
    }
  } else if (format === 'int8_symmetric_per_row') {
    const expectedScaleBytes = rowCount * 4;
    const expectedVectorBytes = rowCount * dimension;
    const expectedBytes = expectedScaleBytes + expectedVectorBytes;
    if (arrayBuffer.byteLength !== expectedBytes) {
      throw new Error(
        `Precomputed int8 semantic vector length mismatch (${arrayBuffer.byteLength} vs expected ${expectedBytes}).`
      );
    }

    const scales = new Float32Array(arrayBuffer, 0, rowCount);
    const quantized = new Int8Array(arrayBuffer, expectedScaleBytes, expectedVectorBytes);
    for (let row = 0; row < rowCount; row += 1) {
      const scale = Number(scales[row]);
      if (!Number.isFinite(scale) || scale <= 0) {
        throw new Error(`Invalid int8 semantic scale for row ${row}.`);
      }
      const vector = new Float32Array(dimension);
      const offset = row * dimension;
      for (let dim = 0; dim < dimension; dim += 1) {
        vector[dim] = quantized[offset + dim] * scale;
      }
      vectors[row] = normalizeVector(vector);
    }
  } else {
    throw new Error(`Unsupported precomputed semantic vector format '${format}'.`);
  }

  return {
    vectors,
    dimension,
    model: {
      modelId: meta.model_id || SEMANTIC_MODEL_ID,
      dtype: meta.dtype || SEMANTIC_MODEL_DTYPE,
      source: 'precomputed',
      format,
      generated_at: meta.generated_at || null,
    },
  };
}

async function loadPrecomputedScheduleVectors(requestId) {
  postProgress(requestId, 'Loading precomputed semantic schedule embeddings...');
  const shards = Array.isArray(scheduleIndex?.semantic_embedding_shards) && scheduleIndex.semantic_embedding_shards.length
    ? scheduleIndex.semantic_embedding_shards
    : [
        {
          key: scheduleIndex?.shard_key || 'schedule',
          row_count: scheduleIndex?.rows?.length || 0,
          meta_url: scheduleIndex?.semantic_embeddings_meta_url,
          bin_url: scheduleIndex?.semantic_embeddings_bin_url,
        },
      ];

  const vectors = [];
  let dimension = 0;
  let model = null;

  for (const shard of shards) {
    if (!shard?.meta_url || !shard?.bin_url) {
      throw new Error(`Missing precomputed semantic embedding URLs for ${shard?.key || 'schedule shard'}.`);
    }
    const shardResult = await fetchPrecomputedVectorShard({
      requestId,
      metaUrl: shard.meta_url,
      binUrl: shard.bin_url,
      expectedRows: Number(shard.row_count),
      label: shard.label || shard.key,
    });
    if (!dimension) {
      dimension = shardResult.dimension;
    } else if (dimension !== shardResult.dimension) {
      throw new Error(`Precomputed semantic dimension mismatch (${dimension} vs ${shardResult.dimension}).`);
    }
    for (const vector of shardResult.vectors) {
      vectors.push(vector);
    }
    model = model || shardResult.model;
  }

  const expectedRows = scheduleIndex?.rows?.length || 0;
  if (vectors.length !== expectedRows) {
    throw new Error(`Precomputed semantic row_count mismatch (${vectors.length} vs ${expectedRows}).`);
  }

  return {
    vectors,
    model: model || {
      modelId: SEMANTIC_MODEL_ID,
      dtype: SEMANTIC_MODEL_DTYPE,
      source: 'precomputed',
      generated_at: null,
    },
  };
}

function semanticScheduleCacheKeyForIndex(index) {
  const rowCount = index?.rows?.length || 0;
  const shards = Array.isArray(index?.semantic_embedding_shards) ? index.semantic_embedding_shards : [];
  if (shards.length) {
    return `${rowCount}:${shards
      .map((shard) => `${shard.key || ''}:${shard.row_count || 0}:${shard.meta_url || ''}:${shard.bin_url || ''}`)
      .join('|')}`;
  }
  return `${rowCount}:${index?.semantic_embeddings_meta_url || ''}:${index?.semantic_embeddings_bin_url || ''}`;
}

function semanticScheduleCacheKey() {
  return semanticScheduleCacheKeyForIndex(scheduleIndex);
}

async function ensureSemanticScheduleVectors(requestId) {
  const rowCount = scheduleIndex?.rows?.length || 0;
  const cacheKey = semanticScheduleCacheKey();
  if (semanticScheduleCache && semanticScheduleCache.rowCount === rowCount && semanticScheduleCache.cacheKey === cacheKey) {
    return semanticScheduleCache;
  }

  try {
    const precomputed = await loadPrecomputedScheduleVectors(requestId);
    semanticScheduleCache = {
      cacheKey,
      rowCount,
      vectors: precomputed.vectors,
      model: precomputed.model,
    };
    postProgress(requestId, 'Loaded precomputed semantic schedule index.', {
      stage: 'semantic_schedule_precomputed',
    });
    return semanticScheduleCache;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    postProgress(requestId, 'Precomputed semantic index unavailable. Building semantic schedule index locally...', {
      stage: 'semantic_schedule_fallback',
      reason: message,
    });
  }

  postProgress(requestId, 'Loading local semantic model...');
  const model = await preloadSemanticModel({
    ...SEMANTIC_OPTIONS,
    progressCallback: (info) => relayModelLoadProgress(requestId, info),
  });
  const scheduleTexts = buildScheduleSemanticTexts(scheduleIndex.rows || []);
  postProgress(requestId, 'Building semantic vectors for conference schedule...');

  const { vectors, model: embedModel } = await embedTexts(scheduleTexts, {
    ...SEMANTIC_OPTIONS,
    onBatch: createBatchProgressReporter(requestId, 'Building local semantic schedule index'),
  });

  semanticScheduleCache = {
    cacheKey,
    rowCount,
    vectors,
    model: embedModel || model?.model || null,
  };
  postProgress(requestId, 'Semantic schedule index ready.');
  return semanticScheduleCache;
}

function buildCombinedRows(tfidfResult, semanticScoresByIndex, matchingMode, topN) {
  const baseRows = Array.isArray(tfidfResult.rows_all) ? tfidfResult.rows_all : [];
  const rows = baseRows.map((row) => {
    const semantic = semanticScoresByIndex[row.row_index] || {
      semanticScore: 0,
      centroidSim: 0,
      maxWorkSim: 0,
    };
    const tfidfScore = row.relevance_score_tfidf ?? row.relevance_score ?? 0;
    const semanticScore = semantic.semanticScore;

    let finalScore = tfidfScore;
    if (matchingMode === 'semantic') {
      finalScore = semanticScore;
    } else if (matchingMode === 'hybrid') {
      finalScore = HYBRID_SEMANTIC_WEIGHT * semanticScore + (1 - HYBRID_SEMANTIC_WEIGHT) * tfidfScore;
    }

    return {
      ...row,
      relevance_mode: matchingMode,
      relevance_score_tfidf: tfidfScore,
      relevance_score_tfidf_pretty: tfidfScore.toFixed(2),
      relevance_score_semantic: semanticScore,
      relevance_score_semantic_pretty: semanticScore.toFixed(2),
      relevance_semantic_centroid_sim: semantic.centroidSim,
      relevance_semantic_max_work_sim: semantic.maxWorkSim,
      relevance_score: finalScore,
      relevance_score_pretty: finalScore.toFixed(2),
    };
  });

  return {
    totalMatches: rows.length,
    excluded_own_work_count: Number(tfidfResult.excluded_own_work_count || 0),
    excluded_own_work_min_words: tfidfResult.excluded_own_work_min_words ?? null,
    rows: rankRows(rows, topN),
    keywords: tfidfResult.keywords || [],
    workSummaries: tfidfResult.workSummaries || [],
  };
}

self.onmessage = async (event) => {
  const { type, requestId } = event.data || {};

  if (type === 'init') {
    const nextScheduleIndex = event.data.scheduleIndex;
    const currentCacheKey = scheduleIndex ? semanticScheduleCacheKeyForIndex(scheduleIndex) : '';
    const nextCacheKey = semanticScheduleCacheKeyForIndex(nextScheduleIndex);
    scheduleIndex = nextScheduleIndex;
    if (currentCacheKey !== nextCacheKey) {
      semanticScheduleCache = null;
    }
    self.postMessage({ type: 'ready', quiet: Boolean(event.data.quiet), rowCount: scheduleIndex?.row_count || 0 });
    return;
  }

  if (type === 'run') {
    if (!scheduleIndex) {
      self.postMessage({
        type: 'error',
        requestId,
        error: 'Worker not initialized with schedule index.',
      });
      return;
    }

    try {
      const started = performance.now();
      const matchingMode = normalizeMatchingMode(event.data.matchingMode);
      const activeScheduleIndex = scheduleIndexForScope();

      if (!activeScheduleIndex.rows.length) {
        throw new Error('No schedule rows found for the selected conference scope.');
      }

      const tfidfResult = scoreSchedule({
        worksTexts: event.data.worksTexts || [],
        workNames: event.data.workNames || [],
        topN: event.data.topN,
        customKeywords: event.data.customKeywords || [],
        scheduleIndex: activeScheduleIndex,
        ownWorkExclusion: event.data.ownWorkExclusion || {},
      });

      let result = {
        ...tfidfResult,
      };
      delete result.rows_all;
      if (matchingMode === 'semantic' || matchingMode === 'hybrid') {
        const semanticSchedule = await ensureSemanticScheduleVectors(requestId);
        const workChunkPlan = buildWorkChunkPlan(event.data.worksTexts || [], event.data.customKeywords || []);

        postProgress(requestId, 'Embedding uploaded papers...');
        const { vectors: workChunkVectors, model } = await embedTexts(workChunkPlan.chunks, {
          ...SEMANTIC_OPTIONS,
          onBatch: createBatchProgressReporter(requestId, 'Embedding uploaded papers'),
        });

        const workVectors = reduceChunkVectorsToWorkVectors(workChunkVectors, workChunkPlan.spans);
        const semanticScoresByIndex = computeSemanticScores(semanticSchedule.vectors, workVectors);
        result = buildCombinedRows(tfidfResult, semanticScoresByIndex, matchingMode, event.data.topN);
        result.semantic_model = model || semanticSchedule.model || null;
      }

      result.matching_mode = matchingMode;
      const elapsedMs = performance.now() - started;
      self.postMessage({ type: 'result', requestId, result, elapsedMs });
    } catch (error) {
      self.postMessage({
        type: 'error',
        requestId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  if (type === 'prefetch-model') {
    try {
      const started = performance.now();
      postProgress(requestId, 'Preparing semantic model download...', { stage: 'model_download' });
      const loaded = await preloadSemanticModel({
        ...SEMANTIC_PREFETCH_OPTIONS,
        progressCallback: (info) => relayModelLoadProgress(requestId, info),
      });
      const elapsedMs = performance.now() - started;
      self.postMessage({
        type: 'result',
        requestId,
        command: 'prefetch-model',
        result: {
          ok: true,
          semantic_model: loaded.model || null,
        },
        elapsedMs,
      });
    } catch (error) {
      self.postMessage({
        type: 'error',
        requestId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  if (type === 'probe-model') {
    try {
      const loaded = await preloadSemanticModel({
        ...SEMANTIC_OPTIONS,
      });
      self.postMessage({
        type: 'result',
        requestId,
        command: 'probe-model',
        result: {
          ok: true,
          available: true,
          semantic_model: loaded.model || null,
        },
      });
    } catch (error) {
      self.postMessage({
        type: 'result',
        requestId,
        command: 'probe-model',
        result: {
          ok: true,
          available: false,
          error: error instanceof Error ? error.message : String(error),
        },
      });
    }
  }

};
