import { counterFromText } from './text.js';

const SUBSTRING_WORD_RE = /[a-z0-9]+/g;

function buildIdf(docFreq, numDocs) {
  const idf = Object.create(null);
  for (const [term, freq] of Object.entries(docFreq)) {
    idf[term] = Math.log((1 + numDocs) / (1 + freq)) + 1;
  }
  return idf;
}

function vectorizeCounter(counter, idf) {
  const vec = Object.create(null);
  let normSq = 0;
  for (const [term, count] of Object.entries(counter)) {
    const idfWeight = idf[term] || 0;
    const weight = (1 + Math.log(count)) * idfWeight;
    if (weight > 0) {
      vec[term] = weight;
      normSq += weight * weight;
    }
  }
  return { vec, norm: Math.sqrt(normSq) };
}

function vectorizePackedTokenPairs(tokenPairs, idf) {
  const vec = Object.create(null);
  let normSq = 0;
  for (const [term, count] of tokenPairs) {
    const idfWeight = idf[term] || 0;
    const weight = (1 + Math.log(count)) * idfWeight;
    if (weight > 0) {
      vec[term] = weight;
      normSq += weight * weight;
    }
  }
  return { vec, norm: Math.sqrt(normSq) };
}

function cosineSimilarity(leftVec, leftNorm, rightVec, rightNorm) {
  if (!leftNorm || !rightNorm) return 0;
  let a = leftVec;
  let b = rightVec;
  if (Object.keys(a).length > Object.keys(b).length) {
    a = rightVec;
    b = leftVec;
  }
  let dot = 0;
  for (const [term, value] of Object.entries(a)) {
    dot += value * (b[term] || 0);
  }
  return dot / (leftNorm * rightNorm);
}

function topKeywords(workCounters, idf, topN = 15) {
  const accum = Object.create(null);
  for (const counter of workCounters) {
    for (const [term, count] of Object.entries(counter)) {
      accum[term] = (accum[term] || 0) + (1 + Math.log(count)) * (idf[term] || 0);
    }
  }
  return Object.entries(accum)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([term, weight]) => ({ term, weight }));
}

function originalRowIndex(row, fallbackIndex) {
  const parsed = Number(row?.row_index);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallbackIndex;
}

function clampOwnWorkSubstringWordCount(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 6;
  return Math.min(8, Math.max(4, Math.floor(parsed)));
}

function wordsForSubstringMatch(text) {
  return String(text || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .match(SUBSTRING_WORD_RE) || [];
}

function ngramKey(words, startIndex, size) {
  return words.slice(startIndex, startIndex + size).join(' ');
}

function buildOwnWorkSubstringSet(worksTexts, minWords) {
  const ngrams = new Set();
  for (const text of worksTexts) {
    const words = wordsForSubstringMatch(text);
    if (words.length < minWords) continue;
    for (let idx = 0; idx <= words.length - minWords; idx += 1) {
      ngrams.add(ngramKey(words, idx, minWords));
    }
  }
  return ngrams;
}

function ownWorkSubstringMatchForRow(row, ownWorkNgrams, minWords) {
  if (!ownWorkNgrams?.size) return null;
  const rowText = [row?.title, row?.abstract].filter(Boolean).join(' ');
  const rowWords = wordsForSubstringMatch(rowText);
  if (rowWords.length < minWords) return null;

  for (let idx = 0; idx <= rowWords.length - minWords; idx += 1) {
    const phrase = ngramKey(rowWords, idx, minWords);
    if (ownWorkNgrams.has(phrase)) {
      return phrase;
    }
  }
  return null;
}

function normalizeOwnWorkExclusion(ownWorkExclusion) {
  return {
    enabled: Boolean(ownWorkExclusion?.enabled),
    minWords: clampOwnWorkSubstringWordCount(ownWorkExclusion?.minWords),
  };
}

export function scoreSchedule({
  worksTexts,
  scheduleIndex,
  topN = 20,
  workNames = [],
  customKeywords = [],
  ownWorkExclusion = {},
}) {
  const normalizedKeywords = Array.isArray(customKeywords)
    ? customKeywords
        .map((entry) => String(entry || '').trim())
        .filter(Boolean)
    : [];
  const normalizedWorksTexts = Array.isArray(worksTexts)
    ? worksTexts
        .map((text) => String(text || '').trim())
        .filter(Boolean)
    : [];
  const hasWorkTexts = normalizedWorksTexts.length > 0;
  const effectiveWorksTexts = hasWorkTexts ? normalizedWorksTexts : normalizedKeywords.length ? [normalizedKeywords.join(' ')] : [];

  if (!effectiveWorksTexts.length) {
    throw new Error('No papers, abstracts, or keywords provided.');
  }
  if (!scheduleIndex || !Array.isArray(scheduleIndex.rows) || !scheduleIndex.schedule_doc_freq) {
    throw new Error('Invalid schedule index payload.');
  }

  const exclusion = normalizeOwnWorkExclusion(ownWorkExclusion);
  const ownWorkNgrams = exclusion.enabled && normalizedWorksTexts.length
    ? buildOwnWorkSubstringSet(normalizedWorksTexts, exclusion.minWords)
    : null;
  let excludedOwnWorkCount = 0;
  const scheduleRows = [];
  for (const row of scheduleIndex.rows) {
    const matchedPhrase = ownWorkSubstringMatchForRow(row, ownWorkNgrams, exclusion.minWords);
    if (matchedPhrase) {
      excludedOwnWorkCount += 1;
      continue;
    }
    scheduleRows.push(row);
  }
  const workCounters = effectiveWorksTexts.map((text) => counterFromText(text));

  if (hasWorkTexts && normalizedKeywords.length > 0) {
    const keywordCounter = counterFromText(normalizedKeywords.join(' '));
    const keywordBoost = 6;
    for (const workCounter of workCounters) {
      for (const [term, count] of Object.entries(keywordCounter)) {
        workCounter[term] = (workCounter[term] || 0) + count * keywordBoost;
      }
    }
  }

  const docFreq = { ...scheduleIndex.schedule_doc_freq };
  for (const workCounter of workCounters) {
    for (const term of Object.keys(workCounter)) {
      docFreq[term] = (docFreq[term] || 0) + 1;
    }
  }

  const numDocs = scheduleRows.length + workCounters.length;
  const idf = buildIdf(docFreq, numDocs);

  const workVectors = workCounters.map((counter) => vectorizeCounter(counter, idf));

  const centroid = Object.create(null);
  for (const { vec } of workVectors) {
    for (const [term, weight] of Object.entries(vec)) {
      centroid[term] = (centroid[term] || 0) + weight;
    }
  }
  const workCount = workVectors.length || 1;
  let centroidNormSq = 0;
  for (const term of Object.keys(centroid)) {
    centroid[term] /= workCount;
    centroidNormSq += centroid[term] * centroid[term];
  }
  const centroidNorm = Math.sqrt(centroidNormSq);

  const rows = [];
  for (const [rowIndex, row] of scheduleRows.entries()) {
    const { vec, norm } = vectorizePackedTokenPairs(row.tokens || [], idf);
    const sourceRowIndex = originalRowIndex(row, rowIndex);

    const centroidSim = cosineSimilarity(vec, norm, centroid, centroidNorm);
    let maxWorkSim = 0;
    for (const workVec of workVectors) {
      const sim = cosineSimilarity(vec, norm, workVec.vec, workVec.norm);
      if (sim > maxWorkSim) {
        maxWorkSim = sim;
      }
    }

    const score = 100 * (0.7 * centroidSim + 0.3 * maxWorkSim);
    rows.push({
      row_index: sourceRowIndex,
      conference_key: row.conference_key || '',
      conference_id: row.conference_id || '',
      conference_short_name: row.conference_short_name || '',
      conference_year: row.conference_year || '',
      conference_label: row.conference_label || '',
      conference_timezone: row.conference_timezone || '',
      title: row.title || '',
      authors: row.authors || '',
      abstract: row.abstract || '',
      room: row.room || '',
      building: row.building || '',
      day: row.day || '',
      session_type: row.session_type || '',
      start_date_unix_ms: row.start_date_unix_ms ?? null,
      end_date_unix_ms: row.end_date_unix_ms ?? null,
      relevance_mode: 'tfidf',
      relevance_score_tfidf: score,
      relevance_score_tfidf_pretty: score.toFixed(2),
      relevance_score: score,
      relevance_score_pretty: score.toFixed(2),
      relevance_centroid_sim: centroidSim,
      relevance_max_work_sim: maxWorkSim,
    });
  }

  const rowsAll = rows.map((row) => ({ ...row }));
  rows.sort((a, b) => b.relevance_score - a.relevance_score);
  rows.forEach((row, idx) => {
    row.relevance_rank = idx + 1;
  });

  const normalizedTopN = Number.isFinite(topN) ? Math.max(1, Math.floor(topN)) : rows.length;
  const resultRows = rows.slice(0, Math.min(normalizedTopN, rows.length));

  return {
    totalMatches: rows.length,
    excluded_own_work_count: excludedOwnWorkCount,
    excluded_own_work_min_words: exclusion.enabled ? exclusion.minWords : null,
    rows_all: rowsAll,
    rows: resultRows,
    keywords: topKeywords(workCounters, idf, 15),
    workSummaries: effectiveWorksTexts.map((text, idx) => ({
      name: workNames[idx] || (!hasWorkTexts ? 'Keywords' : `Work ${idx + 1}`),
      chars: text.length,
      token_count: Object.values(workCounters[idx]).reduce((acc, count) => acc + count, 0),
      unique_terms: Object.keys(workCounters[idx]).length,
    })),
  };
}
