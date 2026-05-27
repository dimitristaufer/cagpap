import './styles.css';
import * as pdfjsLib from 'pdfjs-dist/build/pdf.mjs';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url';
import { baseUrlPath, semanticModelUrl } from './shared/semantic-config.js';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const MAX_PROFILE_ABSTRACTS = 20;
const PROFILE_ABSTRACT_PREVIEW_WORDS = 20;
const EXTRACTION_CONCURRENCY = 2;
const DEFAULT_VISIBLE_RESULTS = 20;
const LOAD_MORE_STEP = 20;
const KEYWORD_DEBOUNCE_MS = 350;
const APP_NAME = 'CAGPAP';
const CALENDAR_DOWNLOADS_STORAGE_KEY = 'cagpap_downloaded_calendar_items_v1';
const MATCHING_MODES = new Set(['tfidf', 'semantic', 'hybrid']);
const ICS_PROD_ID = '-//CAGPAP//EN';
const FALLBACK_EVENT_DURATION_MS = 30 * 60 * 1000;
const DEFAULT_CONFERENCE_TIMEZONE = 'Europe/Madrid';
const SCHEDULE_WALL_CLOCK_TIMEZONE = 'UTC';
const THREE_MONTHS_MS = 90 * 24 * 60 * 60 * 1000;
const FIXED_TIME_SCOPE_KEY = 'all';
const TIME_SCOPE_OPTIONS = new Map([
  [
    'all',
    {
      label: 'anytime',
      searchLabel: 'past + upcoming conferences',
    },
  ],
  [
    'upcoming',
    {
      label: 'upcoming',
      searchLabel: 'upcoming conferences',
    },
  ],
  [
    'past',
    {
      label: 'past',
      searchLabel: 'past conferences',
    },
  ],
  [
    'upcoming-3m',
    {
      label: '> 3 months',
      searchLabel: 'conferences more than 3 months out',
    },
  ],
]);
const DAY_PILL_CLASS_MAP = {
  monday: 'day-pill-monday',
  tuesday: 'day-pill-tuesday',
  wednesday: 'day-pill-wednesday',
  thursday: 'day-pill-thursday',
  friday: 'day-pill-friday',
  saturday: 'day-pill-saturday',
  sunday: 'day-pill-sunday',
};
const startDateFormatter = new Intl.DateTimeFormat(undefined, {
  weekday: 'short',
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  timeZone: SCHEDULE_WALL_CLOCK_TIMEZONE,
});
const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  minute: '2-digit',
  timeZone: SCHEDULE_WALL_CLOCK_TIMEZONE,
});

const form = document.getElementById('analyze-form');
const timeScopeSelect = document.getElementById('time-scope-select');
const conferenceSelect = document.getElementById('conference-select');
const searchAllConferencesToggle = document.getElementById('search-all-conferences');
const searchAllConferencesLabel = document.getElementById('search-all-conferences-label');
const excludeSubstringMatchesInput = document.getElementById('exclude-substring-matches');
const substringMatchWordCountSelect = document.getElementById('substring-match-word-count');
const fileInput = document.getElementById('pdfs');
const fileSelection = document.getElementById('file-selection');
const clearPdfsBtn = document.getElementById('clear-pdfs-btn');
const sourceTabButtons = Array.from(document.querySelectorAll('[data-source-tab]'));
const sourcePanels = Array.from(document.querySelectorAll('[data-source-panel]'));
const profileAuthorQueryInput = document.getElementById('profile-author-query');
const fetchProfileBtn = document.getElementById('fetch-profile-btn');
const profileFetchStatus = document.getElementById('profile-fetch-status');
const profileAuthorCandidatesPanel = document.getElementById('profile-author-candidates-panel');
const profileAuthorCandidatesList = document.getElementById('profile-author-candidates-list');
const profileAuthorCandidatesCount = document.getElementById('profile-author-candidates-count');
const profileAbstractsPanel = document.getElementById('profile-abstracts-panel');
const profileAbstractsList = document.getElementById('profile-abstracts-list');
const profileAbstractsCount = document.getElementById('profile-abstracts-count');
const selectAllAbstractsBtn = document.getElementById('select-all-abstracts-btn');
const toggleAbstractPreviewBtn = document.getElementById('toggle-abstract-preview-btn');
const clearAllAbstractsBtn = document.getElementById('clear-all-abstracts-btn');
const keywordInput = document.getElementById('custom-keywords');
const matchingModeInput = document.getElementById('matching-mode');
const downloadModelBtn = document.getElementById('download-model-btn');
const modelDownloadStatus = document.getElementById('model-download-status');
const modelDownloadProgressWrap = document.getElementById('model-download-progress-wrap');
const modelDownloadProgress = document.getElementById('model-download-progress');
const modelDownloadProgressText = document.getElementById('model-download-progress-text');
const runSelectedCount = document.getElementById('run-selected-count');
const boostedKeywordsList = document.getElementById('boosted-keywords-list');
const runBtn = document.getElementById('run-btn');
const loadMoreBtn = document.getElementById('load-more-btn');
const statusLine = document.getElementById('status-line');
const statusList = document.getElementById('status-list');
const resultsPanel = document.getElementById('results-panel');
const resultsActions = document.getElementById('results-actions');

const scoreWorker = new Worker(new URL('./score-worker.js', import.meta.url), { type: 'module' });

let requestSeq = 0;
const pendingRequests = new Map();

let scheduleIndexPayload = null;
let scheduleRowCount = 0;
let selectedConferenceKey = '';
let searchAcrossAllConferences = false;
let availableConferences = [];
let conferenceSelectShowingCounts = false;
const scheduleIndexCache = new Map();
let selectedTimeScopeKey = FIXED_TIME_SCOPE_KEY;
let selectedField = 'CS';
let extractedContext = null;
let latestRows = [];
let visibleResults = DEFAULT_VISIBLE_RESULTS;
let scoringRunSeq = 0;
let keywordDebounceHandle = null;
let isExtracting = false;
let latestMatchingMode = 'semantic';
let fetchedProfileAbstracts = [];
let fetchedAuthorCandidates = [];
let selectedAuthorCandidateId = '';
let activeAuthorLookupKey = '';
let profileFetchRunSeq = 0;
let profileAbstractsExpanded = false;
const selectedProfileAbstractIds = new Set();
let modelDownloadRequestId = null;
let modelDownloadBusy = false;
let semanticModelAvailable = false;
let modelAvailabilityRefreshSeq = 0;
const semanticFallbackNotedRequests = new Set();
const downloadedCalendarItems = loadDownloadedCalendarItems();
const authorCandidatesCache = new Map();
const authorAbstractsCache = new Map();

scoreWorker.onmessage = (event) => {
  const payload = event.data || {};

  if (payload.type === 'ready') {
    if (payload.quiet) return;
    const count = updateActiveScheduleRowCount();
    setStatus(`Schedule index ready for ${scoringScopeLabel()} (${count.toLocaleString()} rows).`);
    return;
  }

  if (payload.type === 'progress') {
    const pending = pendingRequests.get(payload.requestId);
    if (pending && payload.message) {
      setStatus(payload.message);
    }

    if (payload.stage === 'semantic_schedule_fallback' && payload.requestId != null) {
      if (!semanticFallbackNotedRequests.has(payload.requestId)) {
        semanticFallbackNotedRequests.add(payload.requestId);
        const detail = payload.reason ? ` (${payload.reason})` : '';
        appendStatusItem(`Precomputed semantic embeddings were unavailable; using local fallback${detail}`);
      }
    }

    if (payload.requestId === modelDownloadRequestId && payload.stage === 'model_download') {
      setModelDownloadUi({
        busy: true,
        statusText: payload.message || 'Downloading semantic model...',
        progressPercent: Number.isFinite(payload.percent) ? payload.percent : null,
        loadedBytes: payload.loaded,
        totalBytes: payload.total,
      });
    }
    return;
  }

  if (payload.type === 'result' || payload.type === 'error') {
    let requestId = payload.requestId;
    if (requestId == null && pendingRequests.size === 1) {
      requestId = pendingRequests.keys().next().value;
    }
    const pending = pendingRequests.get(requestId);
    if (!pending) return;
    pendingRequests.delete(requestId);

    if (payload.type === 'error') {
      pending.reject(new Error(payload.error || 'Unknown worker error'));
    } else {
      pending.resolve(payload);
    }
  }
};

function setStatus(text) {
  statusLine.textContent = text;
}

function appendStatusItem(text) {
  const li = document.createElement('li');
  li.textContent = text;
  statusList.appendChild(li);
  return li;
}

function clearResults() {
  resultsPanel.innerHTML = '';
  resultsActions.classList.add('hidden');
  latestRows = [];
  visibleResults = DEFAULT_VISIBLE_RESULTS;
}

function clearStatusItems() {
  statusList.innerHTML = '';
}

function parsePixelValue(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function selectedOptionText(select) {
  return select.options[select.selectedIndex]?.textContent || select.value || '';
}

function syncSelectContentWidth(select) {
  if (!select) return;

  const style = window.getComputedStyle(select);
  const mirror = document.createElement('span');
  mirror.textContent = selectedOptionText(select);
  mirror.style.position = 'fixed';
  mirror.style.left = '-9999px';
  mirror.style.top = '0';
  mirror.style.visibility = 'hidden';
  mirror.style.whiteSpace = 'pre';
  mirror.style.font = style.font;
  mirror.style.letterSpacing = style.letterSpacing;
  mirror.style.textTransform = style.textTransform;
  document.body.appendChild(mirror);

  const textWidth = mirror.getBoundingClientRect().width;
  mirror.remove();

  const width =
    Math.ceil(textWidth) +
    parsePixelValue(style.paddingLeft) +
    parsePixelValue(style.paddingRight) +
    parsePixelValue(style.borderLeftWidth) +
    parsePixelValue(style.borderRightWidth) +
    2;
  select.style.width = `${Math.ceil(width)}px`;
}

function syncTitleSelectWidths() {
  syncSelectContentWidth(timeScopeSelect);
  syncSelectContentWidth(conferenceSelect);
}

function setActiveSourceTab(tabName, { focus = false } = {}) {
  if (!sourceTabButtons.length || !sourcePanels.length) return;

  let matchedTab = false;
  for (const button of sourceTabButtons) {
    const isActive = button.dataset.sourceTab === tabName;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    button.setAttribute('tabindex', isActive ? '0' : '-1');
    if (isActive) {
      matchedTab = true;
      if (focus) {
        button.focus();
      }
    }
  }

  if (!matchedTab) {
    return;
  }

  for (const panel of sourcePanels) {
    const isActive = panel.dataset.sourcePanel === tabName;
    panel.classList.toggle('is-active', isActive);
    panel.classList.toggle('hidden', !isActive);
  }
}

function initializeSourceTabs() {
  if (!sourceTabButtons.length || !sourcePanels.length) return;

  const activeButton =
    sourceTabButtons.find((button) => button.classList.contains('is-active')) || sourceTabButtons[0];

  if (activeButton?.dataset?.sourceTab) {
    setActiveSourceTab(activeButton.dataset.sourceTab);
  }

  for (const [index, button] of sourceTabButtons.entries()) {
    button.addEventListener('click', () => {
      const targetTab = button.dataset.sourceTab;
      if (!targetTab) return;
      setActiveSourceTab(targetTab);
    });

    button.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        return;
      }

      event.preventDefault();
      const lastIndex = sourceTabButtons.length - 1;
      let nextIndex = index;

      if (event.key === 'Home') {
        nextIndex = 0;
      } else if (event.key === 'End') {
        nextIndex = lastIndex;
      } else if (event.key === 'ArrowRight') {
        nextIndex = index >= lastIndex ? 0 : index + 1;
      } else if (event.key === 'ArrowLeft') {
        nextIndex = index <= 0 ? lastIndex : index - 1;
      }

      const nextTab = sourceTabButtons[nextIndex]?.dataset?.sourceTab;
      if (nextTab) {
        setActiveSourceTab(nextTab, { focus: true });
      }
    });
  }
}

async function loadScheduleIndex() {
  const resp = await fetch(baseUrlPath('data/schedule_manifest.json'));
  if (!resp.ok) {
    throw new Error(`Could not load schedule manifest (${resp.status}).`);
  }
  return resp.json();
}

function normalizeConferenceKey(value) {
  return String(value || '')
    .trim()
    .toLowerCase();
}

function normalizeConferenceField(value) {
  return compactWhitespace(value || 'CS').toUpperCase();
}

function toFiniteEpochMs(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function selectedTimeScope() {
  return TIME_SCOPE_OPTIONS.get(selectedTimeScopeKey) || TIME_SCOPE_OPTIONS.get('all');
}

function conferenceKeyFromParts(shortName, year) {
  const normalizedShortName = String(shortName || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  const normalizedYear = String(year || '').replace(/[^0-9]+/g, '');
  if (normalizedShortName && normalizedYear) return `${normalizedShortName}-${normalizedYear}`;
  return normalizedShortName || normalizedYear || 'unknown-conference';
}

function conferenceLabelFromParts(shortName, year) {
  const short = compactWhitespace(shortName).toUpperCase();
  const normalizedYear = compactWhitespace(year);
  if (short && normalizedYear) return `${short} ${normalizedYear}`;
  return short || normalizedYear || 'Unknown Conference';
}

function normalizeConferenceRecord(record = {}) {
  const shortName = compactWhitespace(record.short_name || record.conference_short_name).toUpperCase();
  const year = compactWhitespace(record.year || record.conference_year);
  const key = normalizeConferenceKey(record.key || record.conference_key || conferenceKeyFromParts(shortName, year));
  return {
    key,
    id: compactWhitespace(record.id || record.conference_id),
    shortName,
    year,
    field: normalizeConferenceField(record.field || record.conference_field || 'CS'),
    timezone: compactWhitespace(record.timezone || record.conference_timezone) || DEFAULT_CONFERENCE_TIMEZONE,
    rowCount: toFiniteEpochMs(record.row_count),
    firstStartMs: toFiniteEpochMs(record.first_start_unix_ms),
    lastEndMs: toFiniteEpochMs(record.last_end_unix_ms),
    indexUrl: compactWhitespace(record.index_url),
    semanticEmbeddingsMetaUrl: compactWhitespace(record.semantic_embeddings_meta_url),
    semanticEmbeddingsBinUrl: compactWhitespace(record.semantic_embeddings_bin_url),
    label: compactWhitespace(record.label || record.conference_label) || conferenceLabelFromParts(shortName, year),
  };
}

function mergeConferenceTiming(conference, row = {}) {
  conference.rowCount = (conference.rowCount || 0) + 1;
  const startMs = toFiniteEpochMs(row.start_date_unix_ms);
  const endMs = toFiniteEpochMs(row.end_date_unix_ms);
  if (startMs != null && (conference.firstStartMs == null || startMs < conference.firstStartMs)) {
    conference.firstStartMs = startMs;
  }
  if (endMs != null && (conference.lastEndMs == null || endMs > conference.lastEndMs)) {
    conference.lastEndMs = endMs;
  } else if (startMs != null && conference.lastEndMs == null) {
    conference.lastEndMs = startMs;
  }
}

function conferencesFromScheduleIndex(scheduleIndex) {
  const records = Array.isArray(scheduleIndex?.conferences) ? scheduleIndex.conferences : [];
  const byKey = new Map();

  for (const record of records) {
    const conference = normalizeConferenceRecord(record);
    if (conference.key) {
      byKey.set(conference.key, conference);
    }
  }

  for (const row of Array.isArray(scheduleIndex?.rows) ? scheduleIndex.rows : []) {
    const conference = normalizeConferenceRecord(row);
    if (!conference.key) continue;
    if (!byKey.has(conference.key)) {
      byKey.set(conference.key, conference);
    }
    const stored = byKey.get(conference.key);
    if (stored) mergeConferenceTiming(stored, row);
  }

  return Array.from(byKey.values()).sort((left, right) => {
    const yearDiff = Number(right.year || 0) - Number(left.year || 0);
    if (yearDiff) return yearDiff;
    return left.label.localeCompare(right.label);
  });
}

function conferenceMatchesTimeScope(conference, timeScopeKey = selectedTimeScopeKey) {
  const nowMs = Date.now();
  const startMs = conference.firstStartMs;
  const endMs = conference.lastEndMs ?? startMs;

  if (timeScopeKey === 'past') {
    return endMs != null && endMs < nowMs;
  }
  if (timeScopeKey === 'upcoming') {
    return endMs == null || endMs >= nowMs;
  }
  if (timeScopeKey === 'upcoming-3m') {
    return startMs != null && startMs >= nowMs + THREE_MONTHS_MS;
  }
  return true;
}

function conferenceMatchesFilters(conference, { timeScopeKey = selectedTimeScopeKey, field = selectedField } = {}) {
  return normalizeConferenceField(conference.field) === normalizeConferenceField(field) && conferenceMatchesTimeScope(conference, timeScopeKey);
}

function hasConferencesFor({ timeScopeKey = selectedTimeScopeKey, field = selectedField } = {}) {
  return availableConferences.some((conference) => conferenceMatchesFilters(conference, { timeScopeKey, field }));
}

function editionConferences() {
  return availableConferences.filter((conference) => conferenceMatchesFilters(conference));
}

function getSelectedConference() {
  const conferences = editionConferences();
  return conferences.find((conference) => conference.key === selectedConferenceKey) || conferences[0] || null;
}

function selectedConferenceLabel() {
  return getSelectedConference()?.label || 'Selected Conference';
}

function selectedConferenceTimezone() {
  return getSelectedConference()?.timezone || DEFAULT_CONFERENCE_TIMEZONE;
}

function scoringScopeLabel() {
  return searchAcrossAllConferences ? `all ${selectedTimeScope().searchLabel}` : selectedConferenceLabel();
}

function activeScheduleRows() {
  if (searchAcrossAllConferences) {
    return editionConferences().flatMap((conference) => new Array(Number(conference.rowCount || 0)).fill(conference.key));
  }
  const selected = getSelectedConference();
  return selected ? new Array(Number(selected.rowCount || 0)).fill(selected.key) : [];
}

function updateActiveScheduleRowCount() {
  const conferences = searchAcrossAllConferences ? editionConferences() : [getSelectedConference()].filter(Boolean);
  scheduleRowCount = conferences.reduce((sum, conference) => sum + Number(conference.rowCount || 0), 0);
  return scheduleRowCount;
}

function conferenceIndexUrl(conference) {
  return conference?.indexUrl || `data/conferences/${conference?.key || 'unknown-conference'}/schedule_index.json`;
}

async function loadConferenceScheduleIndex(conference) {
  if (!conference?.key) {
    throw new Error('No conference selected.');
  }
  const indexUrl = conferenceIndexUrl(conference);
  if (scheduleIndexCache.has(indexUrl)) {
    return scheduleIndexCache.get(indexUrl);
  }
  const response = await fetch(baseUrlPath(indexUrl));
  if (!response.ok) {
    throw new Error(`Could not load ${conference.label || conference.key} schedule index (${response.status}).`);
  }
  const scheduleIndex = await response.json();
  scheduleIndexCache.set(indexUrl, scheduleIndex);
  return scheduleIndex;
}

function mergeScheduleDocFreq(indexes) {
  const merged = Object.create(null);
  for (const index of indexes) {
    const docFreq = index?.schedule_doc_freq || {};
    for (const [term, count] of Object.entries(docFreq)) {
      merged[term] = (merged[term] || 0) + Number(count || 0);
    }
  }
  return merged;
}

function rowsWithSequentialIndexes(indexes) {
  const rows = [];
  for (const index of indexes) {
    for (const row of Array.isArray(index?.rows) ? index.rows : []) {
      rows.push({
        ...row,
        row_index: rows.length,
      });
    }
  }
  return rows;
}

function semanticEmbeddingShardFor(index, conference, rowCount) {
  const metaUrl =
    index?.semantic_embeddings_meta_url ||
    conference?.semanticEmbeddingsMetaUrl ||
    `data/conferences/${conference?.key || index?.shard_key}/schedule_semantic_embeddings_q4.json`;
  const binUrl =
    index?.semantic_embeddings_bin_url ||
    conference?.semanticEmbeddingsBinUrl ||
    `data/conferences/${conference?.key || index?.shard_key}/schedule_semantic_embeddings_q4.bin`;
  return {
    key: conference?.key || index?.shard_key || '',
    label: conference?.label || '',
    row_count: rowCount,
    meta_url: metaUrl,
    bin_url: binUrl,
  };
}

async function loadActiveScheduleIndex() {
  const conferences = searchAcrossAllConferences ? editionConferences() : [getSelectedConference()].filter(Boolean);
  if (!conferences.length) {
    throw new Error(`No conferences found for ${scoringScopeLabel()}.`);
  }

  const indexes = await Promise.all(conferences.map((conference) => loadConferenceScheduleIndex(conference)));
  if (indexes.length === 1) {
    const index = indexes[0];
    const rows = rowsWithSequentialIndexes([index]);
    const conference = conferences[0];
    return {
      ...index,
      rows,
      row_count: rows.length,
      schedule_doc_freq: index.schedule_doc_freq || mergeScheduleDocFreq([index]),
      semantic_embeddings_meta_url:
        index.semantic_embeddings_meta_url || conference.semanticEmbeddingsMetaUrl,
      semantic_embeddings_bin_url:
        index.semantic_embeddings_bin_url || conference.semanticEmbeddingsBinUrl,
      semantic_embedding_shards: [semanticEmbeddingShardFor(index, conference, rows.length)],
    };
  }

  const rows = rowsWithSequentialIndexes(indexes);
  return {
    version: 4,
    sharded: true,
    shard_key: 'active-scope',
    row_count: rows.length,
    conferences: conferences.map((conference) => ({
      key: conference.key,
      id: conference.id,
      short_name: conference.shortName,
      year: conference.year,
      field: conference.field,
      timezone: conference.timezone,
      label: conference.label,
      row_count: conference.rowCount,
      first_start_unix_ms: conference.firstStartMs,
      last_end_unix_ms: conference.lastEndMs,
      index_url: conference.indexUrl,
      semantic_embeddings_meta_url: conference.semanticEmbeddingsMetaUrl,
      semantic_embeddings_bin_url: conference.semanticEmbeddingsBinUrl,
    })),
    rows,
    schedule_doc_freq: mergeScheduleDocFreq(indexes),
    semantic_embedding_shards: indexes.map((index, idx) =>
      semanticEmbeddingShardFor(index, conferences[idx], Array.isArray(index?.rows) ? index.rows.length : 0)
    ),
  };
}

function conferenceOptionText(conference, includeCount = false) {
  const label = conference?.label || 'Unknown Conference';
  if (!includeCount) return label;
  const rowCount = Number(conference?.rowCount || 0);
  return `${label} (${rowCount.toLocaleString()})`;
}

function setConferenceSelectCountVisibility(showCounts) {
  if (!conferenceSelect || conferenceSelectShowingCounts === showCounts) return;
  conferenceSelectShowingCounts = showCounts;
  for (const option of Array.from(conferenceSelect.options)) {
    const label = option.dataset.label || option.textContent || '';
    const count = Number(option.dataset.rowCount || 0);
    option.textContent = showCounts && count > 0 ? `${label} (${count.toLocaleString()})` : label;
  }
}

function populateConferenceSelector() {
  if (!conferenceSelect) return;
  conferenceSelect.innerHTML = '';
  conferenceSelectShowingCounts = false;

  const conferences = editionConferences();
  if (!conferences.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No matching conferences';
    conferenceSelect.appendChild(option);
    conferenceSelect.disabled = true;
    selectedConferenceKey = '';
    syncSelectContentWidth(conferenceSelect);
    return;
  }

  if (!conferences.some((conference) => conference.key === selectedConferenceKey)) {
    selectedConferenceKey = conferences[0]?.key || '';
  }

  for (const conference of conferences) {
    const option = document.createElement('option');
    option.value = conference.key;
    option.dataset.label = conference.label;
    option.dataset.rowCount = String(conference.rowCount || 0);
    option.textContent = conferenceOptionText(conference, false);
    option.selected = conference.key === selectedConferenceKey;
    conferenceSelect.appendChild(option);
  }

  conferenceSelect.value = selectedConferenceKey || conferences[0]?.key || '';
  conferenceSelect.disabled = searchAcrossAllConferences;
  conferenceSelect.title = searchAcrossAllConferences
    ? 'Disabled while searching across all conferences.'
    : '';
  syncSelectContentWidth(conferenceSelect);
}

function ensureValidConferenceFilters({ preferredTimeScopeKey = selectedTimeScopeKey } = {}) {
  if (TIME_SCOPE_OPTIONS.has(FIXED_TIME_SCOPE_KEY) && hasConferencesFor({ timeScopeKey: FIXED_TIME_SCOPE_KEY })) {
    selectedTimeScopeKey = FIXED_TIME_SCOPE_KEY;
    return;
  }

  if (TIME_SCOPE_OPTIONS.has(preferredTimeScopeKey) && hasConferencesFor({ timeScopeKey: preferredTimeScopeKey })) {
    selectedTimeScopeKey = preferredTimeScopeKey;
    return;
  }

  for (const timeScopeKey of TIME_SCOPE_OPTIONS.keys()) {
    if (hasConferencesFor({ timeScopeKey })) {
      selectedTimeScopeKey = timeScopeKey;
      return;
    }
  }
}

function populateTimeScopeSelector() {
  if (!timeScopeSelect) return;
  selectedTimeScopeKey = FIXED_TIME_SCOPE_KEY;
  timeScopeSelect.innerHTML = '';
  for (const [key, optionConfig] of TIME_SCOPE_OPTIONS.entries()) {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = optionConfig.label;
    option.disabled = !hasConferencesFor({ timeScopeKey: key });
    option.selected = key === selectedTimeScopeKey;
    timeScopeSelect.appendChild(option);
  }
  timeScopeSelect.value = selectedTimeScopeKey;
  syncSelectContentWidth(timeScopeSelect);
}

function updateSearchAllConferencesLabel() {
  if (!searchAllConferencesLabel) return;
  searchAllConferencesLabel.textContent = `Search across all ${selectedTimeScope().searchLabel}`;
}

function updateConferenceDocumentTitle() {
  const label = selectedConferenceLabel();
  document.title = label ? `${APP_NAME} | ${label}` : APP_NAME;
}

function initializeConferenceControls(scheduleIndex) {
  scheduleIndexPayload = scheduleIndex;
  availableConferences = conferencesFromScheduleIndex(scheduleIndexPayload);
  selectedTimeScopeKey = FIXED_TIME_SCOPE_KEY;
  ensureValidConferenceFilters();
  populateTimeScopeSelector();
  selectedConferenceKey =
    normalizeConferenceKey(conferenceSelect?.value) ||
    editionConferences()[0]?.key ||
    normalizeConferenceKey(scheduleIndexPayload?.rows?.[0]?.conference_key);
  searchAcrossAllConferences = Boolean(searchAllConferencesToggle?.checked);
  populateConferenceSelector();
  updateSearchAllConferencesLabel();
  updateActiveScheduleRowCount();
  updateConferenceDocumentTitle();
}

async function rerunForConferenceScopeChange() {
  selectedTimeScopeKey = FIXED_TIME_SCOPE_KEY;
  ensureValidConferenceFilters();
  searchAcrossAllConferences = Boolean(searchAllConferencesToggle?.checked);
  selectedConferenceKey = normalizeConferenceKey(conferenceSelect?.value) || selectedConferenceKey;
  populateTimeScopeSelector();
  populateConferenceSelector();
  updateSearchAllConferencesLabel();
  updateActiveScheduleRowCount();
  updateConferenceDocumentTitle();

  if (!extractedContext || isExtracting) {
    clearResults();
    setStatus(`Selected ${scoringScopeLabel()} (${scheduleRowCount.toLocaleString()} rows). Ready.`);
    return;
  }

  try {
    await runScoringFromExtracted(`Recomputing for ${scoringScopeLabel()}...`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    appendStatusItem(`Error: ${message}`);
  }
}

async function rerunForOwnWorkExclusionChange() {
  selectedOwnWorkExclusion();
  if (!extractedContext || isExtracting) return;

  try {
    await runScoringFromExtracted('Recomputing with own-work exclusion settings...');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    appendStatusItem(`Error: ${message}`);
  }
}

function validateUploads(files, { allowEmpty = false } = {}) {
  if (!files.length && !allowEmpty) throw new Error('Upload PDFs, select abstracts, or add keywords.');

  for (const file of files) {
    const isPdfByType = file.type === 'application/pdf';
    const isPdfByName = file.name.toLowerCase().endsWith('.pdf');
    if (!isPdfByType && !isPdfByName) {
      throw new Error(`File '${file.name}' is not a PDF.`);
    }
  }
}

function compactWhitespace(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .trim();
}

function truncateWords(text, maxWords = PROFILE_ABSTRACT_PREVIEW_WORDS) {
  const normalized = compactWhitespace(text);
  if (!normalized) return '';
  const words = normalized.split(' ');
  if (words.length <= maxWords) return normalized;
  return `${words.slice(0, maxWords).join(' ')}…`;
}

function normalizePersonName(text) {
  return compactWhitespace(text)
    .toLowerCase()
    .replace(/[^a-z0-9 ]+/g, '')
    .replace(/\s+/g, ' ');
}

function formatCoauthorsLabel(authorNames, selectedAuthorName = '', maxShown = 3) {
  const selectedNorm = normalizePersonName(selectedAuthorName);
  const seen = new Set();
  const coauthors = [];

  for (const rawName of Array.isArray(authorNames) ? authorNames : []) {
    const name = compactWhitespace(rawName);
    if (!name) continue;
    const norm = normalizePersonName(name);
    if (!norm) continue;
    if (selectedNorm && norm === selectedNorm) continue;
    if (seen.has(norm)) continue;
    seen.add(norm);
    coauthors.push(name);
  }

  if (!coauthors.length) return '';
  const shown = coauthors.slice(0, maxShown);
  const hiddenCount = coauthors.length - shown.length;
  if (hiddenCount > 0) {
    return `${shown.join(', ')} +${hiddenCount} more`;
  }
  return shown.join(', ');
}

function looksLikeRealAbstract(text) {
  const normalized = compactWhitespace(text);
  if (normalized.length < 45) return false;
  const lower = normalized.toLowerCase();
  const blockedFragments = [
    'request full-text',
    'discover researchgate',
    'researchgate has not been able to resolve',
    'no abstract available',
  ];
  return !blockedFragments.some((fragment) => lower.includes(fragment));
}

function slugToName(input) {
  return decodeURIComponent(String(input || ''))
    .replace(/[-_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function parseResearchGateProfileName(profileUrl) {
  const parsed = new URL(profileUrl);
  const parts = parsed.pathname.split('/').filter(Boolean);
  const profileIndex = parts.findIndex((part) => part.toLowerCase() === 'profile');
  if (profileIndex >= 0 && parts[profileIndex + 1]) {
    return slugToName(parts[profileIndex + 1]);
  }
  if (parts[0]) {
    return slugToName(parts[0]);
  }
  return '';
}

function normalizeAuthorNameHint(input) {
  return compactWhitespace(String(input || ''));
}

function toOpenAlexAuthorId(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  const parts = raw.split('/');
  const tail = compactWhitespace(parts[parts.length - 1] || '');
  return tail.toUpperCase().startsWith('A') ? tail.toUpperCase() : '';
}

function formatOpenAlexSourceUrl(urlValue) {
  const raw = compactWhitespace(urlValue);
  if (!raw) return '';
  if (raw.startsWith('http://') || raw.startsWith('https://')) {
    return raw;
  }
  if (raw.toUpperCase().startsWith('HTTPS://OPENALEX.ORG/')) {
    return raw;
  }
  if (raw.toUpperCase().startsWith('A')) {
    return `https://openalex.org/${raw.toUpperCase()}`;
  }
  return raw;
}

function reconstructOpenAlexAbstract(invertedIndex) {
  if (!invertedIndex || typeof invertedIndex !== 'object') return '';
  const positioned = [];
  for (const [token, indexes] of Object.entries(invertedIndex)) {
    if (!Array.isArray(indexes)) continue;
    for (const index of indexes) {
      if (!Number.isInteger(index)) continue;
      positioned.push({ index, token });
    }
  }
  if (!positioned.length) return '';
  positioned.sort((a, b) => a.index - b.index);
  return compactWhitespace(
    positioned
      .map((entry) => entry.token)
      .join(' ')
      .replace(/\s+([,.;:!?])/g, '$1')
  );
}

function authorMatchScore(author, targetName) {
  const normalizedTarget = compactWhitespace(targetName).toLowerCase();
  const normalizedAuthor = compactWhitespace(author?.display_name || '').toLowerCase();
  if (!normalizedTarget || !normalizedAuthor) return -1;
  const tokens = normalizedTarget.split(' ').filter(Boolean);
  let score = 0;
  for (const token of tokens) {
    if (normalizedAuthor.includes(token)) score += 3;
  }
  if (normalizedAuthor === normalizedTarget) score += 8;
  if (String(author?.orcid || '').length > 0) score += 1;
  if (Number.isFinite(author?.works_count)) score += Math.min(4, Math.log10((author.works_count || 0) + 1));
  if (Number.isFinite(author?.relevance_score)) score += Math.min(4, Math.log10((author.relevance_score || 0) + 1));
  return score;
}

function toOpenAlexCandidate(author, authorQuery) {
  const id = toOpenAlexAuthorId(author?.id);
  if (!id) return null;
  const score = authorMatchScore(author, authorQuery);
  const institution = compactWhitespace(author?.last_known_institution?.display_name || '');
  return {
    id,
    displayName: compactWhitespace(author?.display_name || 'Unknown author'),
    institution,
    worksCount: Number.isFinite(author?.works_count) ? Number(author.works_count) : 0,
    citedByCount: Number.isFinite(author?.cited_by_count) ? Number(author.cited_by_count) : 0,
    orcid: compactWhitespace(author?.orcid || ''),
    score,
  };
}

async function fetchOpenAlexJson(url) {
  const response = await fetch(url, { method: 'GET' });
  if (!response.ok) {
    throw new Error(`OpenAlex request failed (${response.status})`);
  }
  return response.json();
}

async function fetchOpenAlexAuthorCandidates(authorName, { perPage = 10 } = {}) {
  const normalizedAuthorName = normalizeAuthorNameHint(authorName);
  if (!normalizedAuthorName || normalizedAuthorName.length < 3) {
    return [];
  }

  const authorSearchUrl = `https://api.openalex.org/authors?search=${encodeURIComponent(normalizedAuthorName)}&per-page=${Math.min(
    20,
    Math.max(5, Number(perPage) || 10)
  )}`;
  const authorPayload = await fetchOpenAlexJson(authorSearchUrl);
  const rawCandidates = Array.isArray(authorPayload?.results) ? authorPayload.results : [];
  if (!rawCandidates.length) {
    return [];
  }

  return rawCandidates
    .map((author) => toOpenAlexCandidate(author, normalizedAuthorName))
    .filter(Boolean)
    .sort((left, right) => right.score - left.score);
}

async function fetchOpenAlexAbstractsByAuthorId(authorId, { source = 'openalex', fallbackUrl = '' } = {}) {
  const normalizedAuthorId = toOpenAlexAuthorId(authorId);
  if (!normalizedAuthorId) {
    return [];
  }

  const worksUrl = `https://api.openalex.org/works?filter=author.id:${encodeURIComponent(normalizedAuthorId)}&per-page=40&sort=publication_year:desc`;
  const worksPayload = await fetchOpenAlexJson(worksUrl);
  const works = Array.isArray(worksPayload?.results) ? worksPayload.results : [];

  const results = [];
  const seen = new Set();
  for (const work of works) {
    const title = compactWhitespace(work?.display_name || work?.title || 'Untitled Research Output');
    const abstract = reconstructOpenAlexAbstract(work?.abstract_inverted_index);
    if (!looksLikeRealAbstract(abstract)) continue;

    const sourceUrl =
      work?.primary_location?.landing_page_url ||
      work?.doi ||
      work?.id ||
      fallbackUrl;

    const fingerprint = `${title.toLowerCase()}::${abstract.slice(0, 170).toLowerCase()}`;
    if (seen.has(fingerprint)) continue;
    seen.add(fingerprint);

    results.push({
      title,
      abstract,
      authorNames: Array.isArray(work?.authorships)
        ? work.authorships
            .map((authorship) => compactWhitespace(authorship?.author?.display_name || ''))
            .filter(Boolean)
        : [],
      sourceUrl: formatOpenAlexSourceUrl(sourceUrl),
      source,
      publicationYear: Number.isFinite(work?.publication_year) ? work.publication_year : null,
    });
    if (results.length >= MAX_PROFILE_ABSTRACTS) break;
  }

  return results;
}

function buildAuthorLookupKey(authorQuery) {
  return normalizeAuthorNameHint(authorQuery).toLowerCase();
}

function tryParseProfileLikeUrl(raw) {
  const value = compactWhitespace(raw);
  if (!value) return null;
  const looksLikeUrl =
    /^https?:\/\//i.test(value) ||
    value.includes('scholar.google.') ||
    value.includes('researchgate.net/');
  if (!looksLikeUrl) return null;
  const prefixed = /^https?:\/\//i.test(value) ? value : `https://${value}`;
  try {
    return new URL(prefixed);
  } catch {
    return null;
  }
}

function resolveAuthorLookupInput(rawAuthorQuery) {
  const authorQuery = normalizeAuthorNameHint(rawAuthorQuery);

  if (!authorQuery) {
    throw new Error('Enter an author name.');
  }

  const parsedUrl = tryParseProfileLikeUrl(authorQuery);
  if (!parsedUrl) {
    if (authorQuery.length < 3) {
      throw new Error('Author name must be at least 3 characters.');
    }
    return { authorQuery };
  }

  const host = parsedUrl.hostname.toLowerCase();
  if (host.includes('scholar.google.')) {
    throw new Error('Google Scholar URLs are blocked in browser-only mode. Enter the author name instead.');
  }
  if (host === 'researchgate.net' || host.endsWith('.researchgate.net')) {
    const parsedName = normalizeAuthorNameHint(parseResearchGateProfileName(parsedUrl.toString()));
    if (parsedName.length < 3) {
      throw new Error('Could not infer an author name from this ResearchGate URL. Enter the author name instead.');
    }
    return { authorQuery: parsedName };
  }
  throw new Error('Enter an author name, or a ResearchGate profile URL.');
}

function getSelectedAuthorCandidate() {
  if (!fetchedAuthorCandidates.length) return null;
  return (
    fetchedAuthorCandidates.find((candidate) => candidate.id === selectedAuthorCandidateId) ||
    fetchedAuthorCandidates[0] ||
    null
  );
}

function clearAuthorCandidates({ keepLookupKey = false } = {}) {
  fetchedAuthorCandidates = [];
  selectedAuthorCandidateId = '';
  if (!keepLookupKey) {
    activeAuthorLookupKey = '';
  }
  renderAuthorCandidates();
}

function clearFetchedProfileAbstracts() {
  profileAbstractsExpanded = false;
  fetchedProfileAbstracts = [];
  selectedProfileAbstractIds.clear();
  renderProfileAbstracts();
}

function applyFetchedProfileAbstracts(abstracts, sourceLabel, selectedAuthorName = '') {
  profileAbstractsExpanded = false;
  fetchedProfileAbstracts = abstracts.slice(0, MAX_PROFILE_ABSTRACTS).map((item, idx) => ({
    id: `${idx + 1}`,
    title: compactWhitespace(item.title).slice(0, 240),
    abstract: compactWhitespace(item.abstract).slice(0, 3000),
    coauthorsLabel: formatCoauthorsLabel(item.authorNames, selectedAuthorName),
    publicationYear: Number.isFinite(item.publicationYear) ? Number(item.publicationYear) : null,
    sourceUrl: item.sourceUrl || '',
    source: item.source || '',
    sourceLabel,
  }));

  selectedProfileAbstractIds.clear();
  for (const item of fetchedProfileAbstracts) {
    selectedProfileAbstractIds.add(item.id);
  }

  renderProfileAbstracts();
}

function updateAbstractPreviewToggleButtonLabel() {
  if (!toggleAbstractPreviewBtn) return;
  toggleAbstractPreviewBtn.textContent = profileAbstractsExpanded ? 'Collapse all' : 'Expand all';
}

function updateRunSelectedCount() {
  if (!runSelectedCount) return;
  const pdfCount = Array.from(fileInput?.files || []).length;
  const abstractCount = getSelectedProfileAbstracts().length;
  const keywordCount = parseCustomKeywords(keywordInput?.value || '').length;
  runSelectedCount.textContent = `${pdfCount} PDFs · ${abstractCount} profile abstracts · ${keywordCount} keywords`;
}

function updateAuthorCandidatesCount() {
  if (!profileAuthorCandidatesCount) return;
  if (!fetchedAuthorCandidates.length) {
    profileAuthorCandidatesCount.textContent = '';
    return;
  }
  const selectedCandidate = getSelectedAuthorCandidate();
  if (!selectedCandidate) {
    profileAuthorCandidatesCount.textContent = `${fetchedAuthorCandidates.length} matches`;
    return;
  }
  profileAuthorCandidatesCount.textContent = `${fetchedAuthorCandidates.length} matches, ${selectedCandidate.displayName} selected`;
}

function renderAuthorCandidates() {
  if (!profileAuthorCandidatesPanel || !profileAuthorCandidatesList) return;
  if (!fetchedAuthorCandidates.length) {
    profileAuthorCandidatesPanel.classList.add('hidden');
    profileAuthorCandidatesList.innerHTML = '';
    updateAuthorCandidatesCount();
    return;
  }

  const selectedCandidate = getSelectedAuthorCandidate();
  if (selectedCandidate) {
    selectedAuthorCandidateId = selectedCandidate.id;
  }

  profileAuthorCandidatesPanel.classList.remove('hidden');
  profileAuthorCandidatesList.innerHTML = fetchedAuthorCandidates
    .map((candidate) => {
      const checked = candidate.id === selectedAuthorCandidateId ? 'checked' : '';
      const metaBits = [];
      if (candidate.institution) metaBits.push(candidate.institution);
      if (candidate.worksCount > 0) metaBits.push(`${candidate.worksCount.toLocaleString()} works`);
      if (candidate.citedByCount > 0) metaBits.push(`${candidate.citedByCount.toLocaleString()} citations`);
      if (candidate.orcid) metaBits.push('ORCID listed');
      return `
        <label class="profile-author-candidate-item">
          <input type="radio" name="profile-author-candidate" data-author-id="${escapeHtml(candidate.id)}" ${checked} />
          <div class="profile-author-candidate-body">
            <p class="profile-author-candidate-name">${escapeHtml(candidate.displayName)}</p>
            <p class="profile-author-candidate-meta">${escapeHtml(metaBits.join(' • ') || 'No metadata details')}</p>
          </div>
        </label>
      `;
    })
    .join('');

  updateAuthorCandidatesCount();
}

async function fetchAbstractsForAuthorCandidate(candidate) {
  if (!candidate?.id) {
    return [];
  }
  const cacheKey = candidate.id;
  if (authorAbstractsCache.has(cacheKey)) {
    return authorAbstractsCache.get(cacheKey);
  }

  const abstracts = await fetchOpenAlexAbstractsByAuthorId(candidate.id, {
    source: 'openalex_author',
    fallbackUrl: `https://openalex.org/${candidate.id}`,
  });
  authorAbstractsCache.set(cacheKey, abstracts);
  return abstracts;
}

async function getAuthorCandidatesForQuery(authorQuery, { forceRefresh = false } = {}) {
  const cacheKey = buildAuthorLookupKey(authorQuery);
  if (!forceRefresh && authorCandidatesCache.has(cacheKey)) {
    return authorCandidatesCache.get(cacheKey);
  }
  const candidates = (await fetchOpenAlexAuthorCandidates(authorQuery, { perPage: 12 })).slice(0, 8);
  authorCandidatesCache.set(cacheKey, candidates);
  return candidates;
}

async function fetchAndRenderAbstractsForSelectedCandidate({ runSeq = null } = {}) {
  const selectedCandidate = getSelectedAuthorCandidate();
  if (!selectedCandidate) return false;

  updateProfileFetchStatus(`Fetching abstracts for ${selectedCandidate.displayName}...`);
  const abstracts = await fetchAbstractsForAuthorCandidate(selectedCandidate);
  if (runSeq != null && runSeq !== profileFetchRunSeq) {
    return false;
  }

  if (!abstracts.length) {
    clearFetchedProfileAbstracts();
    updateProfileFetchStatus(
      `No abstracts were found for ${selectedCandidate.displayName}. Select another match to refresh.`,
      true
    );
    return false;
  }

  const sourceLabel = `OpenAlex metadata (${selectedCandidate.displayName})`;
  applyFetchedProfileAbstracts(abstracts, sourceLabel, selectedCandidate.displayName);
  updateProfileFetchStatus(`Fetched ${fetchedProfileAbstracts.length} abstracts from ${selectedCandidate.displayName}.`);
  return true;
}

function updateProfileFetchStatus(text, isError = false) {
  if (!profileFetchStatus) return;
  profileFetchStatus.textContent = text;
  profileFetchStatus.classList.toggle('is-error', Boolean(isError));
}

function getSelectedProfileAbstracts() {
  if (!fetchedProfileAbstracts.length || !selectedProfileAbstractIds.size) return [];
  return fetchedProfileAbstracts.filter((item) => selectedProfileAbstractIds.has(item.id));
}

function updateProfileAbstractSelectionCount() {
  const selected = getSelectedProfileAbstracts().length;
  if (profileAbstractsCount) {
    profileAbstractsCount.textContent = `${selected} of ${fetchedProfileAbstracts.length} selected`;
  }
  if (toggleAbstractPreviewBtn) {
    toggleAbstractPreviewBtn.disabled = fetchedProfileAbstracts.length === 0;
  }
  updateRunSelectedCount();
  updateAbstractPreviewToggleButtonLabel();
}

function renderProfileAbstracts() {
  if (!profileAbstractsPanel || !profileAbstractsList) return;
  if (!fetchedProfileAbstracts.length) {
    profileAbstractsPanel.classList.add('hidden');
    profileAbstractsList.innerHTML = '';
    updateProfileAbstractSelectionCount();
    return;
  }

  profileAbstractsPanel.classList.remove('hidden');
  profileAbstractsList.innerHTML = fetchedProfileAbstracts
    .map((item) => {
      const checked = selectedProfileAbstractIds.has(item.id) ? 'checked' : '';
      const preview = profileAbstractsExpanded
        ? compactWhitespace(item.abstract)
        : truncateWords(item.abstract, PROFILE_ABSTRACT_PREVIEW_WORDS);
      const coauthorsLine = item.coauthorsLabel
        ? `<p class="profile-abstract-authors">Co-authors: ${escapeHtml(item.coauthorsLabel)}</p>`
        : '';
      return `
        <label class="profile-abstract-item">
          <input type="checkbox" data-abstract-id="${escapeHtml(item.id)}" ${checked} />
          <div class="profile-abstract-body">
            <p class="profile-abstract-title">${escapeHtml(item.title || 'Untitled')}</p>
            <p class="profile-abstract-source">${escapeHtml(item.sourceLabel || '')}</p>
            ${coauthorsLine}
            <p class="profile-abstract-text">${escapeHtml(preview)}</p>
          </div>
        </label>
      `;
    })
    .join('');
  updateProfileAbstractSelectionCount();
}

function parseCustomKeywords(raw) {
  const seen = new Set();
  const keywords = [];
  const parts = String(raw || '')
    .split(/[\n,]/g)
    .map((part) => part.trim())
    .filter(Boolean);

  for (const entry of parts) {
    const key = entry.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    keywords.push(entry);
  }

  return keywords.slice(0, 50);
}

function normalizeMatchingMode(value) {
  const mode = String(value || 'semantic')
    .trim()
    .toLowerCase();
  return MATCHING_MODES.has(mode) ? mode : 'semantic';
}

function normalizeSubstringMatchWordCount(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 6;
  return Math.min(8, Math.max(4, Math.floor(parsed)));
}

function selectedOwnWorkExclusion() {
  const minWords = normalizeSubstringMatchWordCount(substringMatchWordCountSelect?.value);
  if (substringMatchWordCountSelect) {
    substringMatchWordCountSelect.value = String(minWords);
    substringMatchWordCountSelect.disabled = !excludeSubstringMatchesInput?.checked;
  }
  return {
    enabled: Boolean(excludeSubstringMatchesInput?.checked),
    minWords,
  };
}

function matchingModeLabel(mode) {
  if (mode === 'semantic') return 'Semantic';
  if (mode === 'hybrid') return 'Hybrid';
  return 'TF-IDF';
}

function scoringStatusText(mode) {
  if (mode === 'semantic') return 'Computing semantic relevance in scoring worker...';
  if (mode === 'hybrid') return 'Computing hybrid relevance (TF-IDF + semantic) in scoring worker...';
  return 'Computing TF-IDF relevance in scoring worker...';
}

function modeNeedsSemanticModel(mode = normalizeMatchingMode(matchingModeInput?.value)) {
  return mode === 'semantic' || mode === 'hybrid';
}

function updateRunButtonAvailability() {
  if (!runBtn) return;
  if (isExtracting) {
    runBtn.disabled = true;
    return;
  }

  const needsSemanticModel = modeNeedsSemanticModel();
  const disabled = modelDownloadBusy || (needsSemanticModel && !semanticModelAvailable);
  runBtn.disabled = disabled;

  if (needsSemanticModel && !semanticModelAvailable) {
    runBtn.title = 'Download the semantic model, or use a deployment that bundles /models files.';
  } else {
    runBtn.removeAttribute('title');
  }
}

function formatBytes(bytes) {
  const numeric = Number(bytes);
  if (!Number.isFinite(numeric) || numeric < 0) return '';
  if (numeric < 1024) return `${Math.round(numeric)} B`;
  if (numeric < 1024 * 1024) return `${(numeric / 1024).toFixed(1)} KB`;
  return `${(numeric / (1024 * 1024)).toFixed(1)} MB`;
}

async function isUrlReachable(url) {
  try {
    const headResponse = await fetch(url, { method: 'HEAD', cache: 'no-store' });
    if (headResponse.ok) return true;
    if (headResponse.status !== 405 && headResponse.status !== 501) {
      return false;
    }
  } catch {
    // Fallback to GET below.
  }

  try {
    const getResponse = await fetch(url, { method: 'GET', cache: 'no-store' });
    return getResponse.ok;
  } catch {
    return false;
  }
}

async function isBundledSemanticModelAvailable() {
  const requiredLocalChecks = ['config.json', 'onnx/model_q4.onnx_data'];
  for (const file of requiredLocalChecks) {
    const url = semanticModelUrl(file, window.location.origin);
    const reachable = await isUrlReachable(url);
    if (!reachable) return false;
  }
  return true;
}

function setModelDownloadUi({
  busy = modelDownloadBusy,
  available = semanticModelAvailable,
  statusText = '',
  progressPercent = null,
  loadedBytes = null,
  totalBytes = null,
} = {}) {
  modelDownloadBusy = Boolean(busy);
  const modelReady = Boolean(available);
  if (downloadModelBtn) {
    downloadModelBtn.classList.toggle('hidden', modelReady && !modelDownloadBusy);
    downloadModelBtn.disabled = modelDownloadBusy || modelReady;
    downloadModelBtn.classList.toggle('is-ready', modelReady);
  }

  if (modelDownloadBusy) {
    if (downloadModelBtn) downloadModelBtn.textContent = 'Downloading model...';
  } else if (modelReady) {
    if (downloadModelBtn) downloadModelBtn.textContent = 'Model ready';
  } else {
    if (downloadModelBtn) downloadModelBtn.textContent = 'Download semantic model';
  }

  if (statusText && modelDownloadStatus) {
    modelDownloadStatus.textContent = statusText;
  }

  updateRunButtonAvailability();

  if (progressPercent == null) {
    if (modelDownloadProgressWrap) modelDownloadProgressWrap.classList.add('hidden');
    if (modelDownloadProgress) modelDownloadProgress.value = 0;
    if (modelDownloadProgressText) modelDownloadProgressText.textContent = '0%';
    return;
  }

  const normalized = Math.max(0, Math.min(100, Number(progressPercent)));
  if (modelDownloadProgressWrap) modelDownloadProgressWrap.classList.remove('hidden');
  if (modelDownloadProgress) modelDownloadProgress.value = normalized;

  const loaded = formatBytes(loadedBytes);
  const total = formatBytes(totalBytes);
  if (loaded && total) {
    if (modelDownloadProgressText) {
      modelDownloadProgressText.textContent = `${Math.round(normalized)}% (${loaded}/${total})`;
    }
  } else {
    if (modelDownloadProgressText) modelDownloadProgressText.textContent = `${Math.round(normalized)}%`;
  }
}

function pageTextFromItems(items) {
  const arranged = [];
  for (const item of items) {
    if (!item || typeof item.str !== 'string') continue;
    const text = item.str.trim();
    if (!text) continue;
    arranged.push({
      str: text,
      x: Array.isArray(item.transform) ? item.transform[4] || 0 : 0,
      y: Array.isArray(item.transform) ? item.transform[5] || 0 : 0,
    });
  }

  arranged.sort((a, b) => {
    const yDiff = b.y - a.y;
    if (Math.abs(yDiff) > 2) return yDiff;
    return a.x - b.x;
  });

  return arranged.map((entry) => entry.str).join(' ');
}

async function extractPdfText(file, onProgress) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const loadingTask = pdfjsLib.getDocument({ data: bytes });
  const doc = await loadingTask.promise;

  const chunks = [];
  for (let pageNo = 1; pageNo <= doc.numPages; pageNo += 1) {
    const page = await doc.getPage(pageNo);
    const content = await page.getTextContent({ normalizeWhitespace: true });
    chunks.push(pageTextFromItems(content.items));
    onProgress(pageNo, doc.numPages);
    page.cleanup();
  }

  await doc.cleanup();
  await doc.destroy();
  return chunks.join('\n');
}

async function extractAllPdfs(files) {
  const results = new Array(files.length);
  let cursor = 0;
  let active = 0;

  return new Promise((resolve, reject) => {
    const launch = () => {
      if (cursor >= files.length && active === 0) {
        resolve(results);
        return;
      }

      while (active < EXTRACTION_CONCURRENCY && cursor < files.length) {
        const idx = cursor;
        const file = files[cursor];
        cursor += 1;
        active += 1;

        const statusItem = appendStatusItem(`Extracting ${file.name} (0%)`);

        extractPdfText(file, (pageNo, totalPages) => {
          const pct = Math.round((pageNo / totalPages) * 100);
          statusItem.textContent = `Extracting ${file.name} (${pct}%)`;
        })
          .then((text) => {
            results[idx] = text;
            statusItem.textContent = `Extracted ${file.name} (${text.length.toLocaleString()} chars)`;
          })
          .catch((err) => {
            statusItem.textContent = `Failed ${file.name}: ${err.message}`;
            reject(err);
          })
          .finally(() => {
            active -= 1;
            launch();
          });
      }
    };

    launch();
  });
}

async function runWorkerScore({
  worksTexts,
  workNames,
  customKeywords,
  matchingMode,
  conferenceKey,
  searchAcrossAllConferences,
  ownWorkExclusion,
}) {
  const requestId = ++requestSeq;
  const normalizedMode = normalizeMatchingMode(matchingMode);
  const timeoutMs = normalizedMode === 'tfidf' ? 120000 : 900000;
  const activeScheduleIndex = await loadActiveScheduleIndex();
  scheduleRowCount = Number(activeScheduleIndex.row_count || activeScheduleIndex.rows?.length || 0);
  const topN = scheduleRowCount || 10000;

  return new Promise((resolve, reject) => {
    const timeoutHandle = window.setTimeout(() => {
      if (!pendingRequests.has(requestId)) return;
      pendingRequests.delete(requestId);
      reject(new Error(`Scoring timed out after ${Math.round(timeoutMs / 1000)}s.`));
    }, timeoutMs);

    pendingRequests.set(requestId, {
      resolve: (value) => {
        window.clearTimeout(timeoutHandle);
        resolve(value);
      },
      reject: (error) => {
        window.clearTimeout(timeoutHandle);
        reject(error);
      },
    });

    scoreWorker.postMessage({
      type: 'init',
      quiet: true,
      scheduleIndex: activeScheduleIndex,
    });

    scoreWorker.postMessage({
      type: 'run',
      requestId,
      worksTexts,
      workNames,
      customKeywords,
      matchingMode: normalizedMode,
      conferenceKey,
      searchAcrossAllConferences: Boolean(searchAcrossAllConferences),
      ownWorkExclusion,
      topN,
    });
  });
}

function startWorkerPrefetchModel() {
  const requestId = ++requestSeq;
  const timeoutMs = 1800000;

  const promise = new Promise((resolve, reject) => {
    const timeoutHandle = window.setTimeout(() => {
      if (!pendingRequests.has(requestId)) return;
      pendingRequests.delete(requestId);
      reject(new Error(`Model download timed out after ${Math.round(timeoutMs / 1000)}s.`));
    }, timeoutMs);

    pendingRequests.set(requestId, {
      resolve: (value) => {
        window.clearTimeout(timeoutHandle);
        resolve(value);
      },
      reject: (error) => {
        window.clearTimeout(timeoutHandle);
        reject(error);
      },
    });

    scoreWorker.postMessage({
      type: 'prefetch-model',
      requestId,
    });
  });

  return { requestId, promise };
}

function startWorkerProbeModelAvailability() {
  const requestId = ++requestSeq;
  const timeoutMs = 120000;

  const promise = new Promise((resolve, reject) => {
    const timeoutHandle = window.setTimeout(() => {
      if (!pendingRequests.has(requestId)) return;
      pendingRequests.delete(requestId);
      reject(new Error(`Model availability probe timed out after ${Math.round(timeoutMs / 1000)}s.`));
    }, timeoutMs);

    pendingRequests.set(requestId, {
      resolve: (value) => {
        window.clearTimeout(timeoutHandle);
        resolve(value);
      },
      reject: (error) => {
        window.clearTimeout(timeoutHandle);
        reject(error);
      },
    });

    scoreWorker.postMessage({
      type: 'probe-model',
      requestId,
    });
  });

  return { requestId, promise };
}

function asEpochMs(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatStartDate(epochMs) {
  if (epochMs == null) return 'n/a';
  return startDateFormatter.format(new Date(epochMs));
}

function formatTimeRange(startMs, endMs) {
  if (startMs == null) return 'n/a';
  const startText = timeFormatter.format(new Date(startMs));
  if (endMs == null || endMs <= startMs) return startText;
  const endText = timeFormatter.format(new Date(endMs));
  return `${startText} - ${endText}`;
}

function dayPillClass(day) {
  const normalized = String(day || '')
    .trim()
    .toLowerCase();
  return DAY_PILL_CLASS_MAP[normalized] || 'day-pill-other';
}

function pad2(value) {
  return String(value).padStart(2, '0');
}

function formatUtcIcsDateTime(epochMs) {
  const dt = new Date(epochMs);
  return `${dt.getUTCFullYear()}${pad2(dt.getUTCMonth() + 1)}${pad2(dt.getUTCDate())}T${pad2(dt.getUTCHours())}${pad2(dt.getUTCMinutes())}${pad2(dt.getUTCSeconds())}Z`;
}

function formatFloatingIcsDateTime(epochMs) {
  const dt = new Date(epochMs);
  return `${dt.getUTCFullYear()}${pad2(dt.getUTCMonth() + 1)}${pad2(dt.getUTCDate())}T${pad2(dt.getUTCHours())}${pad2(dt.getUTCMinutes())}${pad2(dt.getUTCSeconds())}`;
}

function escapeIcsText(value) {
  return String(value || '')
    .replaceAll('\r', '')
    .replaceAll('\\', '\\\\')
    .replaceAll('\n', '\\n')
    .replaceAll(';', '\\;')
    .replaceAll(',', '\\,')
    .replaceAll(/\u0000/g, '');
}

function foldIcsLine(line) {
  const text = String(line || '');
  if (text.length <= 73) return text;

  let folded = text.slice(0, 73);
  let cursor = 73;
  while (cursor < text.length) {
    folded += `\r\n ${text.slice(cursor, cursor + 72)}`;
    cursor += 72;
  }
  return folded;
}

function slugifyForFilename(value) {
  return String(value || '')
    .normalize('NFKD')
    .replace(/[^\x00-\x7F]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 72);
}

function buildCalendarPayload(row, rowIndex) {
  const startMs = asEpochMs(row.start_date_unix_ms);
  const endMs = asEpochMs(row.end_date_unix_ms);
  if (startMs == null) {
    throw new Error('This session is missing a start time in the schedule data.');
  }

  const normalizedEndMs = endMs && endMs > startMs ? endMs : startMs + FALLBACK_EVENT_DURATION_MS;
  const conferenceLabel = compactWhitespace(row.conference_label) || selectedConferenceLabel();
  const summary = String(row.title || `${conferenceLabel} session #${row.relevance_rank || rowIndex + 1}`);
  const location = [row.room, row.building].filter(Boolean).join(', ');
  const descriptionParts = [];
  if (conferenceLabel) descriptionParts.push(`Conference: ${conferenceLabel}`);
  if (row.authors) descriptionParts.push(`Authors: ${row.authors}`);
  if (row.day) descriptionParts.push(`Day: ${row.day}`);
  if (row.session_type) descriptionParts.push(`Session type: ${row.session_type}`);
  if (row.abstract) descriptionParts.push(row.abstract);
  const description = descriptionParts.join('\n\n');

  const uidSlug = slugifyForFilename(`${conferenceLabel}-${summary}-${startMs}-${rowIndex + 1}`) || `conference-${rowIndex + 1}`;
  const dtStamp = formatUtcIcsDateTime(Date.now());
  const dtStart = formatFloatingIcsDateTime(startMs);
  const dtEnd = formatFloatingIcsDateTime(normalizedEndMs);
  const calendarName = conferenceLabel ? `${conferenceLabel} ${APP_NAME}` : APP_NAME;
  const conferenceTimezone = compactWhitespace(row.conference_timezone) || selectedConferenceTimezone();

  const lines = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    `PRODID:${ICS_PROD_ID}`,
    'CALSCALE:GREGORIAN',
    'METHOD:PUBLISH',
    `X-WR-CALNAME:${escapeIcsText(calendarName)}`,
    `X-WR-TIMEZONE:${conferenceTimezone}`,
    'BEGIN:VEVENT',
    `UID:${uidSlug}@cagpap-client`,
    `DTSTAMP:${dtStamp}`,
    `CREATED:${dtStamp}`,
    `LAST-MODIFIED:${dtStamp}`,
    'SEQUENCE:0',
    'STATUS:CONFIRMED',
    'TRANSP:OPAQUE',
    `DTSTART;TZID=${conferenceTimezone}:${dtStart}`,
    `DTEND;TZID=${conferenceTimezone}:${dtEnd}`,
    `SUMMARY:${escapeIcsText(summary)}`,
  ];

  if (location) {
    lines.push(`LOCATION:${escapeIcsText(location)}`);
  }
  if (description) {
    lines.push(`DESCRIPTION:${escapeIcsText(description)}`);
  }

  lines.push('END:VEVENT', 'END:VCALENDAR');

  const icsText = `${lines.map(foldIcsLine).join('\r\n')}\r\n`;
  const fileBase = slugifyForFilename(`${conferenceLabel}-${summary}`) || `conference-session-${rowIndex + 1}`;

  return {
    icsText,
    fileName: `${fileBase}.ics`,
    summary,
  };
}

function updateFileSelectionText() {
  const files = Array.from(fileInput.files || []);
  if (clearPdfsBtn) {
    const hasFiles = files.length > 0;
    clearPdfsBtn.hidden = !hasFiles;
    clearPdfsBtn.disabled = !hasFiles;
  }
  if (!files.length) {
    if (fileSelection) {
      fileSelection.textContent = '';
      fileSelection.hidden = true;
    }
  } else if (files.length === 1) {
    if (fileSelection) {
      fileSelection.textContent = files[0].name;
      fileSelection.hidden = false;
    }
  } else {
    if (fileSelection) {
      fileSelection.textContent = `${files.length} files selected`;
      fileSelection.hidden = false;
    }
  }
  updateRunSelectedCount();
}

function normalizeCalendarKeyPart(value) {
  return String(value || '')
    .trim()
    .toLowerCase();
}

function calendarItemKey(row) {
  const payload = {
    conference: normalizeCalendarKeyPart(row?.conference_key || row?.conference_label),
    title: normalizeCalendarKeyPart(row?.title),
    start: asEpochMs(row?.start_date_unix_ms),
    end: asEpochMs(row?.end_date_unix_ms),
    room: normalizeCalendarKeyPart(row?.room),
    building: normalizeCalendarKeyPart(row?.building),
  };
  return JSON.stringify(payload);
}

function loadDownloadedCalendarItems() {
  try {
    const raw = window.localStorage.getItem(CALENDAR_DOWNLOADS_STORAGE_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((entry) => typeof entry === 'string'));
  } catch {
    return new Set();
  }
}

function persistDownloadedCalendarItems() {
  try {
    window.localStorage.setItem(CALENDAR_DOWNLOADS_STORAGE_KEY, JSON.stringify(Array.from(downloadedCalendarItems)));
  } catch {
    // Ignore storage write failures (e.g., private mode quota restrictions).
  }
}

function hasDownloadedCalendarItem(row) {
  return downloadedCalendarItems.has(calendarItemKey(row));
}

function markCalendarItemDownloaded(row) {
  downloadedCalendarItems.add(calendarItemKey(row));
  persistDownloadedCalendarItems();
}

function setCalendarButtonDownloadedState(button, downloaded) {
  if (!(button instanceof HTMLElement)) return;
  button.classList.toggle('is-added', downloaded);
  button.setAttribute('data-downloaded', downloaded ? 'true' : 'false');
}

function renderBoostedKeywords() {
  const keywords = parseCustomKeywords(keywordInput.value);
  boostedKeywordsList.innerHTML = '';

  if (!keywords.length) {
    const empty = document.createElement('p');
    empty.className = 'muted boosted-keywords-empty';
    empty.textContent = 'No keywords yet.';
    boostedKeywordsList.appendChild(empty);
    return;
  }

  for (const keyword of keywords) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'boosted-keyword-chip';
    chip.dataset.keyword = keyword;
    chip.setAttribute('aria-label', `Remove keyword: ${keyword}`);

    const text = document.createElement('span');
    text.textContent = keyword;

    const remove = document.createElement('span');
    remove.className = 'chip-remove';
    remove.setAttribute('aria-hidden', 'true');
    remove.textContent = 'x';

    chip.append(text, remove);
    boostedKeywordsList.appendChild(chip);
  }

  updateRunSelectedCount();
}

async function refreshModelCacheUi({ forceUi = false } = {}) {
  if (modelDownloadBusy && !forceUi) {
    updateRunButtonAvailability();
    return {
      bundled: false,
      probeAvailable: semanticModelAvailable,
      available: semanticModelAvailable,
      probeError: '',
    };
  }

  const refreshSeq = ++modelAvailabilityRefreshSeq;
  let bundled = false;
  let probeAvailable = false;
  let probeError = '';

  try {
    bundled = await isBundledSemanticModelAvailable();
  } catch {
    bundled = false;
  }

  if (!bundled) {
    try {
      const task = startWorkerProbeModelAvailability();
      const payload = await task.promise;
      const probeResult = payload?.result || {};
      probeAvailable = Boolean(probeResult.available);
      if (!probeAvailable && probeResult.error) {
        probeError = String(probeResult.error);
      }
    } catch (error) {
      probeAvailable = false;
      probeError = error instanceof Error ? error.message : String(error);
    }
  } else {
    probeAvailable = true;
  }

  const available = bundled || probeAvailable;
  const snapshot = {
    bundled,
    probeAvailable,
    available,
    probeError,
  };

  if (refreshSeq !== modelAvailabilityRefreshSeq) {
    return snapshot;
  }

  semanticModelAvailable = available;

  if (bundled) {
    setModelDownloadUi({
      busy: false,
      available: true,
      statusText: 'Semantic model files are bundled in this deployment. You can run Semantic/Hybrid now.',
    });
    return snapshot;
  }

  if (probeAvailable) {
    setModelDownloadUi({
      busy: false,
      available: true,
      statusText: 'Semantic model is already available in this browser session. You can run Semantic/Hybrid now.',
    });
    return snapshot;
  }

  setModelDownloadUi({
    busy: false,
    available: false,
    statusText: 'Semantic model not available yet in this browser. Click "Download semantic model".',
  });

  return snapshot;
}

async function openCalendarImport(icsText, fileName, summary) {
  void summary;
  const blob = new Blob([icsText], { type: 'text/calendar;charset=utf-8' });
  const objectUrl = URL.createObjectURL(blob);
  const downloadLink = document.createElement('a');
  downloadLink.href = objectUrl;
  downloadLink.download = fileName;
  document.body.appendChild(downloadLink);
  downloadLink.click();
  downloadLink.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
  return 'downloaded';
}

function formatPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'n/a';
  return `${numeric.toFixed(2)}%`;
}

function scoreBreakdownText(row) {
  const mode = normalizeMatchingMode(row.relevance_mode || latestMatchingMode);
  const tfidf = Number(row.relevance_score_tfidf);
  const semantic = Number(row.relevance_score_semantic);

  if (mode === 'hybrid' && Number.isFinite(tfidf) && Number.isFinite(semantic)) {
    return `TF-IDF ${tfidf.toFixed(1)}% | Semantic ${semantic.toFixed(1)}%`;
  }
  if (mode === 'semantic' && Number.isFinite(tfidf)) {
    return `TF-IDF baseline ${tfidf.toFixed(1)}%`;
  }
  return '';
}

function isPastEvent(row) {
  const startMs = asEpochMs(row?.start_date_unix_ms);
  const endMs = asEpochMs(row?.end_date_unix_ms);
  const eventEndMs = endMs ?? startMs;
  return eventEndMs != null && eventEndMs < Date.now();
}

function semanticScholarSearchUrl(row) {
  const query = compactWhitespace(row?.title || '');
  const url = new URL('https://www.semanticscholar.org/search');
  url.searchParams.set('q', query || 'untitled paper');
  url.searchParams.set('sort', 'relevance');
  return url.toString();
}

function renderResults(rows, startIndex = 0) {
  resultsPanel.innerHTML = rows
    .map(
      (row, idx) => {
        const startMs = asEpochMs(row.start_date_unix_ms);
        const endMs = asEpochMs(row.end_date_unix_ms);
        const formattedStartDate = formatStartDate(startMs);
        const formattedTimeRange = formatTimeRange(startMs, endMs);
        const dayText = row.day || 'n/a';
        const dayClass = dayPillClass(dayText);
        const modeLabel = matchingModeLabel(normalizeMatchingMode(row.relevance_mode || latestMatchingMode));
        const scoreBreakdown = scoreBreakdownText(row);
        const isCalendarAdded = hasDownloadedCalendarItem(row);
        const isPastSession = isPastEvent(row);
        const conferenceLabel = compactWhitespace(row.conference_label) || selectedConferenceLabel();
        const paperSearchUrl = semanticScholarSearchUrl(row);
        return `
      <article class="result-card">
        <header>
          <p class="rank">#${row.relevance_rank}</p>
          <div class="score-wrap">
            <p class="score">${escapeHtml(formatPercent(row.relevance_score))}</p>
            <p class="score-mode">${escapeHtml(modeLabel)}</p>
            ${scoreBreakdown ? `<p class="score-breakdown">${escapeHtml(scoreBreakdown)}</p>` : ''}
          </div>
        </header>
        <h2>${escapeHtml(row.title || 'Untitled')}</h2>
        <p class="authors">${escapeHtml(row.authors || 'No authors listed')}</p>
        <dl class="facts">
          <div><dt>Conference</dt><dd>${escapeHtml(conferenceLabel || 'n/a')}</dd></div>
          <div><dt>Room</dt><dd>${escapeHtml(row.room || 'n/a')}</dd></div>
          <div><dt>Building</dt><dd>${escapeHtml(row.building || 'n/a')}</dd></div>
          <div><dt>Start</dt><dd>${escapeHtml(formattedStartDate)}</dd></div>
          <div><dt>Time</dt><dd>${escapeHtml(formattedTimeRange)}</dd></div>
          <div><dt>Day</dt><dd><span class="day-pill ${dayClass}">${escapeHtml(dayText)}</span></dd></div>
        </dl>
        <p class="abstract">${escapeHtml(row.abstract || 'No abstract provided.')}</p>
        <div class="card-actions">
          <button
            type="button"
            class="card-action-btn calendar-btn ${isCalendarAdded ? 'is-added' : ''} ${isPastSession ? 'is-past' : ''}"
            data-row-index="${startIndex + idx}"
            data-downloaded="${isCalendarAdded ? 'true' : 'false'}"
            ${isPastSession ? 'disabled aria-disabled="true" title="This event is in the past."' : ''}
          >
            Add to calendar (.ics)
            ${isCalendarAdded ? '<span class="calendar-check" aria-hidden="true">✓</span>' : ''}
          </button>
          <a
            class="card-action-btn paper-search-btn"
            href="${escapeHtml(paperSearchUrl)}"
            target="_blank"
            rel="noopener noreferrer"
          >
            Find paper
          </a>
        </div>
      </article>
    `
      }
    )
    .join('');
}

function updateLoadMoreControls() {
  const remaining = latestRows.length - visibleResults;
  if (remaining > 0) {
    const next = Math.min(LOAD_MORE_STEP, remaining);
    loadMoreBtn.textContent = `Load more (${next} more)`;
    resultsActions.classList.remove('hidden');
  } else {
    resultsActions.classList.add('hidden');
  }
}

function renderResultsPage() {
  renderResults(latestRows.slice(0, visibleResults), 0);
  updateLoadMoreControls();
}

async function runScoringFromExtracted(statusText, skipDoneStatus = false) {
  if (!extractedContext) {
    throw new Error('Upload PDFs, select abstracts, or add keywords, then run analysis.');
  }

  const runId = ++scoringRunSeq;
  const customKeywords = parseCustomKeywords(keywordInput.value);
  const matchingMode = normalizeMatchingMode(matchingModeInput?.value);
  const ownWorkExclusion = selectedOwnWorkExclusion();
  updateActiveScheduleRowCount();
  if (!scheduleRowCount) {
    throw new Error(`No schedule rows found for ${scoringScopeLabel()}.`);
  }
  if (modeNeedsSemanticModel(matchingMode) && !semanticModelAvailable) {
    throw new Error('Semantic model is not available. Click "Download semantic model" first.');
  }
  setStatus(statusText || scoringStatusText(matchingMode));

  const { result } = await runWorkerScore({
    worksTexts: extractedContext.worksTexts,
    workNames: extractedContext.workNames,
    customKeywords,
    matchingMode,
    conferenceKey: selectedConferenceKey,
    searchAcrossAllConferences,
    ownWorkExclusion,
  });

  if (runId !== scoringRunSeq) {
    return;
  }

  latestMatchingMode = normalizeMatchingMode(result.matching_mode || matchingMode);
  latestRows = result.rows || [];
  visibleResults = Math.min(DEFAULT_VISIBLE_RESULTS, latestRows.length || DEFAULT_VISIBLE_RESULTS);

  renderResultsPage();

  if (!skipDoneStatus) {
    const shown = Math.min(visibleResults, latestRows.length);
    const excludedOwnWorkCount = Number(result.excluded_own_work_count || 0);
    const exclusionText = excludedOwnWorkCount > 0
      ? ` Excluded ${excludedOwnWorkCount.toLocaleString()} own-work match${excludedOwnWorkCount === 1 ? '' : 'es'}.`
      : '';
    setStatus(
      `Done (${matchingModeLabel(latestMatchingMode)}, ${scoringScopeLabel()}). Showing ${shown.toLocaleString()} of ${latestRows.length.toLocaleString()} results.${exclusionText}`
    );
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

resultsPanel.addEventListener('click', async (event) => {
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;

  const calendarBtn = target.closest('.calendar-btn');
  if (!calendarBtn) return;

  const rowIndex = Number(calendarBtn.dataset.rowIndex);
  if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= latestRows.length) {
    return;
  }

  const row = latestRows[rowIndex];
  calendarBtn.disabled = true;

  try {
    const { icsText, fileName, summary } = buildCalendarPayload(row, rowIndex);
    const result = await openCalendarImport(icsText, fileName, summary);
    if (result === 'cancelled') return;
    markCalendarItemDownloaded(row);
    setCalendarButtonDownloadedState(calendarBtn, true);
    if (!calendarBtn.querySelector('.calendar-check')) {
      const check = document.createElement('span');
      check.className = 'calendar-check';
      check.setAttribute('aria-hidden', 'true');
      check.textContent = '✓';
      calendarBtn.appendChild(check);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    appendStatusItem(`Calendar error: ${message}`);
  } finally {
    calendarBtn.disabled = false;
  }
});

boostedKeywordsList.addEventListener('click', (event) => {
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;
  const chip = target.closest('.boosted-keyword-chip');
  if (!chip) return;

  const keyword = String(chip.dataset.keyword || '').trim();
  if (!keyword) return;

  const keywordLower = keyword.toLowerCase();
  const nextKeywords = parseCustomKeywords(keywordInput.value).filter((entry) => entry.toLowerCase() !== keywordLower);
  keywordInput.value = nextKeywords.join(', ');
  keywordInput.dispatchEvent(new Event('input', { bubbles: true }));
});

if (fetchProfileBtn) {
  fetchProfileBtn.addEventListener('click', async () => {
    const runSeq = ++profileFetchRunSeq;
    try {
      const { authorQuery } = resolveAuthorLookupInput(profileAuthorQueryInput?.value);
      const lookupKey = buildAuthorLookupKey(authorQuery);
      const shouldRefreshCandidates = lookupKey !== activeAuthorLookupKey || !fetchedAuthorCandidates.length;

      fetchProfileBtn.disabled = true;

      if (shouldRefreshCandidates) {
        clearFetchedProfileAbstracts();
        updateProfileFetchStatus('Finding author matches...');
        fetchedAuthorCandidates = await getAuthorCandidatesForQuery(authorQuery);
        if (runSeq !== profileFetchRunSeq) return;
        selectedAuthorCandidateId = fetchedAuthorCandidates[0]?.id || '';
        activeAuthorLookupKey = lookupKey;
        renderAuthorCandidates();
      }

      const selectedCandidate = getSelectedAuthorCandidate();
      if (!selectedCandidate) {
        clearFetchedProfileAbstracts();
        clearAuthorCandidates({ keepLookupKey: true });
        updateProfileFetchStatus('No matching OpenAlex author profiles found. Enter a more specific author name.', true);
        return;
      }

      await fetchAndRenderAbstractsForSelectedCandidate({ runSeq });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      clearFetchedProfileAbstracts();
      updateProfileFetchStatus(message, true);
    } finally {
      if (runSeq === profileFetchRunSeq) {
        fetchProfileBtn.disabled = false;
      }
    }
  });
}

if (profileAuthorCandidatesList) {
  profileAuthorCandidatesList.addEventListener('change', async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (target.type !== 'radio') return;
    const authorId = String(target.dataset.authorId || '');
    if (!authorId) return;
    selectedAuthorCandidateId = authorId;
    updateAuthorCandidatesCount();
    const runSeq = ++profileFetchRunSeq;
    if (fetchProfileBtn) {
      fetchProfileBtn.disabled = true;
    }

    try {
      await fetchAndRenderAbstractsForSelectedCandidate({ runSeq });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      clearFetchedProfileAbstracts();
      updateProfileFetchStatus(message, true);
    } finally {
      if (fetchProfileBtn && runSeq === profileFetchRunSeq) {
        fetchProfileBtn.disabled = false;
      }
    }
  });
}

if (profileAuthorQueryInput) {
  profileAuthorQueryInput.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    event.stopPropagation();
    if (!fetchProfileBtn?.disabled) {
      fetchProfileBtn?.click();
    }
  });

  profileAuthorQueryInput.addEventListener('input', () => {
    profileFetchRunSeq += 1;
    if (fetchProfileBtn) {
      fetchProfileBtn.disabled = false;
    }
    clearAuthorCandidates();
    clearFetchedProfileAbstracts();
    updateProfileFetchStatus('Find your profile and fetch paper abstracts.');
  });
}

if (selectAllAbstractsBtn) {
  selectAllAbstractsBtn.addEventListener('click', () => {
    selectedProfileAbstractIds.clear();
    for (const item of fetchedProfileAbstracts) {
      selectedProfileAbstractIds.add(item.id);
    }
    renderProfileAbstracts();
  });
}

if (toggleAbstractPreviewBtn) {
  toggleAbstractPreviewBtn.addEventListener('click', () => {
    if (!fetchedProfileAbstracts.length) return;
    profileAbstractsExpanded = !profileAbstractsExpanded;
    renderProfileAbstracts();
  });
}

if (clearAllAbstractsBtn) {
  clearAllAbstractsBtn.addEventListener('click', () => {
    selectedProfileAbstractIds.clear();
    renderProfileAbstracts();
  });
}

if (profileAbstractsList) {
  profileAbstractsList.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    const id = String(target.dataset.abstractId || '');
    if (!id) return;
    if (target.checked) {
      selectedProfileAbstractIds.add(id);
    } else {
      selectedProfileAbstractIds.delete(id);
    }
    updateProfileAbstractSelectionCount();
  });
}

fileInput.addEventListener('change', () => {
  updateFileSelectionText();
});

if (clearPdfsBtn) {
  clearPdfsBtn.addEventListener('click', () => {
    fileInput.value = '';
    updateFileSelectionText();
  });
}

loadMoreBtn.addEventListener('click', () => {
  visibleResults = Math.min(latestRows.length, visibleResults + LOAD_MORE_STEP);
  renderResultsPage();
});

if (downloadModelBtn) {
  downloadModelBtn.addEventListener('click', async () => {
    if (modelDownloadBusy) return;

    const availability = await refreshModelCacheUi({ forceUi: true });
    if (availability.available) {
      setStatus('Semantic model is already available.');
      return;
    }

    setModelDownloadUi({
      busy: true,
      available: false,
      statusText: 'Starting semantic model download...',
      progressPercent: 0,
    });

    try {
      const task = startWorkerPrefetchModel();
      modelDownloadRequestId = task.requestId;
      await task.promise;
      const updated = await refreshModelCacheUi({ forceUi: true });
      if (updated.available) {
        setStatus('Semantic model is ready for Semantic/Hybrid matching.');
      } else {
        setStatus('Model download finished, but readiness could not be confirmed. Try Download again.');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setModelDownloadUi({
        busy: false,
        available: false,
        statusText: `Model download failed: ${message}`,
      });
      setStatus(message);
      appendStatusItem(`Error: ${message}`);
    } finally {
      modelDownloadRequestId = null;
      if (modelDownloadBusy) {
        modelDownloadBusy = false;
        updateRunButtonAvailability();
      }
    }
  });
}

window.addEventListener('focus', () => {
  void refreshModelCacheUi();
});

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    void refreshModelCacheUi();
  }
});

keywordInput.addEventListener('input', () => {
  renderBoostedKeywords();
  if (keywordDebounceHandle) {
    window.clearTimeout(keywordDebounceHandle);
  }

  keywordDebounceHandle = window.setTimeout(async () => {
    if (!extractedContext || isExtracting) return;
    const mode = normalizeMatchingMode(matchingModeInput?.value);
    if (modeNeedsSemanticModel(mode) && !semanticModelAvailable) return;

    try {
      await runScoringFromExtracted(null, true);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(message);
      appendStatusItem(`Error: ${message}`);
    }
  }, KEYWORD_DEBOUNCE_MS);
});

if (matchingModeInput) {
  matchingModeInput.addEventListener('change', async () => {
    updateRunButtonAvailability();
    if (!extractedContext || isExtracting) return;
    const mode = normalizeMatchingMode(matchingModeInput.value);
    if (modeNeedsSemanticModel(mode) && !semanticModelAvailable) {
      setStatus('Semantic model is not available. Download it first or deploy bundled /models files.');
      return;
    }

    try {
      await runScoringFromExtracted(`Recomputing with ${matchingModeLabel(mode)} matching...`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(message);
      appendStatusItem(`Error: ${message}`);
    }
  });
}

if (conferenceSelect) {
  conferenceSelect.addEventListener('pointerdown', () => {
    setConferenceSelectCountVisibility(true);
  });

  conferenceSelect.addEventListener('keydown', (event) => {
    if (['ArrowDown', 'ArrowUp', ' ', 'Enter'].includes(event.key)) {
      setConferenceSelectCountVisibility(true);
    }
  });

  conferenceSelect.addEventListener('change', () => {
    setConferenceSelectCountVisibility(false);
    syncSelectContentWidth(conferenceSelect);
    void rerunForConferenceScopeChange();
  });

  conferenceSelect.addEventListener('blur', () => {
    setConferenceSelectCountVisibility(false);
    syncSelectContentWidth(conferenceSelect);
  });
}

if (timeScopeSelect) {
  timeScopeSelect.addEventListener('change', () => {
    timeScopeSelect.value = FIXED_TIME_SCOPE_KEY;
    syncSelectContentWidth(timeScopeSelect);
    void rerunForConferenceScopeChange();
  });
}

if (document.fonts?.ready) {
  void document.fonts.ready.then(syncTitleSelectWidths);
}

if (searchAllConferencesToggle) {
  searchAllConferencesToggle.addEventListener('change', () => {
    void rerunForConferenceScopeChange();
  });
}

if (excludeSubstringMatchesInput) {
  excludeSubstringMatchesInput.addEventListener('change', () => {
    void rerunForOwnWorkExclusionChange();
  });
}

if (substringMatchWordCountSelect) {
  substringMatchWordCountSelect.addEventListener('change', () => {
    void rerunForOwnWorkExclusionChange();
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearStatusItems();
  clearResults();

  try {
    const files = Array.from(fileInput.files || []);
    const selectedAbstracts = getSelectedProfileAbstracts();
    const customKeywords = parseCustomKeywords(keywordInput?.value || '');
    validateUploads(files, { allowEmpty: selectedAbstracts.length > 0 || customKeywords.length > 0 });

    runBtn.disabled = true;
    isExtracting = true;
    extractedContext = null;

    let worksTexts = [];
    const workNames = [];

    if (files.length) {
      setStatus('Extracting text from PDFs with pdf.js...');
      const pdfWorksTexts = await extractAllPdfs(files);
      worksTexts = worksTexts.concat(pdfWorksTexts);
      workNames.push(...files.map((f) => f.name));
    }

    if (selectedAbstracts.length) {
      const prefix = files.length ? 'Combining selected profile abstracts with extracted PDF text...' : 'Preparing selected profile abstracts...';
      setStatus(prefix);
      worksTexts = worksTexts.concat(selectedAbstracts.map((item) => item.abstract));
      workNames.push(
        ...selectedAbstracts.map((item, idx) => `profile_abstract_${idx + 1}: ${item.title || 'Untitled'}`)
      );
    }

    extractedContext = {
      worksTexts,
      workNames,
    };

    isExtracting = false;
    await runScoringFromExtracted();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    appendStatusItem(`Error: ${message}`);
  } finally {
    isExtracting = false;
    updateRunButtonAvailability();
  }
});

(async () => {
  initializeSourceTabs();
  renderBoostedKeywords();
  renderProfileAbstracts();
  updateFileSelectionText();
  selectedOwnWorkExclusion();
  latestMatchingMode = normalizeMatchingMode(matchingModeInput?.value);
  await refreshModelCacheUi();
  updateRunButtonAvailability();
  try {
    const scheduleIndex = await loadScheduleIndex();
    initializeConferenceControls(scheduleIndex);
    setStatus(`Loaded schedule manifest for ${scoringScopeLabel()} (${scheduleRowCount.toLocaleString()} rows). Ready.`);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error));
  }
})();
