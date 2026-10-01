export function estimateTrainingRemainingSeconds(job, now = Date.now()) {
  if (job?.status === "completed") return 0;
  if (job?.status === "failed" || job?.tracking_paused) return null;
  if (job?.phase !== "training") return null;
  const batches = Number(job.completed_batches);
  const totalBatches = Number(job.total_batches);
  const batchStartedAt = Date.parse(job.phase_started_at || "");
  if (batches > 0 && totalBatches > 0 && Number.isFinite(batchStartedAt) && now >= batchStartedAt) {
    return Math.max(0, Math.ceil(((now - batchStartedAt) / 1000 / batches) * (totalBatches - batches)));
  }
  const epoch = Number(job.epoch);
  const maxEpochs = Number(job.max_epochs);
  const startedAt = Date.parse(job.phase_started_at || "");
  if (!Number.isFinite(epoch) || epoch < 1 || !Number.isFinite(maxEpochs)
      || maxEpochs < 1 || !Number.isFinite(startedAt) || now < startedAt) return null;
  const elapsedSeconds = (now - startedAt) / 1000;
  return Math.max(0, Math.ceil((elapsedSeconds / epoch) * Math.max(0, maxEpochs - epoch)));
}

export function formatRemainingTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) return `${Math.ceil(seconds)} s`;
  const minutes = Math.ceil(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes ? `${hours} h ${remainingMinutes} min` : `${hours} h`;
}
