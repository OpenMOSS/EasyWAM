const copyFeedback = document.getElementById('copy-feedback');
let feedbackTimer;
async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(value); return true; } catch { /* Try legacy copy. */ }
  }
  const fallback = document.createElement('textarea');
  fallback.value = value;
  fallback.setAttribute('readonly', '');
  fallback.style.position = 'fixed';
  fallback.style.opacity = '0';
  document.body.appendChild(fallback);
  try {
    fallback.select();
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    fallback.remove();
  }
}
document.querySelectorAll('.copy-code').forEach((button) => button.addEventListener('click', async () => {
  const code = document.getElementById(`code-${button.dataset.copy}`).textContent.trim();
  const copied = await copyText(code);
  copyFeedback.textContent = copied
    ? (currentLanguage === 'zh' ? '命令已复制' : 'Command copied')
    : (currentLanguage === 'zh' ? '复制失败，请手动选择命令' : 'Copy failed; select the command manually');
  window.clearTimeout(feedbackTimer);
  feedbackTimer = window.setTimeout(() => { copyFeedback.textContent = ''; }, 2600);
}));
