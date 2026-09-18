const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window) {
  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.08 });
  revealItems.forEach((item) => revealObserver.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('is-visible'));
}

const sections = [...document.querySelectorAll('main section[id]')];
const navLinks = [...document.querySelectorAll('.side-link, .mobile-nav')];
const updateActive = () => {
  const current = sections.reduce((active, section) => {
    if (section.getBoundingClientRect().top <= 170) return section.id;
    return active;
  }, sections[0]?.id);
  navLinks.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${current}`));
};
window.addEventListener('scroll', updateActive, { passive: true });
updateActive();

const translatedNodes = [...document.querySelectorAll('[data-zh]')];
translatedNodes.forEach((node) => { node.dataset.en = node.textContent.trim(); });
const tableLabels = {
  'Model': '模型', 'Spatial': '空间', 'Object': '物体', 'Goal': '目标',
  'LIBERO-10': 'LIBERO-10', 'Avg.': '平均', 'Orig': '原始',
  'Background': '背景', 'Camera': '相机', 'Language': '语言',
  'Layout': '布局', 'Light': '光照', 'Noise': '噪声', 'Robot': '机器人',
};
const tableHeaders = [...document.querySelectorAll('.benchmark-table th')];
tableHeaders.forEach((header) => { header.dataset.en = header.textContent.trim(); });
const tableNumbers = [...document.querySelectorAll('.perf-table-wrap figcaption > span:first-child')];
const languageButtons = [...document.querySelectorAll('.language-switch button')];
const translatedImages = [...document.querySelectorAll('[data-alt-zh]')];
const translatedAriaLabels = [...document.querySelectorAll('[data-aria-zh]')];
translatedImages.forEach((image) => { image.dataset.altEn = image.alt; });
translatedAriaLabels.forEach((node) => { node.dataset.ariaEn = node.getAttribute('aria-label'); });
let currentLanguage = 'en';

function setLanguage(language) {
  currentLanguage = language === 'zh' ? 'zh' : 'en';
  document.documentElement.lang = currentLanguage === 'zh' ? 'zh-CN' : 'en';
  document.title = currentLanguage === 'zh'
    ? 'EasyWAM：世界动作模型训练与评测框架'
    : 'EasyWAM: A Unified and Efficient Framework for Training and Evaluating World Action Models';
  document.querySelector('meta[name="description"]').content = currentLanguage === 'zh'
    ? 'EasyWAM 是高效统一的世界动作模型训练与评测框架。'
    : 'EasyWAM is a unified, efficient framework for World Action Model research.';
  translatedNodes.forEach((node) => {
    node.textContent = currentLanguage === 'zh' ? node.dataset.zh : node.dataset.en;
  });
  translatedImages.forEach((image) => {
    image.alt = currentLanguage === 'zh' ? image.dataset.altZh : image.dataset.altEn;
  });
  translatedAriaLabels.forEach((node) => {
    node.setAttribute('aria-label', currentLanguage === 'zh' ? node.dataset.ariaZh : node.dataset.ariaEn);
  });
  tableHeaders.forEach((header) => {
    header.textContent = currentLanguage === 'zh'
      ? (tableLabels[header.dataset.en] || header.dataset.en)
      : header.dataset.en;
  });
  tableNumbers.forEach((number) => {
    number.textContent = currentLanguage === 'zh'
      ? number.dataset.en.replace('TABLE', '表')
      : number.dataset.en;
  });
  languageButtons.forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.lang === currentLanguage));
  });
  try { localStorage.setItem('easywam-language', currentLanguage); } catch { /* Storage may be disabled. */ }
}
tableNumbers.forEach((number) => { number.dataset.en = number.textContent.trim(); });
languageButtons.forEach((button) => button.addEventListener('click', () => setLanguage(button.dataset.lang)));
let savedLanguage = 'en';
try { savedLanguage = localStorage.getItem('easywam-language') || 'en'; } catch { /* Keep English. */ }
setLanguage(savedLanguage);

const workflowTabs = [...document.querySelectorAll('.workflow-tab')];
function activateWorkflow(tab, focus = false) {
  workflowTabs.forEach((item) => {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute('aria-controls')).hidden = !selected;
  });
  if (focus) tab.focus();
}
workflowTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateWorkflow(tab));
  tab.addEventListener('keydown', (event) => {
    let next = index;
    if (event.key === 'ArrowRight') next = (index + 1) % workflowTabs.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + workflowTabs.length) % workflowTabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = workflowTabs.length - 1;
    else return;
    event.preventDefault();
    activateWorkflow(workflowTabs[next], true);
  });
});

const evaluationCommands = {
  libero: [
    'python experiments/libero/run_libero_manager.py \\',
    '  task=libero_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt>',
  ],
  'libero-plus': [
    'python experiments/libero_plus/run_libero_plus_manager.py \\',
    '  task=libero_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt>',
  ],
  robotwin: [
    'python experiments/robotwin/run_robotwin_manager.py \\',
    '  task=robotwin_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt>',
  ],
  robodojo: [
    'python experiments/robodojo/run_robodojo_manager.py \\',
    '  task=robodojo_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt>',
  ],
  robocasa: [
    'python experiments/robocasa/run_robocasa_manager.py \\',
    '  task=robocasa_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt> \\',
    '  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json>',
  ],
};
const benchmarkButtons = [...document.querySelectorAll('[data-benchmark]')];
const evaluateCode = document.getElementById('code-evaluate');
benchmarkButtons.forEach((button) => button.addEventListener('click', () => {
  benchmarkButtons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
  evaluateCode.textContent = evaluationCommands[button.dataset.benchmark].join('\n');
}));

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
