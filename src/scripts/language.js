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
