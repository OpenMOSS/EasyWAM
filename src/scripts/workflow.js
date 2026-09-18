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


const benchmarkButtons = [...document.querySelectorAll('[data-benchmark]')];
const evaluateCode = document.getElementById('code-evaluate');
benchmarkButtons.forEach((button) => button.addEventListener('click', () => {
  benchmarkButtons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
  evaluateCode.textContent = evaluationCommands[button.dataset.benchmark].join('\n');
}));
