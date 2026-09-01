const revealItems = document.querySelectorAll('.reveal');
const revealObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add('is-visible');
    observer.unobserve(entry.target);
  });
}, { threshold: 0.08 });
revealItems.forEach((item) => revealObserver.observe(item));

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

document.querySelectorAll('.filter').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.filter').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    const filter = button.dataset.filter;
    document.querySelectorAll('.post').forEach((post) => {
      post.hidden = filter !== 'all' && post.dataset.category !== filter;
    });
  });
});

const copyButton = document.querySelector('#copy-command');
const copyStatus = document.querySelector('#copy-status');
copyButton?.addEventListener('click', async () => {
  const command = 'git clone https://github.com/OpenMOSS/EasyWAM && cd EasyWAM && pip install -e .';
  try {
    await navigator.clipboard.writeText(command);
    copyStatus.textContent = 'Copied to clipboard';
  } catch {
    copyStatus.textContent = 'Select and copy the command';
  }
  window.setTimeout(() => { copyStatus.textContent = ''; }, 2400);
});
