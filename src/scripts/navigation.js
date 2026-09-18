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
