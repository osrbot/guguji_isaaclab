document.addEventListener('DOMContentLoaded', () => {
  const links = document.querySelectorAll('.hero-button');
  for (const link of links) {
    link.addEventListener('click', () => {
      link.classList.add('is-clicked');
      setTimeout(() => link.classList.remove('is-clicked'), 180);
    });
  }
});
