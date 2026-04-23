document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;
  const isZh = /\/zh\//.test(path) || path.endsWith('/zh');

  const addHeroClickEffects = () => {
    const links = document.querySelectorAll('.hero-button');
    for (const link of links) {
      link.addEventListener('click', () => {
        link.classList.add('is-clicked');
        setTimeout(() => link.classList.remove('is-clicked'), 180);
      });
    }
  };

  const normalizePath = (value) => value.replace(/\/+/g, '/');

  const getEnglishPath = () => {
    if (path === '/guguji_isaaclab/zh/' || path.endsWith('/guguji_isaaclab/zh')) return '/guguji_isaaclab/';
    return normalizePath(path.replace('/guguji_isaaclab/zh/', '/guguji_isaaclab/'));
  };

  const getChinesePath = () => {
    if (path === '/guguji_isaaclab/' || path.endsWith('/guguji_isaaclab')) return '/guguji_isaaclab/zh/';
    if (isZh) return path;
    return normalizePath(path.replace('/guguji_isaaclab/', '/guguji_isaaclab/zh/'));
  };

  const createLanguageSwitcher = () => {
    const navbarCollapse = document.querySelector('.navbar-collapse');
    const navbar = document.querySelector('.navbar .container');
    const mountPoint = navbarCollapse || navbar;
    if (!mountPoint || mountPoint.querySelector('.gg-lang-switch')) return;

    const switcher = document.createElement('div');
    switcher.className = 'gg-lang-switch';

    const en = document.createElement('a');
    en.href = getEnglishPath();
    en.textContent = 'EN';
    en.className = `gg-lang-pill ${isZh ? '' : 'is-active'}`.trim();

    const zh = document.createElement('a');
    zh.href = getChinesePath();
    zh.textContent = '中文';
    zh.className = `gg-lang-pill ${isZh ? 'is-active' : ''}`.trim();

    switcher.appendChild(en);
    switcher.appendChild(zh);
    mountPoint.appendChild(switcher);
  };

  const localizeNavbar = () => {
    const navLinks = document.querySelectorAll('.navbar-nav a');
    const map = {
      'Home': { text: '首页', href: '/guguji_isaaclab/zh/' },
      'Getting Started': { text: '快速开始', href: '/guguji_isaaclab/zh/getting-started/' },
      'Training & Evaluation': { text: '训练与评估', href: '/guguji_isaaclab/zh/training/' },
      'Design Notes': { text: '设计说明', href: '/guguji_isaaclab/zh/design/' },
      'Changelog': { text: '更新日志', href: '/guguji_isaaclab/zh/changelog/' },
      'Search': { text: '搜索' },
    };

    navLinks.forEach((link) => {
      const label = link.textContent.trim();
      if (label === 'Previous' || label === 'Next') {
        link.parentElement?.remove();
        return;
      }
      if (isZh && map[label]) {
        link.textContent = map[label].text;
        if (map[label].href) link.setAttribute('href', map[label].href);
      }
    });

    const brand = document.querySelector('.navbar-brand');
    if (brand && isZh) {
      brand.textContent = 'Guguji Isaac Lab';
      brand.setAttribute('href', '/guguji_isaaclab/zh/');
    }
  };

  const setupMobileDrawer = () => {
    const collapse = document.querySelector('.navbar-collapse');
    const toggle = document.querySelector('.navbar-toggle');
    const body = document.body;
    if (!collapse || !toggle) return;

    let overlay = document.querySelector('.gg-mobile-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.className = 'gg-mobile-overlay';
      body.appendChild(overlay);
    }

    const mobileMq = window.matchMedia('(max-width: 1100px)');
    const isMobile = () => mobileMq.matches;

    const syncState = () => {
      const isOpen = isMobile() && collapse.classList.contains('in');
      body.classList.toggle('gg-mobile-nav-open', isOpen);
      collapse.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
      toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    };

    const openDrawer = () => {
      if (!isMobile()) return;
      collapse.classList.add('in');
      syncState();
    };

    const closeDrawer = () => {
      collapse.classList.remove('in');
      syncState();
    };

    const toggleDrawer = (event) => {
      if (!isMobile()) return;
      event.preventDefault();
      if (collapse.classList.contains('in')) {
        closeDrawer();
      } else {
        openDrawer();
      }
    };

    toggle.addEventListener('click', toggleDrawer);
    overlay.addEventListener('click', closeDrawer);

    document.querySelectorAll('.navbar-collapse a').forEach((anchor) => {
      anchor.addEventListener('click', () => {
        if (!isMobile()) return;
        closeDrawer();
      });
    });

    mobileMq.addEventListener('change', () => {
      if (!isMobile()) {
        collapse.classList.remove('in');
      }
      syncState();
    });

    window.addEventListener('resize', syncState);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') closeDrawer();
    });

    syncState();
  };

  addHeroClickEffects();
  localizeNavbar();
  createLanguageSwitcher();
  setupMobileDrawer();
});
