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
      'Start': { text: '开始', href: '/guguji_isaaclab/zh/getting-started/' },
      'Train / Eval': { text: '训练评估', href: '/guguji_isaaclab/zh/training/' },
      'Design': { text: '设计', href: '/guguji_isaaclab/zh/design/' },
      'Envs': { text: '环境', href: '/guguji_isaaclab/zh/tested-environments/' },
      'Contributors': { text: '贡献者', href: '/guguji_isaaclab/zh/contributors/' },
      'Changelog': { text: '更新', href: '/guguji_isaaclab/zh/changelog/' },
      'Search': { text: '搜索' },
      'Edit on osrbot/guguji_isaaclab': { text: '源码' },
      'osrbot/guguji_isaaclab': { text: '源码' },
    };

    navLinks.forEach((link) => {
      const label = link.textContent.trim();
      if (label === 'Previous' || label === 'Next') {
        link.parentElement?.remove();
        return;
      }
      if (map[label]) {
        const next = isZh ? map[label].text : label;
        link.textContent = next;
        if (isZh && map[label].href) link.setAttribute('href', map[label].href);
      }
    });

    document.querySelectorAll('.navbar-right a').forEach((link) => {
      const label = link.textContent.trim();
      if (/Edit on/i.test(label) || /osrbot\/guguji_isaaclab/.test(label)) {
        link.textContent = isZh ? '源码' : 'GitHub';
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

    const isDrawerOpen = () =>
      collapse.classList.contains('show') || collapse.classList.contains('in');

    const syncState = () => {
      if (!isMobile()) {
        body.classList.remove('gg-mobile-nav-open');
        collapse.classList.remove('show', 'in');
        collapse.setAttribute('aria-hidden', 'false');
        toggle.setAttribute('aria-expanded', 'false');
        return;
      }

      const isOpen = isDrawerOpen();
      body.classList.toggle('gg-mobile-nav-open', isOpen);
      collapse.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
      toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    };

    const openDrawer = () => {
      if (!isMobile()) return;
      collapse.classList.add('show', 'in');
      syncState();
    };

    const closeDrawer = () => {
      collapse.classList.remove('show', 'in');
      syncState();
    };

    toggle.addEventListener('click', (event) => {
      if (!isMobile()) return;
      event.preventDefault();
      event.stopPropagation();

      if (isDrawerOpen()) {
        closeDrawer();
      } else {
        openDrawer();
      }
    });

    overlay.addEventListener('click', closeDrawer);

    document.querySelectorAll('.navbar-collapse a').forEach((anchor) => {
      anchor.addEventListener('click', () => {
        if (!isMobile()) return;
        closeDrawer();
      });
    });

    const handleMqChange = () => syncState();
    if (typeof mobileMq.addEventListener === 'function') {
      mobileMq.addEventListener('change', handleMqChange);
    } else if (typeof mobileMq.addListener === 'function') {
      mobileMq.addListener(handleMqChange);
    }

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
