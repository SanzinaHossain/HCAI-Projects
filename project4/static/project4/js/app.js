(function () {
  const list = document.getElementById('rankingList');
  if (!list) return;

  let dragged = null;

  function items() {
    return [...list.querySelectorAll('.rank-item')];
  }

  function renumber() {
    items().forEach((el, i) => {
      const number = el.querySelector('.rank-number');
      number.textContent = i + 1;
      number.setAttribute('aria-label', 'Current rank ' + (i + 1));

      const up = el.querySelector('[data-move="up"]');
      const down = el.querySelector('[data-move="down"]');
      up.disabled = i === 0;
      down.disabled = i === items().length - 1;
    });
  }

  function moveItem(el, direction) {
    if (!el) return;
    if (direction === 'up' && el.previousElementSibling) {
      list.insertBefore(el, el.previousElementSibling);
    }
    if (direction === 'down' && el.nextElementSibling) {
      list.insertBefore(el.nextElementSibling, el);
    }
    renumber();
    el.focus();
  }

  list.addEventListener('click', (e) => {
    const button = e.target.closest('.move-btn');
    if (!button) return;
    moveItem(button.closest('.rank-item'), button.dataset.move);
  });

  list.addEventListener('keydown', (e) => {
    const el = e.target.closest('.rank-item');
    if (!el || !e.altKey) return;
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      moveItem(el, 'up');
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      moveItem(el, 'down');
    }
  });

  list.addEventListener('dragstart', (e) => {
    dragged = e.target.closest('.rank-item');
    if (dragged) dragged.classList.add('dragging');
  });

  list.addEventListener('dragend', () => {
    if (dragged) dragged.classList.remove('dragging');
    dragged = null;
    renumber();
  });

  list.addEventListener('dragover', (e) => {
    e.preventDefault();
    const after = items()
      .filter((el) => !el.classList.contains('dragging'))
      .find((el) => e.clientY <= el.getBoundingClientRect().top + el.offsetHeight / 2);
    if (!dragged) return;
    if (after) list.insertBefore(dragged, after);
    else list.appendChild(dragged);
  });

  const form = document.getElementById('rankForm');
  form.addEventListener('submit', () => {
    document.getElementById('rankingInput').value = items().map((x) => x.dataset.id).join(',');
  });

  renumber();
})();
