document.addEventListener('DOMContentLoaded', () => {
  const options = document.querySelectorAll('.billing-option');
  const amounts = document.querySelectorAll('.amount');
  const billed = document.querySelectorAll('.billed');
  if (!options.length) return;

  function setPeriod(period) {
    options.forEach((btn) => {
      const active = btn.dataset.period === period;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', String(active));
    });
    amounts.forEach((el) => { el.textContent = '$' + el.dataset[period]; });
    billed.forEach((el) => { el.textContent = el.dataset[period]; });
  }

  options.forEach((btn) => btn.addEventListener('click', () => setPeriod(btn.dataset.period)));
});
