<script src="https://cdn.jsdelivr.net/npm/mermaid@11.6.0/dist/mermaid.min.js"></script>

<script> 
function refreshTime() {
  var timeDisplay = document.getElementById("mel-local-time");
  var timeString = new Date().toLocaleTimeString("en-US", {timeZone: "Australia/Melbourne"});
  timeDisplay.innerHTML = timeString;
}

setInterval(refreshTime, 1000);
</script>

<script>
function updateAncestor(el) {
  const ancestor = el.closest('.cell-output.cell-output-display');
  if (!ancestor) return;

  // Only check the form's own display
  const formDisplay = getComputedStyle(el).display;
  const formHidden = formDisplay === 'none';

  ancestor.style.display = formHidden ? 'none' : 'inline-block';
}

// Continuous monitoring
setInterval(() => {
  document.querySelectorAll('.oi-3a86ea').forEach(updateAncestor);
}, 500);

</script>

<script>
function readAttrOrData(el, name) {
  const attr = el.getAttribute(name);
  if (attr !== null) return attr;

  const dataAttr = el.getAttribute('data-' + name);
  if (dataAttr !== null) return dataAttr;

  const camel = name.replace(/-([a-z])/g, g => g[1].toUpperCase());
  if (el.dataset && el.dataset[camel] !== undefined && el.dataset[camel] !== '') {
    return el.dataset[camel];
  }
  return null;
}

function ensurePopup(el) {
  if (el.querySelector('.popup-box')) return;

  const inner = el.querySelector('.popup-content');
  const tooltip = document.createElement('div');
  tooltip.className = 'popup-box';

  if (inner) {
    inner.style.display = 'none';
    tooltip.innerHTML = inner.innerHTML;
  } else {
    const inline = readAttrOrData(el, 'popup-content');
    if (!inline) return;
    tooltip.textContent = inline;
  }

  // Read positioning attributes
  const dir = readAttrOrData(el, 'popup-dir') || 'top';
  const align = readAttrOrData(el, 'popup-align') || 'start';
  const position = readAttrOrData(el, 'popup-position') || 'inner'; // outer or inner

  // Compute position dynamically
  const offsetOuter = 6;  // distance from edge when outer
  const offsetInner = 6;  // distance from edge when inner
  const parentRect = el.getBoundingClientRect();

  // Reset all positioning
  tooltip.style.top = '';
  tooltip.style.bottom = '';
  tooltip.style.left = '';
  tooltip.style.right = '';
  tooltip.style.transform = '';

  let offset = position === 'inner' ? '70%' : offsetOuter + 'px';

  if (dir === 'top') {
    tooltip.style.bottom = position === 'inner' ? offset : '100%';
    tooltip.style.left = align === 'center' ? '50%' : (align === 'end' ? 'auto' : '0');
    if (align === 'center') tooltip.style.transform = 'translateX(-50%)';
    if (align === 'end') tooltip.style.right = '0';
  } else if (dir === 'bottom') {
    tooltip.style.top = position === 'inner' ? offset : '100%';
    tooltip.style.left = align === 'center' ? '50%' : (align === 'end' ? 'auto' : '0');
    if (align === 'center') tooltip.style.transform = 'translateX(-50%)';
    if (align === 'end') tooltip.style.right = '0';
  } else if (dir === 'left') {
    tooltip.style.right = position === 'inner' ? offset : '100%';
    tooltip.style.top = align === 'center' ? '50%' : (align === 'end' ? 'auto' : '0');
    if (align === 'center') tooltip.style.transform = 'translateY(-50%)';
    if (align === 'end') tooltip.style.bottom = '0';
  } else if (dir === 'right') {
    tooltip.style.left = position === 'inner' ? offset : '100%';
    tooltip.style.top = align === 'center' ? '50%' : (align === 'end' ? 'auto' : '0');
    if (align === 'center') tooltip.style.transform = 'translateY(-50%)';
    if (align === 'end') tooltip.style.bottom = '0';
  }

  // Apply style attributes
  const map = {
    'popup-bg': 'backgroundColor',
    'popup-color': 'color',
    'popup-font-size': 'fontSize',
    'popup-width': 'maxWidth',
    'popup-radius': 'borderRadius',
    'popup-padding': 'padding',
    'popup-opacity': 'opacity',
    'popup-border': 'border',
    'popup-z': 'zIndex',
    'popup-wrap': 'whiteSpace'
  };

  for (const attr in map) {
    const val = readAttrOrData(el, attr);
    if (val) tooltip.style[map[attr]] = val;
  }

  el.appendChild(tooltip);
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.popup').forEach(ensurePopup);
});



</script>