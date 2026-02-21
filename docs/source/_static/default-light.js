// Default to light mode on first visit.
// Furo uses localStorage key "__theme" — if unset, it follows OS preference.
// This script sets it to "light" only when no prior preference exists.
if (!localStorage.getItem("__theme")) {
  localStorage.setItem("__theme", "light");
  document.body.dataset.theme = "light";
}
