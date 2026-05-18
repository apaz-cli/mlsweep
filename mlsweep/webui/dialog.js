// Custom styled confirm/alert dialogs that match the mlsweep UI theme.
(function () {
  const css = `
.ml-dialog-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.45);
  display: flex; align-items: center; justify-content: center; z-index: 1000;
  animation: ml-fade-in 0.1s ease;
}
@keyframes ml-fade-in { from { opacity: 0; } to { opacity: 1; } }
.ml-dialog {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 20px 24px; width: 340px; max-width: 95vw;
  display: flex; flex-direction: column; gap: 16px;
  animation: ml-slide-in 0.1s ease;
}
@keyframes ml-slide-in { from { transform: translateY(-6px); opacity: 0; } to { transform: none; opacity: 1; } }
.ml-dialog-message {
  font-size: 13px; color: var(--text); line-height: 1.5; white-space: pre-wrap;
  font-family: system-ui, -apple-system, sans-serif;
}
.ml-dialog-actions { display: flex; justify-content: flex-end; gap: 8px; }
.ml-btn {
  font-size: 12px; padding: 5px 14px; border-radius: 4px; cursor: pointer;
  font-family: system-ui, -apple-system, sans-serif;
}
.ml-btn-cancel {
  border: 1px solid var(--border); background: var(--btn-bg); color: var(--btn-text);
}
.ml-btn-cancel:hover { background: var(--btn-hover); }
.ml-btn-ok {
  border: 1px solid var(--radio-checked-border); background: var(--radio-checked-bg);
  color: var(--radio-checked-color); font-weight: 600;
}
.ml-btn-ok:hover { opacity: 0.85; }
.ml-btn-danger {
  border: 1px solid #c0392b55; background: var(--btn-bg); color: #c0392b; font-weight: 600;
}
.ml-btn-danger:hover { background: #c0392b18; }
`;

  const styleEl = document.createElement("style");
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  function showDialog({ message, buttons }) {
    return new Promise(resolve => {
      const backdrop = document.createElement("div");
      backdrop.className = "ml-dialog-backdrop";

      const dlg = document.createElement("div");
      dlg.className = "ml-dialog";
      dlg.setAttribute("role", "dialog");
      dlg.setAttribute("aria-modal", "true");

      const msg = document.createElement("div");
      msg.className = "ml-dialog-message";
      msg.textContent = message;

      const actions = document.createElement("div");
      actions.className = "ml-dialog-actions";

      function close(value) {
        backdrop.remove();
        resolve(value);
      }

      backdrop.addEventListener("keydown", e => {
        if (e.key === "Escape") close(null);
      });

      for (const { label, value, cls } of buttons) {
        const btn = document.createElement("button");
        btn.className = `ml-btn ${cls}`;
        btn.textContent = label;
        btn.onclick = () => close(value);
        actions.appendChild(btn);
      }

      dlg.append(msg, actions);
      backdrop.appendChild(dlg);
      document.body.appendChild(backdrop);

      // Focus the primary (last) button
      actions.querySelector("button:last-child").focus();
    });
  }

  // mlConfirm(message, opts?) → Promise<boolean>
  // opts: { confirmLabel, cancelLabel, danger }
  window.mlConfirm = function (message, opts = {}) {
    const {
      confirmLabel = "Confirm",
      cancelLabel = "Cancel",
      danger = true,
    } = opts;
    return showDialog({
      message,
      buttons: [
        { label: cancelLabel, value: false, cls: "ml-btn-cancel" },
        { label: confirmLabel, value: true,  cls: danger ? "ml-btn-danger" : "ml-btn-ok" },
      ],
    });
  };

  // mlAlert(message, opts?) → Promise<void>
  // opts: { okLabel }
  window.mlAlert = function (message, opts = {}) {
    const { okLabel = "OK" } = opts;
    return showDialog({
      message,
      buttons: [{ label: okLabel, value: null, cls: "ml-btn-ok" }],
    });
  };
})();
