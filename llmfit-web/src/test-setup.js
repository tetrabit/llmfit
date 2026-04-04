import '@testing-library/jest-dom/vitest';

// Provide a minimal localStorage stub for jsdom.
// jsdom ≥26 may not expose a fully functional Storage on `window`.
// This ensures getItem/setItem/removeItem/clear are always available.
const storage = new Map();
const localStorageStub = {
  getItem: (key) => (storage.has(key) ? storage.get(key) : null),
  setItem: (key, value) => storage.set(key, String(value)),
  removeItem: (key) => storage.delete(key),
  clear: () => storage.clear(),
  get length() {
    return storage.size;
  },
  key: (index) => [...storage.keys()][index] ?? null,
};

Object.defineProperty(window, 'localStorage', { value: localStorageStub });
